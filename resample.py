#!/usr/bin/env python3
"""
Robust BSpline Resampling Pipeline for MRI (Production-Grade)

✅ True RAS ↔ LPS handling (CRITICAL FIX)
✅ Correct axis handling (SITK ↔ NumPy ↔ Nibabel)
✅ BSpline for images, NN for masks
✅ Accurate spacing/origin/direction
✅ No silent file skipping
"""

import os
import logging
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# -------------------------
# 🔁 NIB → SITK (RAS → LPS FIXED)
# -------------------------
def nib_to_sitk(nii):
    """
    Convert nibabel (RAS, XYZ) → SimpleITK (LPS, ZYX)
    """

    data = nii.get_fdata().astype(np.float32)  # (X, Y, Z)

    # ---- axis: XYZ → ZYX ----
    data_zyx = np.transpose(data, (2, 1, 0))
    sitk_img = sitk.GetImageFromArray(data_zyx)

    # ---- affine (RAS) ----
    affine_ras = nii.affine.copy()

    # ---- convert RAS → LPS ----
    ras2lps = np.diag([-1, -1, 1, 1])
    affine_lps = ras2lps @ affine_ras

    # ---- spacing ----
    spacing = np.linalg.norm(affine_lps[:3, :3], axis=0)

    # ---- direction ----
    direction = affine_lps[:3, :3] / spacing

    # ---- origin ----
    origin = affine_lps[:3, 3]

    sitk_img.SetSpacing(tuple(spacing))
    sitk_img.SetDirection(direction.flatten())
    sitk_img.SetOrigin(tuple(origin))

    return sitk_img


# -------------------------
# 🔁 SITK → NIB (LPS → RAS FIXED)
# -------------------------
def sitk_to_nib(sitk_img):
    """
    Convert SimpleITK (LPS, ZYX) → nibabel (RAS, XYZ)
    """

    # ---- array: ZYX → XYZ ----
    data_zyx = sitk.GetArrayFromImage(sitk_img)
    data_xyz = np.transpose(data_zyx, (2, 1, 0))

    spacing = np.array(sitk_img.GetSpacing())
    origin = np.array(sitk_img.GetOrigin())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)

    # ---- build LPS affine ----
    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin

    # ---- convert LPS → RAS ----
    lps2ras = np.diag([-1, -1, 1, 1])
    affine_ras = lps2ras @ affine_lps

    return nib.Nifti1Image(data_xyz.astype(np.float32), affine_ras)


# -------------------------
# 🔧 RESAMPLING
# -------------------------
def resample_image(sitk_img, target_spacing=(0.5, 0.5, 2.0), is_mask=False):
    """
    Resample image safely
    """

    original_spacing = np.array(sitk_img.GetSpacing())
    original_size = np.array(sitk_img.GetSize())

    new_size = np.round(original_size * (original_spacing / target_spacing)).astype(int)
    new_size = np.maximum(new_size, 8)

    resampler = sitk.ResampleImageFilter()

    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline
    )

    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize([int(x) for x in new_size])
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(sitk_img)


# -------------------------
# 🧠 PROCESS ONE FILE
# -------------------------
def process_single(in_path, out_path, target_spacing, force_ras=True):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        filename = os.path.basename(in_path)

        # ---- LOAD ----
        nii = nib.load(in_path)

        if force_ras:
            nii = nib.as_closest_canonical(nii)

        # ---- TO SITK ----
        sitk_img = nib_to_sitk(nii)

        orig_shape = sitk_img.GetSize()
        orig_spacing = sitk_img.GetSpacing()

        # ---- detect mask ----
        is_mask = "_mask" in filename.lower()

        # ---- RESAMPLE ----
        resampled = resample_image(
            sitk_img,
            target_spacing=target_spacing,
            is_mask=is_mask
        )

        new_shape = resampled.GetSize()
        new_spacing = resampled.GetSpacing()

        # ---- BACK TO NIB ----
        out_nii = sitk_to_nib(resampled)

        nib.save(out_nii, out_path)

        logging.info(
            f"✅ {filename} | "
            f"{orig_shape} → {new_shape} | "
            f"{orig_spacing} → {new_spacing}"
        )

        return True

    except Exception as e:
        logging.error(f"❌ Failed {in_path}: {e}", exc_info=True)
        return False


# -------------------------
# 📦 COLLECT FILES
# -------------------------
def collect_tasks(input_dir, output_dir):
    tasks = []

    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                in_path = os.path.join(root, f)

                rel = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel)

                if not out_path.endswith(".nii.gz"):
                    out_path = out_path.replace(".nii", ".nii.gz")

                tasks.append((in_path, out_path))

    logging.info(f"📊 Total files found: {len(tasks)}")
    return tasks


# -------------------------
# 🏁 MAIN
# -------------------------
def main():
    input_dir = ""
    output_dir = ""
    target_spacing = (0.5, 0.5, 2.0)

    logging.info("=" * 80)
    logging.info("🚀 MRI RESAMPLING (CORRECT RAS/LPS)")
    logging.info("=" * 80)

    tasks = collect_tasks(input_dir, output_dir)

    success, fail = 0, 0

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(process_single, in_p, out_p, target_spacing): in_p
            for in_p, out_p in tasks
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Resampling"):
            try:
                if fut.result():
                    success += 1
                else:
                    fail += 1
            except:
                fail += 1

    logging.info("=" * 80)
    logging.info(f"✅ Success: {success}")
    logging.info(f"❌ Fail: {fail}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()