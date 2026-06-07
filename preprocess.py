#!/usr/bin/env python3
"""
Bias Correction + Z-Score Normalization for T2-weighted MRI
✅ Orientation-safe (SITK ↔ nibabel fixed)
✅ Optional canonical RAS alignment
"""

import os
import logging
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# -------------------------
# 🔧 Z-SCORE NORMALIZATION
# -------------------------
def normalize_zscore_stn(data, clip_sigma=3, low_mask_pct=2, high_mask_pct=98):
    brain_mask = (data > np.percentile(data, low_mask_pct)) & \
                 (data < np.percentile(data, high_mask_pct))

    if np.sum(brain_mask) < 1000:
        brain_mask = data > np.percentile(data, 5)

    mu = np.mean(data[brain_mask])
    sigma = np.std(data[brain_mask]) + 1e-8

    z = (data - mu) / sigma
    z = np.clip(z, -clip_sigma, clip_sigma)

    normalized = (z + clip_sigma) / (2 * clip_sigma)
    return normalized.astype(np.float32)


# -------------------------
# 🧠 CORE PROCESSING
# -------------------------
def process_single_image(in_path, out_path, force_ras=True):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        filename = os.path.basename(in_path)

        # ---- LOAD WITH NIBABEL ----
        nii = nib.load(in_path)

        # Optional: enforce canonical orientation (recommended)
        if force_ras:
            nii = nib.as_closest_canonical(nii)

        original_affine = nii.affine.copy()
        original_header = nii.header.copy()

        # ---- LOAD IN SITK (for N4) ----
        sitk_img = sitk.ReadImage(in_path, sitk.sitkFloat32)

        # ---- N4 BIAS CORRECTION ----
        transformed = sitk.RescaleIntensity(sitk_img, 0, 255)
        mask = sitk.LiThreshold(transformed, 0, 1)

        shrink = 4
        img_small = sitk.Shrink(sitk_img, [shrink]*3)
        mask_small = sitk.Shrink(mask, [shrink]*3)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([30, 20, 10])
        _ = corrector.Execute(img_small, mask_small)

        log_bias = corrector.GetLogBiasFieldAsImage(sitk_img)
        corrected = sitk_img / sitk.Exp(log_bias)

        # ---- CONVERT TO NUMPY (FIXED) ----
        corrected_np = sitk.GetArrayFromImage(corrected)   # (Z, Y, X)
        corrected_np = np.transpose(corrected_np, (2, 1, 0))  # → (X, Y, Z)

        # ---- NORMALIZATION ----
        norm = normalize_zscore_stn(corrected_np)

        # ---- SAVE WITH CORRECT ORIENTATION ----
        output_img = nib.Nifti1Image(norm, original_affine, original_header)
        output_img.set_data_dtype(np.float32)
        nib.save(output_img, out_path)

        logging.info(f"✅ {filename} | shape={norm.shape} | range=[{norm.min():.3f}, {norm.max():.3f}]")
        return True

    except Exception as e:
        logging.error(f"❌ Failed {os.path.basename(in_path)}: {e}", exc_info=True)
        return False


# -------------------------
# 📦 TASK COLLECTION
# -------------------------
def collect_tasks(input_dir, output_dir):
    tasks = []

    for root, _, files in os.walk(input_dir):
        for f in files:
            if not (f.endswith(".nii") or f.endswith(".nii.gz")):
                continue
            if "_mask" in f.lower():
                continue

            in_path = os.path.join(root, f)
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)

            if not out_path.endswith(".nii.gz"):
                if out_path.endswith(".nii"):
                    out_path = out_path.replace(".nii", ".nii.gz")
                else:
                    out_path += ".nii.gz"

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if not os.path.exists(out_path):
                tasks.append((in_path, out_path))

    return tasks


# -------------------------
#  MAIN
# -------------------------
def main():
    input_dir = ""
    output_dir = ""

    logging.info(f"🔍 Scanning: {input_dir}")
    tasks = collect_tasks(input_dir, output_dir)
    logging.info(f"📋 Found {len(tasks)} files")

    if not tasks:
        logging.info("✅ Nothing to process")
        return

    max_workers = 2
    success, fail = 0, 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_image, in_p, out_p): (in_p, out_p)
            for in_p, out_p in tasks
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="file"):
            in_p, _ = futures[fut]
            try:
                if fut.result():
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                logging.error(f"❌ Unexpected error: {in_p} | {e}")
                fail += 1

    logging.info(f"\n📊 Done: ✅ {success} | ❌ {fail}")
    logging.info(f"📁 Output: {output_dir}")


if __name__ == "__main__":
    main()