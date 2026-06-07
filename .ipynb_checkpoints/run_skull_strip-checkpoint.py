import os
import gc
import shutil
import tempfile
import nibabel as nib
import subprocess
import torch
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# 🔧 Environment fixes
# -------------------------
# -------------------------
# 🔧 HD-BET Cache Fix (PERSISTENT)
# -------------------------
CACHE_DIR = os.path.join(os.getcwd(), "hd_bet_weights")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HD_BET_CHECKPOINT_DIR"] = CACHE_DIR
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# -------------------------
# 🧹 Free GPU memory
# -------------------------
def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        logging.info("🧹 GPU cache cleared.")

def get_free_vram_mib():
    try:
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return free // (1024 * 1024)
    except Exception:
        pass
    return 0

# -------------------------
# 🧠 Force .nii.gz output
# -------------------------
def enforce_nii_gz(path):
    if path.endswith(".nii.gz"):
        return path
    elif path.endswith(".nii"):
        return path[:-4] + ".nii.gz"
    else:
        return path + ".nii.gz"

# -------------------------
# 📦 Collect valid tasks
# -------------------------
def collect_tasks(input_dir, output_dir, max_z_spacing=3.0):
    tasks = []

    for root, _, files in os.walk(input_dir):
        if "ds002242" in root or os.path.basename(root) == "DS":
            continue

        for f in files:
            if not f.endswith((".nii", ".nii.gz")):
                continue

            in_path = os.path.join(root, f)

            try:
                img = nib.load(in_path)
                spacing = img.header.get_zooms()

                if spacing[2] >= max_z_spacing:
                    continue

                rel_path = os.path.relpath(in_path, input_dir)
                out_path = enforce_nii_gz(
                    os.path.join(os.path.abspath(output_dir), rel_path)
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                if not os.path.exists(out_path):
                    tasks.append((os.path.abspath(in_path), out_path))

            except Exception as e:
                logging.warning(f"⚠️ Skipping {in_path}: {e}")

    return tasks

# -------------------------
# 🚀 Run HD-BET via isolated temp dir
# -------------------------
def run_hd_bet(task, device="cuda"):
    in_path, final_out_path = task

    assert final_out_path.endswith(".nii.gz"), \
        f"Output must end with .nii.gz, got: {final_out_path}"

    # Give HD-BET its own clean temp directory — it can do whatever it wants in there
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = os.path.join(tmp_dir, "output.nii.gz")

        cmd = ["hd-bet", "-i", in_path, "-o", tmp_out, "-device", device]
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode != 0:
            err = (res.stderr or res.stdout).strip().split('\n')[-1][:120]
            logging.error(f"❌ Failed ({device}): {os.path.basename(in_path)} | {err}")
            return False

        # HD-BET produces: output.nii.gz (brain) + output_mask.nii.gz (mask)
        brain_tmp = os.path.join(tmp_dir, "output.nii.gz")
        mask_tmp  = os.path.join(tmp_dir, "output_mask.nii.gz")

        if not os.path.exists(brain_tmp):
            logging.error(f"❌ Output not found after HD-BET: {brain_tmp}")
            return False

        # Move brain to final destination
        shutil.move(brain_tmp, final_out_path)

        # Move mask alongside if it exists
        if os.path.exists(mask_tmp):
            mask_final = final_out_path.replace(".nii.gz", "_mask.nii.gz")
            shutil.move(mask_tmp, mask_final)

    return True

# -------------------------
# 🏁 Main pipeline
# -------------------------
def main():
    input_dir  = ""
    output_dir = ""

    tasks = collect_tasks(input_dir, output_dir, max_z_spacing=3.0)
    logging.info(f"✅ Found {len(tasks)} valid files (z-spacing < 3mm).")

    if not tasks:
        logging.info("Nothing to process.")
        return

    # -------------------------
    # 🧹 Clear GPU memory
    # -------------------------
    free_gpu_memory()

    free_mib  = get_free_vram_mib()
    total_mib = 0
    try:
        if torch.cuda.is_available():
            _, total = torch.cuda.mem_get_info(0)
            total_mib = total // (1024 * 1024)
    except Exception:
        pass

    logging.info(f"🖥️  VRAM after clear: {free_mib} MiB free / {total_mib} MiB total")

    MIN_FREE_MIB = 2000
    VRAM_PER_JOB = 2000

    if torch.cuda.is_available() and free_mib >= MIN_FREE_MIB:
        device      = "cuda"
        max_workers = max(1, min(4, free_mib // VRAM_PER_JOB))
        logging.info(f"🚀 Using GPU ({max_workers} workers). Free VRAM: {free_mib} MiB")
    else:
        device      = "cpu"
        max_workers = 4
        logging.warning(
            f"⚠️  Only {free_mib} MiB free on GPU (need {MIN_FREE_MIB}) "
            f"→ CPU mode (4 workers)."
        )

    # -------------------------
    # 🔄 Process
    # -------------------------
    success, fail = 0, 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_hd_bet, t, device): t for t in tasks}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="file"):
            try:
                if fut.result():
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                logging.error(f"❌ Unexpected exception: {e}")
                fail += 1

    logging.info(f"\n📊 Done: ✅ {success} succeeded | ❌ {fail} failed")

if __name__ == "__main__":
    main()