### nnUNetv2 ResEnc L training (single-pass) for ULS23 Combined

This pipeline trains an nnUNetv2 Residual Encoder UNet (L) on a merged dataset (Dataset100_ULS23_Combined) composed of datasets 031–050. We accept nnUNet resampling and enforce single-pass processing by setting the patch size to cover the entire resampled volume.

Paths (persistent):
- nnUNet raw: `/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw`
- nnUNet preprocessed: `/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_preprocessed`
- nnUNet results: `/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_results`

Project code (mounted by default):
- `/home/nielsrocholl/projects/git_projects/oncology-uls-plus`

SLURM container mount:
- `--container-mounts=/data/bodyct/experiments/nielsrocholl/:/data/bodyct/experiments/nielsrocholl/`

---

## Quick start

1) Prepare environment (on main node)
- Ensure datasets 031–050 are present under `nnUNet_raw`.
- Install local Python deps for helper scripts:

```bash
pip3 install -r nnunet_training/requirements.txt
```

- Checks (must pass):
  - Labels consistent across source datasets (script aborts if not)
  - Report written: `/data/bodyct/experiments/nielsrocholl/ULS+/nnunet_training_logs/step0_report.json`

3) Planning (SLURM job 1)
- Submit:

```bash
sbatch nnunet_training/slurm/01_planning.sbatch
```
- Checks after completion:
  - `dataset_fingerprint.json` exists in `nnUNet_preprocessed/Dataset100_ULS23_Combined/`
  - `nnUNetResEncUNetLPlans.json` exists in the same folder

4) Analyze and adjust plans (interactive)
- Command:

```bash
python nnunet_training/step2_analyze_and_adjust.py
```
- What it does:
  - Reads header shapes/spacings of raw images
  - Uses target spacing from plans to predict resampled sizes
  - Computes a patch size that covers the largest resampled volume (rounded to match pooling divisibility)
  - Creates configuration `3d_fullres_singlepass` in `nnUNetResEncUNetLPlans.json` with computed `patch_size` and `batch_size=1`
- Checks:
  - `patch_size_report.json` in `/data/bodyct/experiments/nielsrocholl/ULS+/nnunet_training_logs/`
  - Plans updated with `3d_fullres_singlepass`

5) Preprocessing (SLURM job 2)
- Submit:

```bash
sbatch nnunet_training/slurm/02_preprocessing.sbatch
```
- Checks after completion:
  - `nnUNet_preprocessed/Dataset100_ULS23_Combined/3d_fullres_singlepass/` exists

6) Create custom split (interactive)
- Command:

```bash
python nnunet_training/step4_create_splits.py
```
- What it does:
  - Generates `splits_final.json` with 98% train / 2% val (stratified by dataset prefix)
- Checks:
  - `splits_final.json` placed in `nnUNet_preprocessed/Dataset100_ULS23_Combined/`

7) Training (SLURM job 3)
- Submit:

```bash
sbatch nnunet_training/slurm/03_training.sbatch
```
- Behavior:
  - Copies preprocessed data to node-local scratch for speed
  - Trains fold `0` using `3d_fullres_singlepass` with `-p nnUNetResEncUNetLPlans`
  - Writes results (checkpoints/weights) directly to `nnUNet_results` (shared)
- Checks during/after:
  - `progress.png`, `checkpoint_final.pth`, `validation/summary.json` under `nnUNet_results/Dataset100_ULS23_Combined/...`

---

## Maximize GPU utilization (A100)

- Keep single-pass `patch_size` fixed. Increase `batch_size` in a derived config to fill VRAM:
  - Add in plans under `configurations`:
    - `"3d_fullres_singlepass_bs4": { "inherits_from": "3d_fullres_singlepass", "batch_size": 4 }`
  - Train:

```bash
nnUNetv2_train 100 3d_fullres_singlepass_bs4 0 -p nnUNetResEncUNetLPlans
```
  - If OOM, try bs=2, then bs=1.
- Optional: multi-GPU

```bash
nnUNetv2_train ... -num_gpus N
```
  - Make `batch_size` divisible by N.
- Optional: planning with higher VRAM target
 
```bash
nnUNetv2_plan_experiment -d 100 -pl nnUNetPlannerResEncL -gpu_memory_target 40 -overwrite_plans_name nnUNetResEncUNetLPlans_40G
```
  - Only useful if you let planner choose `patch_size`. We override it for single-pass.
- Mixed precision is on by default; preprocessed data already uses node-local scratch.

---

## Configuration

Edit `nnunet_training/config.json` if needed. Defaults:
- `dataset_id`: 100
- `dataset_name`: "ULS23_Combined"
- `raw_root`, `preprocessed_root`, `results_root`: point to `/data/bodyct/experiments/nielsrocholl/ULS+/...`
- `source_dataset_ids`: [31..50]
- `validation_fraction`: 0.02
- `configuration_name`: "3d_fullres_singlepass"
- `plans_name`: "nnUNetResEncUNetLPlans"
- `planner`: "nnUNetPlannerResEncL"
- `safety_margin_percent`: 5

---

## Notes
- Single-pass is achieved by computing a `patch_size` that covers the full resampled volume (may set batch size to 1).
- Residual connections and size L are provided by the ResEnc L plans.
- All outputs are persisted under `/data/bodyct/experiments/nielsrocholl/ULS+/`.


