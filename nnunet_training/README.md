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
  - `pip install -r nnunet_training/requirements.txt`

2) Merge datasets (interactive, once)
- Command:
  - `python nnunet_training/step0_merge_datasets.py`
- Checks (must pass):
  - `Dataset100_ULS23_Combined/` created with `imagesTr/`, `labelsTr/`
  - Filenames prefixed with `D031_..`, `D032_..` etc. (traceable, collision-free)
  - Labels consistent across source datasets (script aborts if not)
  - Report written: `/data/bodyct/experiments/nielsrocholl/ULS+/nnunet_training_logs/step0_report.json`

3) Planning (SLURM job 1)
- Submit: `sbatch nnunet_training/slurm/01_planning.sbatch`
- Checks after completion:
  - `dataset_fingerprint.json` exists in `nnUNet_preprocessed/Dataset100_ULS23_Combined/`
  - `nnUNetResEncUNetLPlans.json` exists in the same folder

4) Analyze and adjust plans (interactive)
- Command:
  - `python nnunet_training/step2_analyze_and_adjust.py`
- What it does:
  - Reads header shapes/spacings of raw images
  - Uses target spacing from plans to predict resampled sizes
  - Computes a patch size that covers the largest resampled volume (rounded to match pooling divisibility)
  - Creates configuration `3d_fullres_singlepass` in `nnUNetResEncUNetLPlans.json` with computed `patch_size` and `batch_size=1`
- Checks:
  - `patch_size_report.json` in `/data/bodyct/experiments/nielsrocholl/ULS+/nnunet_training_logs/`
  - Plans updated with `3d_fullres_singlepass`

5) Preprocessing (SLURM job 2)
- Submit: `sbatch nnunet_training/slurm/02_preprocessing.sbatch`
- Checks after completion:
  - `nnUNet_preprocessed/Dataset100_ULS23_Combined/3d_fullres_singlepass/` exists

6) Create custom split (interactive)
- Command:
  - `python nnunet_training/step4_create_splits.py`
- What it does:
  - Generates `splits_final.json` with 98% train / 2% val (stratified by dataset prefix)
- Checks:
  - `splits_final.json` placed in `nnUNet_preprocessed/Dataset100_ULS23_Combined/`

7) Training (SLURM job 3)
- Submit: `sbatch nnunet_training/slurm/03_training.sbatch`
- Behavior:
  - Copies preprocessed data to node-local scratch for speed
  - Trains fold `0` using `3d_fullres_singlepass` with `-p nnUNetResEncUNetLPlans`
  - Syncs results back to `nnUNet_results` on shared storage
- Checks during/after:
  - `progress.png`, `checkpoint_final.pth`, `validation/summary.json` under `nnUNet_results/Dataset100_ULS23_Combined/...`

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


