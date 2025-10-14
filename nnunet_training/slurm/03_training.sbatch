#!/bin/bash
#SBATCH --job-name=nnunet_train
#SBATCH --qos=vram
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/bodyct/experiments/nielsrocholl/:/data/bodyct/experiments/nielsrocholl/
#SBATCH --container-image="dodrio1.umcn.nl#uokbaseimage/diag:latest"

set -euo pipefail

cd /home/nielsrocholl/projects/git_projects/oncology-uls-plus

python -m pip install -U pip
python -m pip install -r nnunet_training/requirements.txt

# Shared (source/destination)
SRC_PP=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_preprocessed/Dataset100_ULS23_Combined
DST_RES=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_results

# Node-local scratch
LOCAL_ROOT=/nnunet_local_${SLURM_JOB_ID}
LOCAL_PP=${LOCAL_ROOT}/nnUNet_preprocessed/Dataset100_ULS23_Combined

mkdir -p "$LOCAL_PP"

echo "Copying preprocessed data to node-local scratch..."
rsync -a --delete "$SRC_PP/" "$LOCAL_PP/"

# Point nnUNet to local scratch for speed
export nnUNet_raw=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw
export nnUNet_preprocessed=${LOCAL_ROOT}/nnUNet_preprocessed
# Write results directly to shared to avoid loss on crash
mkdir -p ${DST_RES}/Dataset100_ULS23_Combined
export nnUNet_results=${DST_RES}

echo "Starting training..."
nnUNetv2_train 100 3d_fullres_singlepass 0 -p nnUNetResEncUNetLPlans --npz

echo "Training complete."


