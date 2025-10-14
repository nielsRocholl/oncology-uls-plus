#!/bin/bash
#SBATCH --job-name=nnunet_preprocess
#SBATCH --qos=vram
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/bodyct/experiments/nielsrocholl/:/data/bodyct/experiments/nielsrocholl/
#SBATCH --container-image="dodrio1.umcn.nl#uokbaseimage/diag:latest"

set -euo pipefail

cd /home/nielsrocholl/projects/git_projects/oncology-uls-plus

python -m pip install -U pip
python -m pip install -r nnunet_training/requirements.txt

export nnUNet_raw=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw
export nnUNet_preprocessed=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_preprocessed
export nnUNet_results=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_results

# Ensure edited plans are in place under nnUNet_preprocessed
if [ ! -f "$nnUNet_preprocessed/Dataset100_ULS23_Combined/nnUNetResEncUNetLPlans.json" ]; then
  echo "ERROR: Edited plans file not found in nnUNet_preprocessed. Please complete step 4."
  exit 1
fi

nnUNetv2_preprocess -d 100 -c 3d_fullres_singlepass -p nnUNetResEncUNetLPlans --verify_dataset_integrity

echo "Preprocessing complete."


