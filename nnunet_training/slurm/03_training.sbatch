#!/bin/bash
#SBATCH --job-name=nnunet_train
#SBATCH --qos=vram
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/bodyct/experiments/nielsrocholl/:/data/bodyct/experiments/nielsrocholl/
#SBATCH --container-image="dodrio1.umcn.nl#uokbaseimage/diag:latest"

set -euo pipefail

cd /home/nielsrocholl/projects/git_projects/oncology-uls-plus

python3 -m pip install -U pip
python3 -m pip install -r nnunet_training/requirements.txt

mkdir -p /nnUNet_local
mkdir -p /data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_results

mkdir -p /nnUNet_local/nnUNet_preprocessed

# Set nnU-Net environment paths
export nnUNet_raw=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw
export nnUNet_preprocessed=/nnUNet_local/nnUNet_preprocessed
export nnUNet_results=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_results

# Preprocess locally (Dataset 90)
nnUNetv2_preprocess -d 90 -c 3d_fullres_singlepass -p nnUNetResEncUNetLPlans -np 16

# Run training (Dataset 90, 3d_fullres_singlepass, fold 0)
nnUNetv2_train 90 3d_fullres_singlepass 0 -p nnUNetResEncUNetLPlans --npz

# Archive and copy preprocessed to shared as last step, then cleanup
if command -v pigz >/dev/null 2>&1; then
    tar -I "pigz -p 16" -cf "/tmp/nnUNet_preprocessed.tar.gz" -C "/nnUNet_local" "nnUNet_preprocessed"
else
    tar -czf "/tmp/nnUNet_preprocessed.tar.gz" -C "/nnUNet_local" "nnUNet_preprocessed"
fi
cp -f "/tmp/nnUNet_preprocessed.tar.gz" "/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_preprocessed/"
rm -f "/tmp/nnUNet_preprocessed.tar.gz" || true
rm -rf "/nnUNet_local/nnUNet_preprocessed"


