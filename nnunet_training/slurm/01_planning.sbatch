#!/bin/bash
#SBATCH --job-name=nnunet_plan
#SBATCH --qos=vram
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
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

nnUNetv2_extract_fingerprint -d 100 --verify_dataset_integrity
nnUNetv2_plan_experiment -d 100 -pl nnUNetPlannerResEncL

# Copy artifacts for inspection (they also remain in nnUNet_preprocessed)
mkdir -p /home/nielsrocholl/projects/git_projects/oncology-uls-plus/nnunet_training/logs
cp /data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_preprocessed/Dataset100_ULS23_Combined/dataset_fingerprint.json \
   /home/nielsrocholl/projects/git_projects/oncology-uls-plus/nnunet_training/logs/
cp /data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_preprocessed/Dataset100_ULS23_Combined/nnUNetResEncUNetLPlans.json \
   /home/nielsrocholl/projects/git_projects/oncology-uls-plus/nnunet_training/logs/plans_original.json

echo "Planning complete. Review logs/ and edit plans before preprocessing."


