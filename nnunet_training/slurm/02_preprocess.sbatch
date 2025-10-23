#!/bin/bash
#SBATCH --job-name=nnunet_preproc
#SBATCH --qos=vram
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/bodyct/experiments/nielsrocholl/:/data/bodyct/experiments/nielsrocholl/
#SBATCH --container-image="dodrio1.umcn.nl#uokbaseimage/diag:latest"

set -euo pipefail

cd /home/nielsrocholl/projects/git_projects/oncology-uls-plus

python3 -m pip install -U pip
python3 -m pip install -r nnunet_training/requirements.txt

# Install rclone if not already available
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    mkdir -p ~/bin
    cd ~/bin
    wget -q https://downloads.rclone.org/rclone-current-linux-amd64.zip
    unzip -q rclone-current-linux-amd64.zip
    cp rclone-*/rclone . && chmod +x rclone
    rm -rf rclone-* rclone-current-linux-amd64.zip
    export PATH="$HOME/bin:$PATH"
    echo "rclone installation complete"
fi

export nnUNet_raw=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw
# write preprocessed to node-local to avoid storage permission issues
export nnUNet_preprocessed=/nnunet_local_${SLURM_JOB_ID}/nnUNet_preprocessed
export nnUNet_results=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_results

# Process combined dataset 100
DATASET_ID=90

mkdir -p /home/nielsrocholl/projects/git_projects/oncology-uls-plus/nnunet_training/logs

ds_padded=$(printf "%03d" ${DATASET_ID})
echo "Preprocessing Dataset${ds_padded} (3d_fullres_singlepass)"
nnUNetv2_preprocess -d ${DATASET_ID} -c 3d_fullres_singlepass -p nnUNetResEncUNetLPlans -np $SLURM_CPUS_PER_TASK 2>&1 | tee /home/nielsrocholl/projects/git_projects/oncology-uls-plus/nnunet_training/logs/preprocess_${ds_padded}.log

echo "Preprocessing complete. Syncing ALL local preprocessed to storage with rclone..."
rclone copy "${nnUNet_preprocessed}/" \
  "/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_preprocessed" \
  --progress --multi-thread-streams=32 --transfers=32

echo "Sync complete. Cleaning up node-local preprocessed..."
rm -rf "${nnUNet_preprocessed}"
echo "Cleanup complete."


