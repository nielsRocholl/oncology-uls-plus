
export nnUNet_raw=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw
# use node-local path for preprocessed to avoid storage permission issues
export nnUNet_preprocessed=/nnunet_local_${SLURM_JOB_ID}/nnUNet_preprocessed
export nnUNet_results=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_results

set -e

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

# Ensure local preprocessed root exists
mkdir -p "${nnUNet_preprocessed}"

# Extract fingerprints with integrity check and parallelism
nnUNetv2_extract_fingerprint -d 90 -np 8 2>&1 | tee /home/nielsrocholl/projects/git_projects/oncology-uls-plus/nnunet_training/logs/step1_integrity.log

# Plan experiments
nnUNetv2_plan_experiment -d 90 -pl nnUNetPlannerResEncL

echo "Planning complete. Syncing ALL local preprocessed to storage with rclone..."
rclone copy "${nnUNet_preprocessed}/" \
  "/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_preprocessed" \
  --progress --multi-thread-streams=6 --transfers=8