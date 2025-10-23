## ULS+ inference (nnU-Net v2 CLI) and evaluation

### 1) Set environment variables
```bash
export nnUNet_raw=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw
export nnUNet_results=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_resultsv2
```

### 2) Run inference to shared storage
```bash
nnUNetv2_predict \
  -i /data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/imagesTr \
  -o /data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/preds \
  -d Dataset090_ULS23_Combined \
  -c 3d_fullres_singlepass \
  -p nnUNetResEncUNetLPlans \
  -f all \
  --checkpoint_name checkpoint_best.pth
```
If your build expects a different checkpoint flag, use `--chk best` instead.

Notes:
- The test dataset.json `numTraining` value does not affect inference.
- Input filenames must mirror training format (`*_0000.nii.gz`).

### 3) Run evaluation and write CSV
```bash
python nnunet_training/pipelines/eval_uls.py \
  --dataset-root /data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128 \
  --preds        /data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/preds \
  --out          /data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/uls_metrics.csv
```

The CSV contains per-type and overall Dice/Boundary IoU and agreement (mean pairwise Dice/Boundary IoU among normal/aug1/aug2).
