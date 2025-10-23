set -euo pipefail

cd /home/nielsrocholl/projects/git_projects/oncology-uls-plus
python3 -m pip install -r nnunet_training/requirements.txt

DATASET_ID=90
CPUS=16

export nnUNet_raw=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw
export nnUNet_preprocessed=/nnunet_local_/nnUNet_preprocessed
export nnUNet_results=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_results
SHARED_PREPROCESSED=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_preprocessed

mkdir -p "${nnUNet_preprocessed}"

ds_padded=$(printf "%03d" ${DATASET_ID})
DATASET_NAME="Dataset${ds_padded}_ULS23_Combined"
if [ -d "${SHARED_PREPROCESSED}/${DATASET_NAME}" ]; then
  cp -a "${SHARED_PREPROCESSED}/${DATASET_NAME}" "${nnUNet_preprocessed}/"
fi

nnUNetv2_preprocess -d ${DATASET_ID} -c 3d_fullres_singlepass -p nnUNetResEncUNetLPlans -np ${CPUS}

ARCHIVE=/tmp/nnUNet_preprocessed.tar.gz
if command -v pigz >/dev/null 2>&1; then
  tar -I "pigz -p ${CPUS}" -cf "${ARCHIVE}" -C "$(dirname "${nnUNet_preprocessed}")" "$(basename "${nnUNet_preprocessed}")"
else
  tar -czf "${ARCHIVE}" -C "$(dirname "${nnUNet_preprocessed}")" "$(basename "${nnUNet_preprocessed}")"
fi

cp -f "${ARCHIVE}" "${SHARED_PREPROCESSED}/"