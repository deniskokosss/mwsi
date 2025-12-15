#!/bin/sh
#SBATCH --gres=gpu:1
date

SCRIPT_PATH='SubstWSI_rombek/generate_substitutes.py'
SUMMER_WSI_PATH=$(dirname $(dirname $(realpath $0)))


export PYTHONPATH="${PYTHONPATH}:${SUMMER_WSI_PATH}"
cd $SUMMER_WSI_PATH || exit

SUBSTS_CFGS_PATH="${SUMMER_WSI_PATH}/multilang_wsi_evaluation/conf/lang_eval/model_vec/subst_params"
SUBSTS_OUT_BASE="SubstWSI_rombek/substs"

if [ -z "${MODEL_PATH}" ]
then
  echo "USING MODEL FROM FAIRSEQ CACHE"
  MODEL_PATH="None"
else
  echo "USING MODEL FROM ${MODEL_PATH}"
fi

for MASK_CNT in 1 2 3
do
  python -u $SCRIPT_PATH \
    --dataset_path="datasets/se20lscd_v2/de/sense-new.tsv" --substs_configs_path="${SUBSTS_CFGS_PATH}/de" --mask_number="${MASK_CNT}" --out_path="${SUBSTS_OUT_BASE}/de" \
    --model_path="${MODEL_PATH}"
  
  if [ $? -eq 1 ]
  then
    echo "ERROR occured"
    exit 0
  fi
  
  python -u $SCRIPT_PATH \
    --dataset_path="datasets/se10" --substs_configs_path="${SUBSTS_CFGS_PATH}/en" --mask_number="${MASK_CNT}" --out_path="${SUBSTS_OUT_BASE}/en" \
    --model_path="${MODEL_PATH}"
  if [ $? -eq 1 ]
  then
    echo "ERROR occured"
    exit 0
  fi
done; 

