#!/bin/bash
#SBATCH --gres=gpu:1
date
nvidia-smi

SCRIPT_PATH='multilang_wsi_evaluation/evaluate_vectorizer.py'
SUMMER_WSI_PATH=$(dirname $(dirname $(pwd)))

export PYTHONPATH="${PYTHONPATH}:${SUMMER_WSI_PATH}"
cd $SUMMER_WSI_PATH || exit


while getopts r:o: flag
do
    case "${flag}" in
      r) runs=${OPTARG};;
      o) output_path=${OPTARG};;
      *) echo "WRONG FLAG" && exit 1
    esac
done

for ((i=1;i<=runs;i++));
do
HYDRA_FULL_ERROR=1 OUTPUT_PATH=$output_path python ${SCRIPT_PATH} --config-name eval_amrami_multilang model_vec=subst_amrami_multilang \
    hydra.run.dir='outputs/${oc.env:OUTPUT_PATH}/${now:%Y-%m-%d_%H-%M-%S}' hydra.sweep.dir='outputs/${oc.env:OUTPUT_PATH}/${now:%Y-%m-%d_%H-%M-%S}' \
    run_settings.random_state=$RANDOM

done

date
