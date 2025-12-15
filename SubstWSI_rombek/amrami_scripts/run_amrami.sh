#!/bin/bash
date
nvidia-smi

SCRIPT_PATH='multilang_wsi_evaluation/evaluate_vectorizer.py'
SUMMER_WSI_PATH=$(dirname $(dirname $(pwd)))

export PYTHONPATH="${PYTHONPATH}:${SUMMER_WSI_PATH}"
cd $SUMMER_WSI_PATH || exit


while getopts m:l:v:r:o: flag
do
    case "${flag}" in
      r) runs=${OPTARG};;
      m) model_path=${OPTARG};;
      l) lemmatizer=${OPTARG};;
      v) vocab_size=${OPTARG};;
      o) output_path=${OPTARG};;
      *) echo "WRONG FLAG" && exit 1
    esac
done

for ((i=1;i<=runs;i++));
do
HYDRA_FULL_ERROR=1 OUTPUT_PATH=$output_path python ${SCRIPT_PATH} --config-name eval_amrami \
    model_vec=subst_amrami model_vec.substs_overrides.model_path=$model_path model_vec.lemma_config.lemmatizer_name=$lemmatizer \
    model_vec.vec_config.vocab_size=$vocab_size run_settings.random_state=$RANDOM \
    hydra.run.dir='outputs/${oc.env:OUTPUT_PATH}/${now:%Y-%m-%d_%H-%M-%S}' hydra.sweep.dir='outputs/${oc.env:OUTPUT_PATH}/${now:%Y-%m-%d_%H-%M-%S}'
done
