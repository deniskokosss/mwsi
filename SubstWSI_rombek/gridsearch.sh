#!/bin/sh
date

SCRIPT_PATH='SubstWSI_rombek/gridsearch_patterns.py'
SUMMER_WSI_PATH=$(dirname $(pwd))
echo $SUMMER_WSI_PATH

export PYTHONPATH="${PYTHONPATH}:${SUMMER_WSI_PATH}"
cd $SUMMER_WSI_PATH || exit

while getopts l:n: flag
do
    case "${flag}" in
        l) lang=${OPTARG};;
        *) echo "WRONG FLAG" && exit 1
    esac
done

python -u $SCRIPT_PATH --config-name="eval_${lang}" model_vec=subst_rombek

