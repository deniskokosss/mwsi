#!/bin/bash

# Table 2
for model in subst_123_or_even
do
  HYDRA_FULL_ERROR=1 .venv/bin/python multilang_wsi_evaluation/evaluate_vectorizer.py \
   --config-name eval_table2 \
    model_vec=$model \
    +runname=test_evaluation --multirun

done


# Table 3
for model in subst_123_or_even
do
  HYDRA_FULL_ERROR=1 .venv/bin/python multilang_wsi_evaluation/evaluate_vectorizer.py \
   --config-name eval_table3 \
    model_vec=$model \
    +runname=test_evaluation --multirun
done