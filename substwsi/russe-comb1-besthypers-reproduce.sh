#!/bin/bash

date
nvidia-smi

SUMMER_WSI_PATH=$(dirname $(pwd))

export PYTHONPATH="${PYTHONPATH}:${SUMMER_WSI_PATH}"

output_dir=preds-bts-rnc
rm -rf ${output_dir}

for part in train test; do
substs=russe_bts-rnc/${part}_1-limitNone-maxexperwordNone/modelNone-beamsearchFalse/"<mask><mask>-(а-также-T)-2ltr2f_topk150_fixspacesTrue.npz+0.0+"
echo $substs
python max_ari.py $substs russe_bts-rnc/${part} -preds_path ${output_dir}/${part}/comb1-fixnc.tsv -dump_images False -vectorizers [TfidfVectorizer] -topks [128] --min_dfs [0.03] --max_dfs [0.9] --ncs [3,4]

python max_ari.py $substs russe_bts-rnc/${part} -preds_path ${output_dir}/${part}/comb1-silnc.tsv -dump_images False -vectorizers [CountVectorizer] -topks [128] --min_dfs [0.03] --max_dfs [0.8] --ncs [2,10]
done

python ../russe-wsi-kit/evaluate.py ${output_dir}/train/comb1-fixnc.tsv | tee ${output_dir}/official-train-fixnc.log
python ../russe-wsi-kit/evaluate.py ${output_dir}/train/comb1-silnc.tsv | tee ${output_dir}/official-train-silnc.log
echo '---------'
echo train fhmaxari, Shall be: 0.6
tail -2 ${output_dir}/official-train-fixnc.log
echo train silari, shall be: 0.58
tail -2 ${output_dir}/official-train-silnc.log

date
