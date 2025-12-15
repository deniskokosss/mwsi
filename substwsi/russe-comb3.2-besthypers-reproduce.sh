#!/bin/bash
date
nvidia-smi

SUMMER_WSI_PATH=$(dirname $(pwd))

export PYTHONPATH="${PYTHONPATH}:${SUMMER_WSI_PATH}"

output_dir=substwsi/preds-bts-rnc
model=comb3.2
rm -rf ${output_dir}
#
for part in  train ; do
  substs=xlm/russe_bts-rnc/${part}_1-limitNone-maxexperwordNone/modelNone/"<mask>-или-T-2ltr2f_topk150_fixspacesTrue.npz+0.0+@1-3:0.0:1.0:0.0@"
  echo $substs
  python max_ari.py $substs russe_bts-rnc/${part} -preds_path ${output_dir}/${part}/${model}-fixnc.tsv -dump_images False -vectorizers [CountVectorizer] -topks [512] --min_dfs [0.05] --max_dfs [0.95] --ncs [3,4]

  python max_ari.py $substs russe_bts-rnc/${part} -preds_path ${output_dir}/${part}/${model}-silnc.tsv -dump_images False -vectorizers [CountVectorizer] -topks [512] --min_dfs [0.05] --max_dfs [0.95] --ncs [3,10]
done

python russe-wsi-kit/evaluate.py ${output_dir}/train/${model}-fixnc.tsv | tee ${output_dir}/official-train-fixnc.log
python russe-wsi-kit/evaluate.py ${output_dir}/train/${model}-silnc.tsv | tee ${output_dir}/official-train-silnc.log
echo '---------'
echo train fhmaxari, Shall be: 0.61
tail -2 ${output_dir}/official-train-fixnc.log
echo train silari, shall be: 0.63
tail -2 ${output_dir}/official-train-silnc.log

date
