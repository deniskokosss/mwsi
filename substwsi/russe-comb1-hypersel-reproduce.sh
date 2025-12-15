#!/bin/bash
date
nvidia-smi
for bm in True False ; do
substs=russe_bts-rnc/train_1-limitNone-maxexperwordNone-maxlen98/modelNone-beamsearch${bm}/'<mask><mask><mask>-(а-также-T)-2ltr3f_topk500_fixspacesTrue.npz+0.0+'
echo $substs
python max_ari.py $substs russe_bts-rnc/train -dump_images False -min_dfs [0.01,0.02,0.03,0.05,0.07,0.1] 
done 

echo Best results for each metric shall be: \ 
"russe_bts-rnc/train_1-limitNone-maxexperwordNone/modelNone-beamsearchFalse/<mask><mask>-(а-также-T)-2ltr2f_topk150_fixspacesTrue.npz+0.0+_dump/dump_general.sdfs.maxari:45" \
"cosine       average     3      0.6035279402154234   0.6597993671837965   0.5148489539355314   0.4911161233437057   128   TfidfVectorizer  0.03    0.9" \
"cosine       average     3      0.5679450783809974   0.6837446012224606   0.5173620802273594   0.5173620802273594   128   TfidfVectorizer  0.05    0.98" \
"cosine       average     3      0.5819531302968249   0.66397860342814     0.5677182234464829   0.5039262186084587   256   CountVectorizer  0.05    0.8" \

date
