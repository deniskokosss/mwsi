# Multilingual Substitution-based Word Sense Induction

This repository contains code to reproduce the results from [our paper Multilingual Substitution-based Word Sense Induction](https://aclanthology.org/2024.lrec-main.1035.pdf).

### How to reproduce
1. Install uv if you haven't already
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Prepare environment
```bash
bash install.sh
```
3. Run prediction and evaluation for our best model "Concat SDP". Requires GPU, ~24 hours of runtime
```
source .vevn/bin/activate && bash run.sh
```
4. If the run is succesfull, the results can be found at VectorizingPipeline/agg_metrics/calinski_harabasz_ARI.tsv , ..._S10_AVG.tsv and _S13_AVG.tsv
