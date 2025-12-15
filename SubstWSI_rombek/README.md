# SubstWSI_rombek vectorizer

### Generate substitutes for one dataset (same language datasets)
Run generation script:
```
python generate_substitutes.py --dataset_path $DATA_PATH --substs_configs_path $CONFIGS_PATH --out_path $OUT_PATH --mask_number $MASK_NUMBER --model_path $MODEL_PATH
```
* ``$DATA_PATH`` - path to the data dir (one of ``datasets`` subdir).
* ``$CONFIGS_PATH`` - path to the dir with all base (1mask) configs
* ``$OUT_PATH`` - path to result substitutions
* ``$MASK_NUMBER`` - number of mask using in template.
* ``$MODEL_PATH`` - path to fairseq model (usually xlm-r)

### Generate 1,2,3-masks substitutes for se10-en and se20lscd_v2/de/sense-new 
If you need to use your model for generation, set ``MODEL_PATH`` env variable.

* Run bash script (using your model):
```
MODEL_PATH="path/to/your/model" bash generate_en_de_substs.sh
```

or

* Run bash script (using model from cache):
```
bash generate_en_de_substs.sh
```

* Result substs will be saved in dir `summer-wsi/SubstWSI_rombek/substs`

* New substs configs may be added in dir 
```
summer-wsi/multilang_wsi_evaluation/conf/lang_eval/model_vec/subst_params
```