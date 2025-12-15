import sys

sys.path += ['.', '../substwsi', '../SubstWSI_rombek', '..']

import hydra
from omegaconf import DictConfig, OmegaConf
import sys

from multilang_wsi_evaluation.vectorizer_evaluator import evaluate_vectorizer


@hydra.main(config_path='conf')
def run(cfg: DictConfig):
    """
        --config-path path_to_conf --config-name config
    """
    print(OmegaConf.to_yaml(cfg))
    return evaluate_vectorizer(
        cfg.model_vec, cfg.model_clusterers, cfg.eval_params, cfg.run_settings
    )


if __name__ == '__main__':
    run()

# --config-name
# eval_v1.yaml
# model_vec=subst_rombek_cwm
# model_vec.vec_config.top_k=5,20,40,60,80,100,120,140
# model_vec.vec_config.min_df=0
# model_vec.vec_config.max_df=1
# +runname=cwm_experiment
# --multirun
#
# --config-name
# eval_v1.yaml
# model_vec=subst_rombek
# model_vec.subst_params.mask_count=[1],[2],[1,2,3]
# model_vec.vec_config.top_k=5,20,40,60,80,100,120,140
# +runname=experiment2404
# --multirun