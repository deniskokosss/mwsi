from pathlib import Path

import hydra
import omegaconf

from multilang_wsi_evaluation import evaluate_vectorizer


@hydra.main(config_path='conf')
def run(cfg: omegaconf.DictConfig):
    run_config = omegaconf.OmegaConf.load(Path(cfg.run_dir) / '.hydra' / 'config.yaml')
    with omegaconf.open_dict(cfg):
        cfg.pop('run_dir')
        cfg.model_vec = run_config.model_vec
    return evaluate_vectorizer.run(cfg)


if __name__ == '__main__':
    run()
