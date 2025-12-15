from dataclasses import dataclass

import fire
from typing import List, Any, Optional
from utils import Sample, IWSIVectorizer, VectorizerEvaluatorCLIWrapper, vec_eval
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import inspect

class DummyVectorizer(IWSIVectorizer):
    # TODO: docs

    def __init__(self, vectors_dim: int = 128):
        self.dummy_vector = [0. for _ in range(vectors_dim)]

    def fit(self, all_samples):
        pass

    def predict(self, samples):
        return [self.dummy_vector for _ in range(len(samples))]


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    print(f'Config: {OmegaConf.to_yaml(cfg)}')
    model_vectorizer = instantiate(cfg.model_vec)
    param = dict(cfg.model_vec)
    del param['_target_']
    metric = cfg.return_value['metric']
    dataset = cfg.return_value['dataset']
    cfg_clusterer = cfg.model_clusterer
    val = vec_eval(model_vectorizer,
                   cfg_clusterer=cfg_clusterer,
                   params=param,
                   gold_datasets_dir="../../../../datasets",
                   vectorizers_metrics_dir="../../../../vectorizers_metrics",
                   opt_metric=metric,
                   opt_dataset=dataset)
    return np.mean(val)


if __name__ == '__main__':
    #fire.Fire(VectorizerEvaluatorCLIWrapper(DummyVectorizer).evaluate)
    run()
