import os
import sys
import itertools
import typing as tp
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod

import yaml
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from multilang_wsi_evaluation import utils
from multilang_wsi_evaluation.interfaces import IWSI
from multilang_wsi_evaluation.vectorizer_evaluator import prepare_semeval_metrics
from multilang_wsi_evaluation.wsi_metrics import compute_mean_metrics, compute_unsup_metrics
from multilang_wsi_evaluation.wsi_vectorizers import VectorizerWSI, AdaptiveClusterer
from multilang_wsi_evaluation.clusterers_models import BaseClusterer


class ParamsStrategy(ABC):
    @abstractmethod
    def create_clusterer(self, dataset_name: str, metric_name: str, method_output_dir: Path) -> AdaptiveClusterer:
        pass

    @property
    def name(self) -> str:
        return self.__str__()


class FhMaxStrategy(ParamsStrategy):
    def create_clusterer(self, dataset_name: str, metric_name: str, method_output_dir: Path) -> AdaptiveClusterer:
        metric_files = list(method_output_dir.glob(f'*/agg_metrics/fh_max_{metric_name}.tsv'))
        assert len(metric_files) == 1

        metric_df = pd.read_csv(metric_files[0], sep='\t')
        metric_df = metric_df[metric_df['part'] == dataset_name]
        assert len(metric_df) == 1
        max_params: dict = eval(metric_df.iloc[0]['hypers'])

        clusterer_type = utils.load_obj(max_params['@clusterer'])
        metric = load_dist_from_merged(max_params['@metric'])
        max_params.pop('@clusterer')
        max_params.pop('@metric')
        max_params = {param: [value] for param, value in max_params.items()}

        def compute_unsup_metric(*args, **kwargs) -> float:
            # unsup metric does not matter since every parameter has only 1 value
            return 0.

        base_clusterer = BaseClusterer(clusterer_type, [metric], max_params, None, None)  # type: ignore
        return AdaptiveClusterer(base_clusterer, compute_unsup_metric)

    @property
    def name(self) -> str:
        return 'fh_max'


class MaxUnsupStrategy(ParamsStrategy):
    def create_clusterer(self, dataset_name: str, metric_name: str, method_output_dir: Path) -> AdaptiveClusterer:
        dataset_files = list(method_output_dir.glob(f'*/agg_metrics/slice_unsup/{dataset_name}.tsv'))
        assert len(dataset_files) == 1

        dataset_df = pd.read_csv(dataset_files[0], sep='\t')
        dataset_df = dataset_df[dataset_df['best_by'] == metric_name]
        assert len(dataset_df) == 1
        max_params: dict = eval(dataset_df.iloc[0]['hypers'])

        def create_unsup_metric(unsup_metric_str: str) -> tp.Callable:
            assert unsup_metric_str in ['calinski_harabasz', 'silhouette']

            def compute_unsup_metric(*args, **kwargs) -> float:
                return compute_unsup_metrics(*args, **kwargs)[unsup_metric_str]

            return compute_unsup_metric

        unsup_metric_name = max_params['@unsup_metric']
        unsup_metric = create_unsup_metric(unsup_metric_name)
        max_params.pop('@unsup_metric')
        assert len(max_params) == 1
        clusterer_name, clusterer_params = next(iter(max_params.items()))
        clusterer_type = utils.load_obj(clusterer_name)
        metrics = [load_dist_from_merged(dist) for dist in clusterer_params['@metric']]
        clusterer_params.pop('@metric')

        base_clusterer = BaseClusterer(clusterer_type, metrics, clusterer_params, None, None)  # type: ignore
        return AdaptiveClusterer(base_clusterer, unsup_metric, unsup_metric_name)

    @property
    def name(self) -> str:
        return 'max_unsup'


def load_yaml_config(config_path: Path) -> tp.Any:
    assert config_path.exists()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_dist_from_merged(dist: str) -> tp.Optional[tp.Union[str, tp.Callable]]:
    if dist is None:
        return None
    try:
        return utils.load_obj(dist)
    except:
        return dist


def compute_part_mean_metrics(wsi_clusterer: IWSI, part: utils.WSIDatasetPart) -> tp.Dict[str, float]:
    group_ids, groups_samples_ids, groups_samples = part.groups_samples()
    samples_ids = [sample_id for group_samples_ids in groups_samples_ids for sample_id in group_samples_ids]
    samples = [sample for group_samples in groups_samples for sample in group_samples]
    clusters = wsi_clusterer.predict(samples)
    id_to_pred = dict(zip(samples_ids, clusters))
    return compute_mean_metrics(part, id_to_pred)


def save_metrics(part_to_records: tp.Dict[str, tp.List[tp.Dict[str, tp.Any]]]) -> None:
    parts_path = Path('metrics')
    for part, part_records in part_to_records.items():
        part_path = parts_path / f'{part}.tsv'
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_df = pd.DataFrame.from_records(part_records)
        part_df.to_csv(part_path, index=False, sep='\t')


def evaluate_strategies(best_by: DictConfig, eval_params: DictConfig, run_settings: DictConfig) -> None:
    utils.set_random_state(run_settings.random_state)
    prepare_semeval_metrics(eval_params.use_fast_semeval_metrics)

    source_method_dir = Path(eval_params.source_method_dir)
    source_method_config = load_yaml_config(source_method_dir / '.hydra' / 'config.yaml')

    vectorizer = hydra.utils.instantiate(source_method_config['model_vec'])
    test_parts = list(utils.load_parts(*eval_params.test_datasets))
    params_strategies = [hydra.utils.instantiate(params_strategy) for params_strategy in best_by.params_strategies]
    part_to_records = defaultdict(list)
    info_file = sys.stdout if run_settings.verbose else open(os.devnull, 'w')

    for params_strategy in params_strategies:
        for part_name, metric_name in itertools.product(best_by.parts, best_by.metrics):
            clusterer = params_strategy.create_clusterer(part_name, metric_name, source_method_dir)
            wsi_pipeline = VectorizerWSI(vectorizer, clusterer, run_settings.verbose)
            params_info = {
                '@params_strategy': params_strategy.name,
                '@best_by_part': part_name,
                '@best_by_metric': metric_name,
            }
            print(f"Params: {params_info}", file=info_file)
            for test_part in test_parts:
                print(f"Evaluating '{test_part.id}'...", file=info_file)
                mean_metrics = compute_part_mean_metrics(wsi_pipeline, test_part)
                clusterer_info = {
                    '@unsup_metric': clusterer.unsup_metric_name,
                    '@clusterer': utils.type_full_name(clusterer.base_clusterer.clusterer_type),
                    '@dists': [utils.dist_full_name(dist) for dist in clusterer.base_clusterer.dists],
                    '@hypers': clusterer.base_clusterer.param_to_values
                }
                part_to_records[test_part.id].append({**params_info, **clusterer_info, **mean_metrics})

    save_metrics(part_to_records)


@hydra.main(config_path='conf', config_name='params_strategy_eval')
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    evaluate_strategies(
        cfg.best_by, cfg.eval_params, cfg.run_settings,
    )


if __name__ == '__main__':
    run()
