import itertools
import os
import re
import sys
import typing as tp
from collections import defaultdict, Counter
from pathlib import Path

import hydra.utils
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig
from lexsubgen.metrics import wsi_metrics
from sklearn.metrics import pairwise_distances

from multilang_wsi_evaluation import utils
from multilang_wsi_evaluation.clusterers_models import BaseClusterer, Clusterer
from multilang_wsi_evaluation.interfaces import IWSIVectorizer
from multilang_wsi_evaluation.subsets_strategy import SubsetsStrategy
from multilang_wsi_evaluation.wsi_metrics import compute_unsup_metrics, compute_wsi_metrics_for_words
from multilang_wsi_evaluation.wsi_metrics import aggregate_metrics, add_unsup_metrics
from multilang_wsi_evaluation.wsi_metrics import SemevalMetricsWrapper
from multilang_wsi_evaluation.semeval_metrics import compute_semeval_2010_metrics as fast_semeval_2010_metrics


def generate_clusterers(base_clusterers: tp.Iterable[BaseClusterer], dist: tp.Optional[tp.Union[str, tp.Callable]]) \
        -> tp.Generator[tp.Tuple[Clusterer, tp.Dict[str, tp.Any]], None, None]:
    for base_clusterer in base_clusterers:
        if dist in base_clusterer.dists:
            for param_values in itertools.product(*base_clusterer.param_to_values.values()):
                param_to_value = dict(zip(base_clusterer.param_to_values.keys(), param_values))
                clusterer = base_clusterer.clusterer_type(**param_to_value)
                param_to_value['@clusterer'] = utils.type_full_name(base_clusterer.clusterer_type)
                param_to_value['@metric'] = utils.dist_full_name(dist)
                yield clusterer, param_to_value


class VectorizerEvaluator:
    def __init__(
            self,
            vectorizer: IWSIVectorizer,
            base_clusterers: tp.List[BaseClusterer],
            clusterers_subsets_strategy: SubsetsStrategy,
            verbose: bool = True,
    ) -> None:
        self.vectorizer = vectorizer
        self.base_clusterers = base_clusterers
        self.clusterers_subsets_strategy = clusterers_subsets_strategy
        self.verbose = verbose

        self.dists = set()
        for base_clusterer in self.base_clusterers:
            self.dists |= set(base_clusterer.dists)

        self._check_clusterers()

    def _check_clusterers(self) -> None:
        if not self.base_clusterers:
            return
        types_counts = Counter([utils.type_full_name(base_clusterer.clusterer_type)
                                for base_clusterer in self.base_clusterers])
        clusterer_type, type_count = types_counts.most_common(1)[0]
        if type_count > 1:
            raise ValueError("Framework does not support duplicates in clusters types: "
                             f"'{clusterer_type}' ({type_count})")

    def evaluate(self, part: utils.WSIDatasetPart):
        group_ids, groups_samples_ids, groups_samples = part.groups_samples()
        self.vectorizer.fit(
            [sample for group_samples in groups_samples for sample in group_samples]
        )
        preds = defaultdict(dict)
        word_metrics = defaultdict(dict)
        word_dists = defaultdict(dict)
        for group_id, group_samples_ids, group_samples in tqdm(zip(group_ids, groups_samples_ids, groups_samples),
                                                               disable=not self.verbose, total=len(group_ids),
                                                               desc=f"Evaluating {part.dataset_name}"):
            matrix = self.vectorizer.predict(group_samples)
            word_dists[group_id]['vectors'] = (group_samples_ids, matrix)
            for dist in self.dists:
                word_dists_matrix = pairwise_distances(matrix, metric=dist) if dist is not None else matrix
                word_dists[group_id][dist] = (group_samples_ids, word_dists_matrix)
                for clusterer, clusterer_params in generate_clusterers(self.base_clusterers, dist):
                    clusters = clusterer.fit_predict(word_dists_matrix)
                    clusterer_params = utils.HashableDict.from_dict(clusterer_params)
                    word_metrics[clusterer_params][group_id] = \
                        compute_unsup_metrics(word_dists_matrix, matrix, clusters)
                    preds[clusterer_params][group_id] = dict(zip(group_samples_ids, clusters))
        mean_metrics, batch_metrics = {}, {}
        for hypers, hypers_preds in tqdm(preds.items(), disable=not self.verbose):
            mean_metrics[hypers], batch_metrics[hypers] = compute_wsi_metrics_for_words(part, hypers_preds)
        agg_metrics, slice_metrics, slice_unsup_metrics = aggregate_metrics(
            part, batch_metrics, mean_metrics, word_metrics, preds,
            self.base_clusterers, self.clusterers_subsets_strategy, self.verbose
        )
        batch_metrics = add_unsup_metrics(batch_metrics, word_metrics)
        return agg_metrics, batch_metrics, slice_metrics, slice_unsup_metrics, preds, word_dists


def save_metrics(word_metrics, aggregated_metrics, name: str) -> None:
    df_save = pd.DataFrame()
    for part_id, metrics in word_metrics.items():
        for hypers, word_searches in metrics.items():
            word_searches['hypers'] = [hypers] * len(word_searches)
            word_searches['part'] = [part_id] * len(word_searches)
            df_save = pd.concat([df_save, word_searches], ignore_index=True)
    list_words = set(df_save.word)
    for temp_word in list_words:
        part_path = os.path.join(name, 'word_metrics', f'{temp_word}.tsv')
        Path(part_path).parent.mkdir(parents=True, exist_ok=True)
        df_save[df_save.word == temp_word].drop(columns=['word']).to_csv(part_path, index=False, sep='\t')
    metrics = list(df_save.columns)[1:-2]
    for metric in metrics:
        metric_df = df_save[['hypers', 'part', 'word', metric]]
        part_path = os.path.join(name, 'grid_metrics', f'{metric}.tsv')
        Path(part_path).parent.mkdir(parents=True, exist_ok=True)
        metric_df.to_csv(part_path, index=False, sep='\t')
    df_save = pd.DataFrame()
    for part_id, metrics_df in aggregated_metrics.items():
        metrics_df['part'] = [part_id] * len(metrics_df)
        df_save = pd.concat([df_save, metrics_df], ignore_index=True)
    df_save = df_save.rename(columns={'index': 'metric', 0: 'score'})
    for metric in df_save['metric'].unique():
        agg_df = df_save[df_save['metric'] == metric]
        part_path = os.path.join(name, 'agg_metrics', f'{metric}.tsv')
        Path(part_path).parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(part_path, index=False, sep='\t')


def save_part_slice_metrics(part_slice_metrics: pd.DataFrame, name: str, part: str) -> None:
    part_path = Path(name) / 'agg_metrics' / 'slice_metrics' / f'{part}.tsv'
    part_path.parent.mkdir(parents=True, exist_ok=True)
    part_slice_metrics.to_csv(part_path, index=False, sep='\t')


def save_part_slice_unsup_metrics(part_slice_unsup_metrics: pd.DataFrame, vectorizer_name: str, part: str) -> None:
    metrics_path = Path(vectorizer_name) / 'agg_metrics' / 'slice_unsup' / f'{part}.tsv'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    part_slice_unsup_metrics.to_csv(metrics_path, index=False, sep='\t')


def save_part_preds(part_preds: tp.Dict[str, tp.Dict[str, tp.Dict[str, int]]], name: str, part: str) -> None:
    for clusterer_params, words_preds in part_preds.items():
        words_records = []
        for word, word_preds in words_preds.items():
            word_records = [{'word': word, 'context_id': context_id, 'predict_sense_id': context_pred}
                            for context_id, context_pred in word_preds.items()]
            words_records.extend(word_records)
        params_df = pd.DataFrame.from_records(words_records)
        clusterer_name = clusterer_params.pop('@clusterer')
        params_path =(Path(name) / 'wsi_preds' / part / f'{clusterer_name}' /
                      (re.sub(r'[\{\}\'\s:,@]', '', str(clusterer_params), ) + '.tsv'))
        params_path.parent.mkdir(parents=True, exist_ok=True)
        params_df.to_csv(params_path, index=False, sep='\t')


def save_part_dists(part_dists: tp.Dict[str, tp.Dict[tp.Any, tp.Tuple]], name: str, part: str) -> None:
    for word, word_dists in part_dists.items():
        for dist, (samples_ids, word_dists_matrix) in word_dists.items():
            dists_df = pd.DataFrame(dict(zip(samples_ids, word_dists_matrix)))
            dists_path = Path(name) / 'word_dists' / part / word / f'{utils.dist_full_name(dist)}.tsv'
            dists_path.parent.mkdir(parents=True, exist_ok=True)
            dists_df.to_csv(dists_path, index=False, sep='\t')


def load_subsets_strategy(subsets_strategy: tp.Optional[DictConfig],
                          default_subsets_strategy: tp.Optional[DictConfig]) -> SubsetsStrategy:
    if subsets_strategy is not None:
        return hydra.utils.instantiate(subsets_strategy)

    if default_subsets_strategy is not None:
        return hydra.utils.instantiate(default_subsets_strategy)

    raise ValueError("set 'default_subsets_strategy' or specify strategy for all params")


def get_base_clusterers(cfg_clusterers: tp.Optional[DictConfig],
                        default_subsets_strategy: tp.Optional[DictConfig]) -> tp.List[BaseClusterer]:
    if cfg_clusterers is None:
        raise ValueError("set your model_clusterers in config.model_clusterers")
    base_clusterers = []

    for clusterer_cfg in cfg_clusterers.values():
        clusterer_type = utils.load_obj(clusterer_cfg.clusterer['_target_'])
        param_to_values = dict(clusterer_cfg.clusterer)
        param_to_values.pop('_target_')
        dists = get_distances(clusterer_cfg.get('metrics'), clusterer_cfg.get('callable_metrics')) or [None]
        dists_strategy = load_subsets_strategy(clusterer_cfg.get('metrics_subsets_strategy'), default_subsets_strategy)

        param_to_strategy: tp.Dict[str, SubsetsStrategy] = {}
        for param in param_to_values.keys():
            strategy_cfg = clusterer_cfg.get('clusterer_subsets_strategy')
            if strategy_cfg is not None:
                strategy_cfg = strategy_cfg.get(param)
            param_to_strategy[param] = load_subsets_strategy(strategy_cfg, default_subsets_strategy)

        for param, val in param_to_values.items():
            if isinstance(val, DictConfig):
                param_to_values[param] = list(hydra.utils.instantiate(val).values())

        base_clusterer = BaseClusterer(clusterer_type, dists, param_to_values, param_to_strategy, dists_strategy)
        base_clusterers.append(base_clusterer)
    
    return base_clusterers


def get_distances(metrics: tp.Optional[tp.Iterable[str]], callable_metrics: tp.Optional[tp.Iterable[str]]) \
        -> tp.List[tp.Union[str, tp.Callable]]:
    loaded_metrics = [utils.load_obj(metric) for metric in (callable_metrics or [])]
    return list(metrics or []) + loaded_metrics


def prepare_semeval_metrics(use_fast_semeval_metrics: bool = True):
    """
        Replace official semeval metrics scripts with fast versions
    """
    wsi_metrics.compute_semeval_2013_metrics = \
        SemevalMetricsWrapper(wsi_metrics.compute_semeval_2013_metrics).compute_semeval_metrics

    if use_fast_semeval_metrics:
        # TODO: add se13 metrics
        wsi_metrics.compute_semeval_2010_metrics = \
            SemevalMetricsWrapper(fast_semeval_2010_metrics).compute_semeval_metrics
    else:
        wsi_metrics.compute_semeval_2010_metrics = \
            SemevalMetricsWrapper(wsi_metrics.compute_semeval_2010_metrics).compute_semeval_metrics


def evaluate_vectorizer(model_vec: DictConfig, cfg_clusterers: tp.Optional[DictConfig],
                        eval_params: DictConfig, run_settings: DictConfig) -> None:
    utils.set_random_state(run_settings.random_state)
    prepare_semeval_metrics(eval_params.use_fast_semeval_metrics)
    vectorizer = hydra.utils.instantiate(model_vec)
    vectorizer_name = run_settings.get('vectorizer_name', vectorizer.__class__.__name__)

    default_subsets_strategy = eval_params.get('default_subsets_strategy')
    clusterers_subsets_strategy = load_subsets_strategy(eval_params.get('clusterers_subsets_strategy'),
                                                        default_subsets_strategy)
    base_clusterers = get_base_clusterers(cfg_clusterers, default_subsets_strategy)
    evaluator = VectorizerEvaluator(vectorizer, base_clusterers, clusterers_subsets_strategy,
                                    verbose=run_settings.verbose)
    info_file = sys.stdout if run_settings.verbose else open(os.devnull, 'w')
    word_metrics, aggregated_metrics = {}, {}

    for part in utils.load_parts(*eval_params.list_paths_datasets):
        print(f"Evaluating '{part.id}'...", file=info_file)
        aggregated_metrics[part.id], word_metrics[part.id], part_slice_metrics, \
            part_slice_unsup_metrics, part_preds, part_dists = evaluator.evaluate(part)
        save_part_slice_metrics(part_slice_metrics, vectorizer_name, part.id)
        save_part_slice_unsup_metrics(part_slice_unsup_metrics, vectorizer_name, part.id)
        save_part_preds(part_preds, vectorizer_name, part.id)
        save_part_dists(part_dists, vectorizer_name, part.id)

    save_metrics(word_metrics, aggregated_metrics, vectorizer_name)
