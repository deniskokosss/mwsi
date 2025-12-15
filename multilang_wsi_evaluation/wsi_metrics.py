import csv
import itertools
import os
import re
import typing as tp
from collections import defaultdict, OrderedDict
from itertools import groupby

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from lexsubgen.utils import wsi
from lexsubgen.metrics import wsi_metrics

from multilang_wsi_evaluation import utils
from multilang_wsi_evaluation.clusterers_models import BaseClusterer
from multilang_wsi_evaluation.subsets_strategy import SubsetsStrategy
from multilang_wsi_evaluation.metrics_aggregation import aggregate_per_word_metrics, aggregate_per_word_metric


def _convert_labels_to_semeval2013_file_format(
    words: tp.List, context_ids: tp.List, labels: tp.List, save_path: os.PathLike
) -> tp.NoReturn:
    """Modified function from lexsubgen.metrics.wsi_metrics that supports soft format labels (list of nd.arrays)"""
    for i in range(len(words)):
        if not context_ids[i].startswith(words[i]):
            context_ids[i] = f"{words[i]}.{context_ids[i]}"

    build_row = lambda w, c, l: (f'{w} {c} {l}',)  # in the hard format case

    if isinstance(labels[0], np.ndarray):  # soft format
        def _build_row(word, context_id, l_weights):
            ls = [(label_id, weight) for label_id, weight in sorted(enumerate(l_weights), key=lambda x: x[1], reverse=True) if weight != 0]
            labels_row = ' '.join(f'{label_id}/{weight}' for label_id, weight in ls)
            return (f'{word} {context_id} {labels_row}',)

        build_row = lambda w, c, l: _build_row(w, c, l)

    with open(save_path, "w") as fd:
        writer = csv.writer(fd)
        writer.writerows(build_row(w, c, l) for w, c, l in zip(words, context_ids, labels))


def compute_scores_per_word(
    y_true: tp.List, y_pred: tp.List, group_by: tp.List[str],
) -> tp.List[tp.List]:
    """Modified function from lexsubgen.metrics.wsi_metrics that supports soft format labels (list of nd.arrays)

    Computes common clustering metrics per each word.
    Also aggregates words results.

    Args:
        y_true: Iterable, ground truth labels.
        y_pred: Iterable, labels from your clusterizer.
        group_by: y_true and y_pred are grouped by these values.
            It is assumed that these values are ambiguous words.

    Returns:
        computed metrics
    """

    def convert_to_hard_format(labels: tp.List):
        if not isinstance(labels[0], np.ndarray):
            return labels
        return [np.argmax(word_labels) for word_labels in labels]

    y_true = convert_to_hard_format(y_true)
    y_pred = convert_to_hard_format(y_pred)

    values_per_word = []
    data = sorted(zip(y_true, y_pred, group_by), key=lambda x: x[2])
    for word, grouped in groupby(data, lambda x: x[2]):
        # unzip data
        local_y_true, local_y_pred, _ = zip(*grouped)
        word_values = wsi_metrics.compute_clustering_metrics(
            y_true=local_y_true, y_pred=local_y_pred
        )
        values_per_word.append([word] + word_values)
    return values_per_word


class SemevalMetricsWrapper:
    def __init__(self, source_semeval_metrics_func):
        self._source_semeval_metrics_func = source_semeval_metrics_func

    def compute_semeval_metrics(
        self,
        gold_labels: tp.List,
        pred_labels: tp.List,
        group_by: tp.List[str],
        context_ids: tp.List[str],
        gold_labels_path: os.PathLike = None,
    ) -> tp.Dict[str, tp.List[float]]:
        try:
            metrics = self._source_semeval_metrics_func(
                gold_labels=gold_labels, pred_labels=pred_labels, group_by=group_by,
                context_ids=context_ids, gold_labels_path=gold_labels_path
            )
        except Exception:
            raise RuntimeError('Invalid dataset or preds file for SemEval evaluation scripts')

        # Since evaluation scripts explicitly add 'all' word with aggregated metrics
        assert 'all' not in set(group_by), "By now evaluation script does not support 'all' as a target word..."
        assert set(metrics.keys()) - {'all'} == set(group_by), \
            f"Not all of the words were processed by the SemEval scripts:" \
            f" {(set(metrics.keys()) - {'all'}) ^ set(group_by)}"

        # For _AVG metrics complex number may occur
        for word, word_metrics in metrics.items():
            metrics[word] = [val if not isinstance(val, complex) else val.real for val in word_metrics]

        return metrics


os.environ['JAVA_TOOL_OPTIONS'] = '-Dfile.encoding=UTF8'  # Set encoding for java eval scripts

# Update invalidated links
wsi.SEMEVAL2013URL = "https://zenodo.org/record/5638384/files/SemEval-2013-Task-13-test-data.zip"
wsi.SEMEVAL2010URL = "https://zenodo.org/record/5638549/files/evaluation.zip"
wsi.SEMEVAL2010TRAINURL = "https://zenodo.org/record/5638549/files/training_data.tar.gz"
wsi.SEMEVAL2010TESTURL = "https://zenodo.org/record/5638549/files/test_data.tar.gz"

# update functions in order to work with soft clustering format
wsi_metrics._convert_labels_to_semeval2013_file_format = _convert_labels_to_semeval2013_file_format
wsi_metrics.compute_scores_per_word = compute_scores_per_word


def compute_unsup_metrics(word_dists_matrix, features_matrix, clusters):
    # convert soft clusters into the hard format
    if isinstance(clusters, np.ndarray) and clusters.ndim == 2:
        clusters = clusters.argmax(axis=-1)

    try:
        sil_score = silhouette_score(word_dists_matrix, clusters, metric='precomputed')
    except Exception as e:
        # print(f'Metrics exception for word {word} in silhouette_score:', e)
        # TODO : check doc or score = -2. Effects
        sil_score = -2
    try:
        cal_har_score = calinski_harabasz_score(features_matrix, clusters)
    except Exception as e:
        # print(f'Metrics exception for word {word} in calinski_harabasz_score:', e)
        # TODO : check doc or score = -2. Effects
        cal_har_score = -2
    return {'silhouette': sil_score, 'calinski_harabasz': cal_har_score}


def compute_wsi_metrics_for_words(part: utils.WSIDatasetPart, word_to_preds: tp.Dict[str, tp.Dict[tp.Any, tp.Any]]) \
        -> tp.Tuple[tp.Dict[str, float], pd.DataFrame]:
    id_to_pred: tp.Dict[tp.Any, tp.Any] = {}
    for word_preds in word_to_preds.values():
        id_to_pred.update(word_preds)
    return compute_wsi_metrics(part, id_to_pred)


def compute_mean_metrics(part: utils.WSIDatasetPart, id_to_pred: tp.Dict[tp.Any, tp.Any]) -> tp.Dict[str, float]:
    mean_metrics, _ = compute_wsi_metrics(part, id_to_pred)
    return mean_metrics


def compute_specific_mean_metric(part: utils.WSIDatasetPart, id_to_pred: tp.Dict[tp.Any, tp.Any], metric: str) \
        -> tp.Dict[str, float]:
    """
    Difference from `compute_mean_metrics`: we aggregate only the specified metric
    Most aggregation scripts give several other metrics along with the specified: we return them too
    """
    eval_data = prepare_wsi_eval_data(part, id_to_pred)
    se10_metrics = [metric for metric in wsi_metrics.SEMEVAL_METRICS if metric.startswith('S10_')]
    se13_metrics = [metric for metric in wsi_metrics.SEMEVAL_METRICS if metric.startswith('S13_')]

    # TODO: Still could be improved: make more fine-grained and distinguish within SemEval metrics (fscore, vmeasure)
    if metric in se10_metrics:
        semeval_2010_values = wsi_metrics.compute_semeval_2010_metrics(
            gold_labels=eval_data['y_true'], pred_labels=eval_data['y_pred'],
            group_by=eval_data['group_by'], context_ids=eval_data['context_ids']
        )
        return dict(zip(se10_metrics, semeval_2010_values['all']))
    elif metric in se13_metrics:
        semeval_2013_values = wsi_metrics.compute_semeval_2013_metrics(
            gold_labels=eval_data['y_true'], pred_labels=eval_data['y_pred'],
            group_by=eval_data['group_by'], context_ids=eval_data['context_ids']
        )
        return dict(zip(se13_metrics, semeval_2013_values['all']))
    elif metric in wsi_metrics.METRICS:
        scores_per_word = wsi_metrics.compute_scores_per_word(
            y_true=eval_data['y_true'], y_pred=eval_data['y_pred'], group_by=eval_data['group_by']
        )
        per_word_df = pd.DataFrame(scores_per_word, columns=['word'] + wsi_metrics.METRICS)
        mean_values = wsi_metrics.compute_weighted_avg(per_word_df, wsi_metrics.METRICS)
        return dict(zip(wsi_metrics.METRICS, mean_values))

    raise ValueError(f"Unsupported metric '{metric}' for aggregation")


def merge_word_metrics(word_metrics: tp.Dict[str, tp.Dict[str, float]], batch_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = list(list(word_metrics.values())[0].keys())
    for metric in metrics:
        metric_dict = {key: value[metric] for key, value in word_metrics.items()}
        batch_metrics[metric] = batch_metrics['word'].map(metric_dict)
    return batch_metrics


def add_unsup_metrics(batch_metrics, word_metrics):
    metrics = list(list(word_metrics.values())[0].values())[0].keys()
    for hyps in batch_metrics.keys():
        for metric in metrics:
            metric_dict = {key: value[metric] for key, value in word_metrics[hyps].items()}
            batch_metrics[hyps][metric] = batch_metrics[hyps]['word'].map(metric_dict)
    return batch_metrics


def compute_part_slice_metrics(all_metrics: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """
    compute path_fh_metrics (all combination of hypers)
    """
    all_metrics = all_metrics.copy()
    metrics_names = set(all_metrics.columns) - {'word', 'hypers'}
    hypers_keys = all_metrics['hypers'].apply(lambda hypers: set(hypers.keys()))
    hypers_keys = list(set(itertools.chain.from_iterable(hypers_keys)))  # keys of params in clusterers

    for key in hypers_keys:
        all_metrics[key] = all_metrics['hypers'].apply(lambda hypers: (key, hypers.get(key)))

    keys_subsets = utils.powerset(hypers_keys)
    df_metrics = []

    for keys_subset in tqdm(keys_subsets, disable=not verbose):
        index_to_records = defaultdict(dict)

        for metric_name in metrics_names:
            max_metric_df = all_metrics.groupby(keys_subset + ["word"]).agg({metric_name: max}).reset_index()
            max_metric_df = pd.merge(max_metric_df, all_metrics, on=keys_subset + ["word", metric_name])
            # Since several hypers can have the same metric value => could be duplicates
            max_metric_df = max_metric_df.groupby(keys_subset + ["word"]).first().reset_index()

            for index_val, words_metrics_df in max_metric_df.groupby(keys_subset):
                agg_metric = aggregate_per_word_metric(words_metrics_df[list(metrics_names | {'word'})], metric_name)
                if len(keys_subset) == 1 and len(index_val) == 2:  # remove squeeze from group_by index
                    index_val = (index_val,)
                index_to_records[index_val][metric_name] = agg_metric

        subset_metrics_df = pd.DataFrame.from_records(list(index_to_records.values()))
        subset_metrics_df['part_fh'] = [dict(params) for params in index_to_records.keys()]
        df_metrics.append(subset_metrics_df)

    slice_metrics_df = pd.concat(df_metrics)
    slice_metrics_df = slice_metrics_df[['part_fh', *sorted(metrics_names)]]

    return slice_metrics_df


def generate_clusterer_subsets(clusterer: BaseClusterer) -> tp.Dict[str, tp.List[tp.List[tp.Any]]]:
    def generate_subsets_for_param(param: str, param_values: tp.List[tp.Any]) -> tp.List[tp.List[tp.Any]]:
        subsets_strategy = clusterer.param_to_strategy[param]
        return subsets_strategy.select_subsets(param_values)

    param_to_subsets = {
        param: generate_subsets_for_param(param, param_values)
        for param, param_values in clusterer.param_to_values.items()
    }
    dists_names = [utils.dist_full_name(dist) for dist in clusterer.dists]
    param_to_subsets['@metric'] = clusterer.dists_strategy.select_subsets(dists_names)
    return param_to_subsets


def get_product_clusterer_subsets(clusterer_subsets: tp.Dict[str, tp.List[tp.List[tp.Any]]]) \
        -> tp.List[tp.Dict[str, tp.List[tp.Any]]]:
    param_to_subset_items = []
    for params_subsets_values in itertools.product(*clusterer_subsets.values()):
        param_to_subset = dict(zip(clusterer_subsets.keys(), params_subsets_values))
        param_to_subset_items.append(param_to_subset)
    return param_to_subset_items


def generate_merged_clusterers_items(base_clusterers: tp.List[BaseClusterer],
                                     clusterers_subsets_strategy: SubsetsStrategy) \
        -> tp.List[tp.Dict[str, tp.List[tp.Any]]]:
    clusterers_params_items = {
        utils.type_full_name(clusterer.clusterer_type):
            get_product_clusterer_subsets(generate_clusterer_subsets(clusterer))
        for clusterer in base_clusterers
    }
    clusterers_subsets = clusterers_subsets_strategy.select_subsets(list(clusterers_params_items.keys()))
    merged_params_items = []

    for clusterers_subset in clusterers_subsets:
        subset_clusterers_params_items = {
            clusterer_name: params_items
            for clusterer_name, params_items in clusterers_params_items.items()
            if clusterer_name in clusterers_subset
        }
        for per_clusterer_subset_values in itertools.product(*subset_clusterers_params_items.values()):
            per_clusterer_subset_items = dict(zip(subset_clusterers_params_items.keys(), per_clusterer_subset_values))
            merged_params_items.append(per_clusterer_subset_items)

    return merged_params_items


def compute_slice_unsup_metrics(all_metrics: pd.DataFrame, base_clusterers: tp.List[BaseClusterer],
                                clusterers_subsets_strategy: SubsetsStrategy,
                                unsup_metrics: tp.Iterable[str], verbose: bool) -> pd.DataFrame:
    merged_params_items = generate_merged_clusterers_items(base_clusterers, clusterers_subsets_strategy)
    metrics_records = []

    for clusterer_to_param_to_subset in tqdm(merged_params_items, disable=not verbose):
        def are_all_params_in_subsets(hypers: tp.Dict[str, tp.Any]) -> bool:
            param_to_subset = clusterer_to_param_to_subset.get(hypers['@clusterer'])
            if param_to_subset is None:
                return False
            return all(value in param_to_subset[param] for param, value in hypers.items() if param != '@clusterer')

        params_metrics = all_metrics[all_metrics['hypers'].apply(are_all_params_in_subsets)]
        for unsup_metric in unsup_metrics:
            max_unsup_df = params_metrics.groupby(by='word').agg({unsup_metric: max}).reset_index()
            max_unsup_df = pd.merge(max_unsup_df, params_metrics, on=('word', unsup_metric))
            # Since several hypers can have the same metric value => could be duplicates
            max_unsup_df = max_unsup_df.groupby('word').first().reset_index()
            metric_to_value = aggregate_per_word_metrics(max_unsup_df)
            unsup_params_hypers = {**{'@unsup_metric': unsup_metric}, **clusterer_to_param_to_subset}
            unsup_metrics_record = {**{'hypers': unsup_params_hypers}, **metric_to_value}
            metrics_records.append(unsup_metrics_record)

    metrics_df = pd.DataFrame.from_records(metrics_records)
    max_metrics_df = metrics_df.max(numeric_only=True).to_frame().transpose()
    metrics_names = set(metrics_df.columns) - {'hypers'}
    best_hypers_records = []

    for metric in metrics_names:
        max_metric_df = pd.merge(max_metrics_df[[metric]], metrics_df, on=metric)
        metric_record = {**{'best_by': metric}, **max_metric_df.iloc[0]}
        best_hypers_records.append(metric_record)

    best_hypers_df = pd.DataFrame.from_records(best_hypers_records)
    best_hypers_df.sort_values(by='best_by', inplace=True)
    best_hypers_df = best_hypers_df[['best_by', 'hypers', *sorted(metrics_names)]]
    return best_hypers_df


def prepare_preds_for_hypers(max_metric_df: pd.DataFrame, preds_df: pd.DataFrame) -> tp.Dict[tp.Any, tp.Any]:
    # Since several hypers can have the same metric value => could be duplicates
    max_metric_df = max_metric_df.groupby('word').first().reset_index()
    max_metric_df = pd.merge(max_metric_df, preds_df, on=('word', 'hypers'))
    id_to_pred = dict(zip(max_metric_df['sample_id'], max_metric_df['pred']))
    return id_to_pred


def compute_mean_max_metrics(all_metrics: pd.DataFrame, preds_df: pd.DataFrame, part: utils.WSIDatasetPart,
                             verbose: bool) -> pd.Series:
    max_metrics = all_metrics.groupby(by='word').max(numeric_only=True).reset_index()
    metrics_names = list(set(max_metrics.columns) - {'word', 'hypers'})
    metrics_vals = []

    for metric in tqdm(metrics_names, disable=not verbose):
        max_metric_df = pd.merge(
            max_metrics[['word', metric]], all_metrics[['word', 'hypers', metric]], on=('word', metric))
        id_to_pred = prepare_preds_for_hypers(max_metric_df, preds_df)
        mean_metrics = compute_specific_mean_metric(part, id_to_pred, metric)
        metrics_vals.append(mean_metrics[metric])

    return pd.Series(data=metrics_vals, index=metrics_names)


def compute_unsup_mean_metrics(all_metrics: pd.DataFrame, unsup_metric: str, preds_df: pd.DataFrame,
                               part: utils.WSIDatasetPart) -> pd.Series:
    max_metric_df = all_metrics.groupby(by='word').agg({unsup_metric: max}).reset_index()
    max_metric_df = pd.merge(max_metric_df, all_metrics, on=('word', unsup_metric))
    id_to_pred = prepare_preds_for_hypers(max_metric_df, preds_df)
    mean_metrics = compute_mean_metrics(part, id_to_pred)
    return pd.Series(data=mean_metrics.values(), index=mean_metrics.keys())


def make_preds_df(preds: tp.Dict[str, tp.Dict[str, tp.Dict[tp.Any, tp.Any]]]) -> pd.DataFrame:
    samples = [
        (word, hypers, sample_id, pred)
        for hypers, hypers_items in preds.items()
        for word, word_items in hypers_items.items()
        for sample_id, pred in word_items.items()
    ]
    return pd.DataFrame(samples, columns=('word', 'hypers', 'sample_id', 'pred'))


def make_mean_metrics_df(mean_metrics: tp.Dict[str, tp.Dict[str, float]]) -> pd.DataFrame:
    records = [
        {**{'hypers': hypers}, **hypers_metrics}
        for hypers, hypers_metrics in mean_metrics.items()
    ]
    return pd.DataFrame.from_records(records)


def aggregate_metrics(part: utils.WSIDatasetPart, word_metrics, mean_metrics, unsup_metrics, preds,
                      base_clusterers: tp.List[BaseClusterer], clusterers_subsets_strategy: SubsetsStrategy,
                      verbose: bool):
    """
    aggregate metrics. Compute max_metrics, fh_max_metrics, part_fh_metrics, unsup_metrics
    """
    all_metrics = pd.DataFrame()
    # concat all metrics and write their hypers
    for hypers, temp_df in word_metrics.items():
        temp_df['hypers'] = [hypers] * len(temp_df)
        all_metrics = pd.concat([all_metrics, temp_df], ignore_index=True)
    # compute max_ metrics (group by word, take max per every hypers and avg by words)
    preds_df = make_preds_df(preds)
    dataset_max_metrics = compute_mean_max_metrics(all_metrics, preds_df, part, verbose)
    dataset_max_metrics.index = ['max_' + i for i in list(dataset_max_metrics.index)]  # rename indexes
    # compute fh_max metrics
    temp = make_mean_metrics_df(mean_metrics).set_index('hypers')
    dataset_fh_max_metrics = temp.max().to_frame()  # take max scores for every metric
    # take hypers for best mean score for all words
    dataset_fh_max_metrics['hypers'] = [temp.idxmax()[metric] for metric in temp.columns]  # compute hypers for fh_max_
    dataset_fh_max_metrics.index = ['fh_max_' + i for i in list(dataset_fh_max_metrics.index)]  # rename indexes
    slice_metrics = compute_part_slice_metrics(all_metrics, verbose)  # compute part slice metrics

    mas_metrics, list_word, list_params = defaultdict(list), [], []
    for key, value in unsup_metrics.items():
        for key1, value1 in value.items():
            list_word.append(key1)
            list_params.append(key)
            for m, score in value1.items():
                mas_metrics[m].append(score)
    unsup_metrics_df = pd.DataFrame({'word': list_word, 'hypers': list_params})
    for m, l_score in mas_metrics.items():
        unsup_metrics_df[m] = l_score
    all_metrics = all_metrics.merge(unsup_metrics_df, how='inner', on=['hypers', 'word'])
    unsup_metrics_names = set(unsup_metrics_df.columns) - {'word', 'hypers'}

    # compute slice unsup metrics
    slice_unsup_metrics = compute_slice_unsup_metrics(
        all_metrics, base_clusterers, clusterers_subsets_strategy, unsup_metrics_names, verbose
    )

    # compute unsup metrics
    dataset_unsup_metrics = {}
    for metric in tqdm(unsup_metrics_names, disable=not verbose):
        mean_metrics = compute_unsup_mean_metrics(all_metrics, metric, preds_df, part)
        mean_metrics.index = [f'{metric}_' + i for i in list(mean_metrics.index)]
        dataset_unsup_metrics[metric] = mean_metrics
    concat_dataset_metrics = [data.reset_index() for key, data in dataset_unsup_metrics.items()]
    concat_dataset_metrics.extend([dataset_max_metrics.reset_index(), dataset_fh_max_metrics.reset_index()])

    return pd.concat(concat_dataset_metrics), slice_metrics, slice_unsup_metrics


def prepare_wsi_eval_data(part: utils.WSIDatasetPart, id_to_pred: tp.Dict[tp.Any, tp.Any]) -> tp.Dict[str, tp.List[tp.Any]]:
    gold_dataset = part.dataset_df
    current_dataset = pd.DataFrame({'context_id': id_to_pred.keys(), 'predict_sense_id': id_to_pred.values()})

    assert len(current_dataset) == len(gold_dataset), 'Lengths of the predictions and golds should be the same'
    assert set(current_dataset['context_id']) == set(gold_dataset['context_id']), \
        f"Invalid context_ids: {set(current_dataset['context_id']) ^ set(gold_dataset['context_id'])}"

    current_dataset = pd.merge(
        left=gold_dataset['context_id'], right=current_dataset, on='context_id', validate='1:1'
    )
    return dict(
        y_true=gold_dataset['gold_sense_id'].tolist(), y_pred=current_dataset['predict_sense_id'].tolist(),
        group_by=gold_dataset['word'].tolist(), context_ids=current_dataset['context_id'].astype(str).tolist()
    )


def compute_wsi_metrics(part: utils.WSIDatasetPart, id_to_pred: tp.Dict[tp.Any, tp.Any]) \
        -> tp.Tuple[tp.Dict[str, float], pd.DataFrame]:
    eval_data = prepare_wsi_eval_data(part, id_to_pred)
    return compute_wsi_metrics_lexsubgen(**eval_data)


# TODO: Do we need these lists? Can we just use wsi.METRICS, ets?
MATCH_SEMEVAL_SCORES_RE = re.compile(r"(\w+|\w+\.\w+)(\t*-?\d+\.?\d*)+")
MATCH_TOTAL_VALUE = re.compile(r"Total (.+):(.+)")
METRICS = [
    "ARI",
    "NMI",
    "goldInstance",
    "sysInstance",
    "goldClusterNum",
    "sysClusterNum",
]
SEMEVAL_METRICS = [
    # "S13_Precision",
    # "S13_Recall",
    # "S13_F1",
    # "S13_FNMI",
    # "S13_AVG",
    "S10_FScore",
    "S10_Precision",
    "S10_Recall",
    "S10_VMeasure",
    "S10_Homogeneity",
    "S10_Completeness",
    "S10_AVG",
]
ALL_METRICS = METRICS + SEMEVAL_METRICS


def compute_wsi_metrics_lexsubgen(
    y_true: tp.List,
    y_pred: tp.List,
    group_by: tp.List[str],
    context_ids: tp.List[str],
    y_true_file: str = None,
    compute_semeval_metrics_f: bool = True
) -> tp.Tuple[tp.Dict[str, float], pd.DataFrame]:
    """
    Computes clustering metrics: @METRICS
    Args:
        y_true: ground truth
        y_pred: predicted labels
        group_by: @y_true and @y_pred must be grouped by @group_by
            and METRICs must be computed for each group
        context_ids: unique indexes of instances
        y_true_file: if not None true labels will be read from @y_true_file file
    """
    if compute_semeval_metrics_f:
        # semeval_2013_values = wsi_metrics.compute_semeval_2013_metrics(
        #     y_true, y_pred, group_by, context_ids, y_true_file
        # )
        semeval_2010_values = wsi_metrics.compute_semeval_2010_metrics(
            y_true, y_pred, group_by, context_ids, y_true_file
        )

    scores_per_word = wsi_metrics.compute_scores_per_word(y_true, y_pred, group_by)
    per_word_df = pd.DataFrame(scores_per_word, columns=['word'] + METRICS)
    if compute_semeval_metrics_f:
        mean_values = (
            wsi_metrics.compute_weighted_avg(per_word_df, METRICS)
            # + semeval_2013_values["all"]
            + semeval_2010_values['all']
        )
    else:
        mean_values = (
            wsi_metrics.compute_weighted_avg(per_word_df, METRICS)
        )
    # TODO: Do we need this "word_weighted_avg" in our list?
    # scores_per_word.append(["word_weighted_avg"] + mean_values)
    if compute_semeval_metrics_f:
        for i, word in enumerate(per_word_df.word):
            scores_per_word[i].extend(
                # semeval_2013_values[word] +
                semeval_2010_values[word]
            )

    all_metrics = pd.DataFrame(scores_per_word, columns=["word"] + ALL_METRICS)
    mean_metrics = OrderedDict(
        (metric, value)
        for metric, value in zip(ALL_METRICS, mean_values)
    )

    return mean_metrics, all_metrics
