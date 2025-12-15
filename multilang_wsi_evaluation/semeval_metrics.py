from collections.abc import Iterable
import os
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_completeness_v_measure


def _prepare_se10_dataframe(
        words: List, context_ids: List, labels: List
) -> pd.DataFrame:
    for ind, word in enumerate(words):
        ctx = context_ids[ind]
        if not ctx.startswith(word):
            context_ids[ind] = f"{word}.{ctx}"
        ctx_labels = labels[ind]
        if isinstance(ctx_labels, (Tuple, List, np.ndarray)):
            labels[ind] = f"{word}.sense.{np.argmax(ctx_labels)}"
    return pd.DataFrame(
        data={
            'word': words,
            'context_id': context_ids,
            'label': labels
        }
    )


def _se10_word_precision_recall_fscore(se10_solution_data: pd.DataFrame):
    same_cl_pairs = se10_solution_data.groupby(['word', 'label_pred']).context_id.count().apply(
        lambda x: x * (x - 1) / 2 if x > 1 else 1).groupby('word').sum()
    same_gs_pairs = se10_solution_data.groupby(['word', 'label_gold']).context_id.count().apply(
        lambda x: x * (x - 1) / 2 if x > 1 else 1).groupby('word').sum()
    same_cl_gs_pairs = se10_solution_data.groupby(['word', 'label_pred', 'label_gold']).context_id.count().apply(
        lambda x: x * (x - 1) / 2).groupby('word').sum()

    se10_words_metrics = pd.DataFrame()
    se10_words_metrics['precision'] = same_cl_gs_pairs / same_cl_pairs
    se10_words_metrics['recall'] = same_cl_gs_pairs / same_gs_pairs
    se10_words_metrics['fscore'] = se10_words_metrics[['precision', 'recall']].apply(
        lambda x: 0.0 if np.isclose(x[0], 0) and np.isclose(x[1], 0) else 2 * x[0] * x[1] / (x[0] + x[1]),
        axis=1
    )

    return se10_words_metrics.loc[:, ~se10_words_metrics.columns.str.startswith('same')]


def _se10_word_homogenity_completeness_vmeasure(se10_solution_data: pd.DataFrame):
    words_hcv = se10_solution_data.groupby('word').apply(
        lambda df: homogeneity_completeness_v_measure(df.label_gold, df.label_pred)
    )
    words_hcv = pd.DataFrame(
        words_hcv.tolist(),
        columns=['homogeneity', 'completeness', 'vmeasure'],
        index=words_hcv.index
    )
    return words_hcv


def _compure_all_words_metrics(se10_solution_data: pd.DataFrame):
    """
    DataFrame must contain word, sample, label_gold and label_pred columns
    """
    total_instances = se10_solution_data.shape[0]
    se10_prf = _se10_word_precision_recall_fscore(se10_solution_data)
    se10_hcv = _se10_word_homogenity_completeness_vmeasure(se10_solution_data)
    se10_word_metrics = pd.concat([se10_prf, se10_hcv], axis=1)

    all_fscore = (se10_word_metrics.fscore * se10_solution_data.groupby(
        'word').context_id.count() / total_instances).sum()
    all_vmeasure = (se10_word_metrics.vmeasure * se10_solution_data.groupby(
        'word').context_id.count() / total_instances).sum()

    se10_word_metrics.loc['all'] = [-1, -1, all_fscore, -1, -1, all_vmeasure]
    se10_word_metrics *= 100
    se10_word_metrics['geom_avg'] = (se10_word_metrics.fscore * se10_word_metrics.vmeasure) ** 0.5

    return se10_word_metrics[
        ['fscore', 'precision', 'recall', 'vmeasure', 'homogeneity', 'completeness', 'geom_avg']].T.to_dict('list')


def compute_semeval_2010_metrics(
        gold_labels: List,
        pred_labels: List,
        group_by: List[str],
        context_ids: List[str],
        gold_labels_path: os.PathLike = None,
) -> Dict[str, List[float]]:
    gold_df = _prepare_se10_dataframe(group_by, context_ids, gold_labels)
    pred_df = _prepare_se10_dataframe(group_by, context_ids, pred_labels)

    all_words_df = gold_df.merge(pred_df, on=['word', 'context_id'], suffixes=('_gold', '_pred'))

    return _compure_all_words_metrics(all_words_df)
