import os
import typing as tp
import numpy as np
from pathlib import Path
import pandas as pd

import pytest

from multilang_wsi_evaluation.semeval_metrics import compute_semeval_2010_metrics
from lexsubgen.metrics import wsi_metrics

from multilang_wsi_evaluation.wsi_metrics import _convert_labels_to_semeval2013_file_format
wsi_metrics._convert_labels_to_semeval2013_file_format = _convert_labels_to_semeval2013_file_format

DATA_PATH = Path('data').resolve()
WSI_LABELS_PATH = DATA_PATH / 'wsi-metrics-data-examples'

def round_custom_metrics(word: str, custom_metrics: tp.List):
    rounded_metrics = custom_metrics.copy()
    for ind in range(len(rounded_metrics) - 1):
        if word == 'all' and ind == 0:
            continue
        rounded_metrics[ind] = round(rounded_metrics[ind], 1)
    rounded_metrics[-1] = (rounded_metrics[0] * rounded_metrics[3]) ** 0.5
    return rounded_metrics

@pytest.mark.parametrize('dataset_labels', [
    WSI_LABELS_PATH / 'labels_perfect_homogenity.tsv',
    WSI_LABELS_PATH / 'labels_article.tsv',
    WSI_LABELS_PATH / 'labels_random.tsv',
])
def test_python_wsi_metrics_se10(dataset_labels: tp.Union[str, os.PathLike]) -> None:
    labeled_df = pd.read_csv(dataset_labels, sep='\t')

    metrics_lexsubgen = wsi_metrics.compute_semeval_2010_metrics(
        gold_labels=labeled_df['gold_sense_id'],
        pred_labels=labeled_df['predict_sense_id'],
        group_by=labeled_df['word'],
        context_ids=labeled_df['context_id']
    )

    metrics_custom = compute_semeval_2010_metrics(
        gold_labels=labeled_df['gold_sense_id'],
        pred_labels=labeled_df['predict_sense_id'],
        group_by=labeled_df['word'],
        context_ids=labeled_df['context_id']
    )

    for word in metrics_lexsubgen:
        lexsubgen_word_metrics = metrics_lexsubgen[word]
        custom_word_metrics = metrics_custom[word]
        custom_word_metrics = round_custom_metrics(word, custom_word_metrics)
        assert lexsubgen_word_metrics == pytest.approx(custom_word_metrics, rel=1e-7)


@pytest.mark.parametrize('dataset_labels', [
    WSI_LABELS_PATH / 'labels_soft_clustering.tsv'
])
def test_python_wsi_metrics_se10_soft_clustering(dataset_labels: tp.Union[str, os.PathLike]) -> None:
    labeled_df = pd.read_csv(dataset_labels, sep='\t')
    labeled_df['gold_sense_id'] = labeled_df['gold_sense_id'].apply(
        lambda x: np.array(list(map(float, x.split())))
    )
    metrics_lexsubgen = wsi_metrics.compute_semeval_2010_metrics(
        gold_labels=labeled_df['gold_sense_id'],
        pred_labels=labeled_df['predict_sense_id'],
        group_by=labeled_df['word'],
        context_ids=labeled_df['context_id']
    )

    metrics_custom = compute_semeval_2010_metrics(
        gold_labels=labeled_df['gold_sense_id'],
        pred_labels=labeled_df['predict_sense_id'],
        group_by=labeled_df['word'],
        context_ids=labeled_df['context_id']
    )

    for word in metrics_lexsubgen:
        lexsubgen_word_metrics = metrics_lexsubgen[word]
        custom_word_metrics = metrics_custom[word]
        custom_word_metrics = round_custom_metrics(word, custom_word_metrics)
        assert lexsubgen_word_metrics == pytest.approx(custom_word_metrics, rel=1e-7)
