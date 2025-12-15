import typing as tp
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import pandas as pd


class MetricAggregator(ABC):
    @abstractmethod
    def aggregate(self, metric_name: str, words_metrics: pd.DataFrame, computed_aggs: tp.Dict[str, float]) -> float:
        pass

    def required_metrics(self) -> tp.Iterable[str]:
        return []


class WeighedAverageAggregator(MetricAggregator):
    def __init__(self, weights_column: str = 'sysInstance') -> None:
        self.weights_column = weights_column

    def aggregate(self, metric_name: str, words_metrics: pd.DataFrame, computed_aggs: tp.Dict[str, float]) -> float:
        weighed_sum = (words_metrics[metric_name] * words_metrics[self.weights_column]).sum()
        weights_sum = words_metrics[self.weights_column].sum()
        return weighed_sum / weights_sum


class AverageAggregator(MetricAggregator):
    def aggregate(self, metric_name: str, words_metrics: pd.DataFrame, computed_aggs: tp.Dict[str, float]) -> float:
        return words_metrics[metric_name].mean()


class SumAggregator(MetricAggregator):
    def aggregate(self, metric_name: str, words_metrics: pd.DataFrame, computed_aggs: tp.Dict[str, float]) -> float:
        return words_metrics[metric_name].sum()


class MetricsGeometricMeanAggregator(MetricAggregator):
    def __init__(self, *metrics_names: str) -> None:
        self.metrics_names = metrics_names

    def aggregate(self, metric_name: str, words_metrics: pd.DataFrame, computed_aggs: tp.Dict[str, float]) -> float:
        prod_val = np.prod([computed_aggs[metric] for metric in self.metrics_names])
        geom_mean = prod_val ** (1 / len(self.metrics_names))
        return geom_mean

    def required_metrics(self) -> tp.Iterable[str]:
        return self.metrics_names


class MetricsHarmonicMeanAggregator(MetricAggregator):
    def __init__(self, *metrics_names: str) -> None:
        self.metrics_names = metrics_names

    def aggregate(self, metric_name: str, words_metrics: pd.DataFrame, computed_aggs: tp.Dict[str, float]) -> float:
        sum_inverse = np.sum([1 / computed_aggs[metric] for metric in self.metrics_names])
        harmonic_mean = len(self.metrics_names) / sum_inverse
        return harmonic_mean

    def required_metrics(self) -> tp.Iterable[str]:
        return self.metrics_names


@lru_cache(maxsize=None)
def build_aggregations() -> tp.Dict[str, MetricAggregator]:
    metric_to_agg = {
        "ARI": WeighedAverageAggregator(),
        "NMI": WeighedAverageAggregator(),
        "goldInstance": SumAggregator(),
        "sysInstance": SumAggregator(),
        "goldClusterNum": AverageAggregator(),
        "sysClusterNum": AverageAggregator(),

        "S13_Precision": AverageAggregator(),
        "S13_Recall": AverageAggregator(),
        "S13_F1": MetricsHarmonicMeanAggregator("S13_Precision", "S13_Recall"),
        "S13_FNMI": AverageAggregator(),
        "S13_AVG": MetricsGeometricMeanAggregator("S13_F1", "S13_FNMI"),
        "S10_FScore": WeighedAverageAggregator(),
        "S10_Precision": WeighedAverageAggregator(),
        "S10_Recall": WeighedAverageAggregator(),
        "S10_VMeasure": WeighedAverageAggregator(),
        "S10_Homogeneity": WeighedAverageAggregator(),
        "S10_Completeness": WeighedAverageAggregator(),
        "S10_AVG": MetricsGeometricMeanAggregator("S10_FScore", "S10_VMeasure"),

        "calinski_harabasz": WeighedAverageAggregator(),
        "silhouette": WeighedAverageAggregator(),
    }
    return metric_to_agg


def aggregate_per_word_metrics(words_metrics: pd.DataFrame) -> tp.Dict[str, float]:
    metrics_names = set(words_metrics.columns) - {'word', 'hypers'}
    metric_to_agg = build_aggregations()

    unsupported_metrics = metrics_names - set(metric_to_agg.keys())
    assert len(unsupported_metrics) == 0, f"unsupported metrics for custom aggregation: {unsupported_metrics}"

    metric_to_value = {}
    for metric, aggregator in metric_to_agg.items():
        if metric in metrics_names:
            metric_to_value[metric] = aggregator.aggregate(metric, words_metrics, metric_to_value)

    return metric_to_value


def aggregate_per_word_metric(words_metrics: pd.DataFrame, metric_name: str) -> float:
    return _aggregate_per_word_metric(words_metrics, metric_name, {})


def _aggregate_per_word_metric(words_metrics: pd.DataFrame, metric_name: str,
                               computed_aggs: tp.Dict[str, float]) -> float:
    metric_agg = build_aggregations()[metric_name]

    for required_metric in metric_agg.required_metrics():
        if required_metric not in computed_aggs:
            computed_aggs[required_metric] = _aggregate_per_word_metric(words_metrics, required_metric, computed_aggs)

    return metric_agg.aggregate(metric_name, words_metrics, computed_aggs)
