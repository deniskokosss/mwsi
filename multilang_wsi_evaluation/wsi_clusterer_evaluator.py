import csv
import typing as tp
from pathlib import Path
from collections import defaultdict

import pandas as pd

from multilang_wsi_evaluation.utils import load_parts
from multilang_wsi_evaluation.utils import WSIDatasetPart
from multilang_wsi_evaluation.interfaces import IWSI
from multilang_wsi_evaluation.files_evaluator import compute_mean_metrics, make_clear_dir


def save_predictions(preds_dir: Path, part: WSIDatasetPart, id_to_pred: tp.Dict[tp.Any, tp.Any]) -> None:
    preds_df = pd.DataFrame({'context_id': id_to_pred.keys(), 'predict_sense_id': id_to_pred.values()})
    preds_path = preds_dir.resolve() / part.dataset_name / part.lang / f'{part.part}.tsv'
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(
        preds_path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL,
        quotechar='"', doublequote=True
    )


def save_parts_metric(part_to_score: tp.Dict[str, float], metric_filepath: Path) -> None:
    parts_df = pd.DataFrame({'dataset': part_to_score.keys(), 'value': part_to_score.values()})
    parts_df.to_csv(metric_filepath, index=False, sep='\t')


def save_metrics(metrics_dir: str, part_to_scores: tp.Dict[str, tp.Dict[str, float]]) -> None:
    metric_to_parts_scores = defaultdict(dict)
    for part_id, metric_to_score in part_to_scores.items():
        for metric, score in metric_to_score.items():
            metric_to_parts_scores[metric][part_id] = score

    make_clear_dir(metrics_dir)
    for metric, part_to_score in metric_to_parts_scores.items():
        metric_filepath = Path(metrics_dir) / f'{metric}.tsv'
        save_parts_metric(part_to_score, metric_filepath)


def score_wsi_clusterer(wsi_clusterer: IWSI, runs_methods_dir: str, *datasets_paths: str,
                        method_name: tp.Optional[str] = None, metrics_dir: tp.Optional[str] = None) -> None:
    if method_name is None:
        method_name = wsi_clusterer.__class__.__name__
    runs_method_dir = Path(runs_methods_dir) / method_name
    make_clear_dir(runs_method_dir)
    parts_scores: tp.List[tp.Tuple[str, tp.Dict[str, float]]] = []

    for part in load_parts(*datasets_paths):
        samples_ids, samples = part.samples()
        clusters = wsi_clusterer.predict(samples)
        id_to_pred = dict(zip(samples_ids, clusters))
        save_predictions(runs_method_dir, part, id_to_pred)
        if metrics_dir is not None:
            parts_scores.extend(compute_mean_metrics(part, id_to_pred))

    if parts_scores:
        part_to_scores = dict(parts_scores)
        save_metrics(metrics_dir, part_to_scores)


class WSIClustererCLIWrapper:
    def __init__(self, wsi_clusterer: IWSI):
        self.wsi_clusterer = wsi_clusterer

    def score(self, runs_methods_dir: str, datasets_path: str, method_name: tp.Optional[str] = None,
              metrics_dir: tp.Optional[str] = None) -> None:
        score_wsi_clusterer(
            self.wsi_clusterer, runs_methods_dir, datasets_path, method_name=method_name, metrics_dir=metrics_dir
        )
