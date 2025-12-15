import os
import shutil
import typing as tp
from collections import defaultdict
from pathlib import Path

import fire
import pandas as pd

from multilang_wsi_evaluation import utils
from multilang_wsi_evaluation import wsi_metrics


def make_clear_dir(dir_path: tp.Union[str, os.PathLike]) -> None:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def compute_mean_metrics(part: utils.WSIDatasetPart, id_to_pred: tp.Dict[tp.Any, tp.Any]) \
        -> tp.Generator[tp.Tuple[str, tp.Dict[str, float]], None, None]:
    try:
        mean_metrics = wsi_metrics.compute_mean_metrics(part, id_to_pred)
        yield part.id, mean_metrics
    except RuntimeError as e:
        print(f'Exception for {part.id}: {e}')


def score_path(method_path: Path, datasets_path: Path) -> tp.Generator[tp.Tuple[str, tp.Dict[str, float]], None, None]:
    if method_path.is_file() and datasets_path.is_file():
        gold_part = utils.WSIDatasetPart.from_file(str(datasets_path))
        predict_part = utils.WSIDatasetPart.from_file(str(method_path))
        id_to_pred = dict(zip(predict_part.dataset_df['context_id'], predict_part.dataset_df['predict_sense_id']))
        yield from compute_mean_metrics(gold_part, id_to_pred)
    elif method_path.is_dir() and datasets_path.is_dir():
        name_to_gold_path = {path.name: path for path in datasets_path.iterdir()}
        for predict_path in method_path.iterdir():
            if predict_path.name in name_to_gold_path:
                gold_path = name_to_gold_path[predict_path.name]
                yield from score_path(predict_path, gold_path)


def score_methods(runs_methods_dir: str, gold_datasets_dir: str = '../datasets/', results_dir: str = '../results/') \
        -> None:
    runs_methods_path, gold_datasets_path = Path(runs_methods_dir), Path(gold_datasets_dir)
    metric_to_all_scores = defaultdict(list)

    for method_dir in runs_methods_path.iterdir():
        metrics_scores = dict(score_path(method_dir, gold_datasets_path))
        metric_to_method_scores = defaultdict(lambda: {'method_name': method_dir.name})

        for dataset_part_id, part_scores in metrics_scores.items():
            for metric_name, score in part_scores.items():
                metric_to_method_scores[metric_name][dataset_part_id] = score

        # TODO: Save separate results for each method (on flag --...)
        for metric_name, metric_scores in metric_to_method_scores.items():
            metric_to_all_scores[metric_name].append(metric_scores)

    make_clear_dir(results_dir)

    for metric_name, all_methods_scores in metric_to_all_scores.items():
        all_methods_scores = pd.DataFrame.from_records(all_methods_scores)
        metric_scores_path = Path(results_dir) / f'{metric_name}.tsv'
        all_methods_scores.to_csv(metric_scores_path, index=False, sep='\t')


if __name__ == '__main__':
    fire.Fire(score_methods)
