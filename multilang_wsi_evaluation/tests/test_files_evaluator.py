import os
import pandas as pd
import subprocess

from pathlib import Path
import pytest

DATA_PATH = Path('data').resolve()
GOLD_DATA_PATH = DATA_PATH / 'gold-data-example'
PREDS_METHODS_PATH = DATA_PATH / 'preds-examples'
GOLD_EN_DATA_PATH = DATA_PATH / 'gold-en-data-example'
PREDS_EN_METHODS_PATH = DATA_PATH / 'en-preds-examples'
INVALID_METHODS_PATH = DATA_PATH / 'invalid-preds-examples'
RESULTS_PATH = DATA_PATH / 'results'
SCORE_METHODS_PY = Path(__file__).resolve().parent / '../files_evaluator.py'


@pytest.mark.parametrize('gold_datasets_dir,runs_methods_dir', [
    (GOLD_DATA_PATH, PREDS_METHODS_PATH),
    (GOLD_EN_DATA_PATH, PREDS_EN_METHODS_PATH),
])
def test_same_scores(gold_datasets_dir: str, runs_methods_dir: str) -> None:
    return_code = subprocess.call([
        'python', SCORE_METHODS_PY, '--gold-datasets-dir', gold_datasets_dir, '--runs-methods-dir', runs_methods_dir,
        '--results-dir', RESULTS_PATH
    ])
    assert return_code == 0

    scores_df = pd.read_csv(RESULTS_PATH / 'ARI.tsv', sep='\t')
    scores = scores_df[scores_df['method_name'] == 'same-scores-test'].iloc[0]
    print(scores)

    for split_name, score in scores.items():
        if split_name != 'method_name':
            assert score is None or score == 100


def test_invalid_dataset() -> None:
    return_code = subprocess.call([
        'python', SCORE_METHODS_PY, '--gold-datasets-dir', GOLD_DATA_PATH, '--runs-methods-dir', INVALID_METHODS_PATH,
        '--results-dir', RESULTS_PATH
    ])
    assert return_code == 0

    _, scores_dirs, _ = next(os.walk(RESULTS_PATH))
    assert len(scores_dirs) == 0
