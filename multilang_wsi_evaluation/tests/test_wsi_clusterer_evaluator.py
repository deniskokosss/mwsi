import os
import typing as tp
import subprocess
from pathlib import Path

import pytest

from multilang_wsi_evaluation.files_evaluator import make_clear_dir

DATA_PATH = Path('data').resolve()
BASELINES_PATH = Path('baselines').resolve()
GOLD_DATA_PATH = DATA_PATH / 'gold-data-example'
RUNS_PATH = DATA_PATH / 'runs'
RESULTS_PATH = DATA_PATH / 'results'


@pytest.mark.parametrize('clusterer_py', [
    BASELINES_PATH / 'all_to_one.py', BASELINES_PATH / 'all_to_sep.py'
])
def test_wsi_clusterer(clusterer_py: tp.Union[str, os.PathLike]) -> None:
    make_clear_dir(RUNS_PATH)
    make_clear_dir(RESULTS_PATH)

    assert len(list(RUNS_PATH.iterdir())) == 0
    assert len(list(RESULTS_PATH.iterdir())) == 0

    return_code = subprocess.call([
        'python', clusterer_py, '--datasets-path', GOLD_DATA_PATH, '--runs_methods_dir', RUNS_PATH,
        '--metrics-dir', RESULTS_PATH
    ])
    assert return_code == 0

    assert len(list(RUNS_PATH.iterdir())) > 0
    assert len(list(RESULTS_PATH.iterdir())) > 0
