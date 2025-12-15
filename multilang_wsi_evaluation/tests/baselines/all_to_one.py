import typing as tp

import fire

from multilang_wsi_evaluation.interfaces import Sample, IWSI
from multilang_wsi_evaluation.wsi_clusterer_evaluator import WSIClustererCLIWrapper


class AllToOne(IWSI):
    def predict(self, samples: tp.List[Sample]) -> tp.List[tp.Any]:
        return [0] * len(samples)


if __name__ == '__main__':
    fire.Fire(WSIClustererCLIWrapper(AllToOne()).score)
