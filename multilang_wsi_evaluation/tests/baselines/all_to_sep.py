import typing as tp
from collections import defaultdict

import fire

from multilang_wsi_evaluation.interfaces import Sample, IWSI
from multilang_wsi_evaluation.wsi_clusterer_evaluator import WSIClustererCLIWrapper


class AllToSep(IWSI):
    def predict(self, samples: tp.List[Sample]) -> tp.List[tp.Any]:
        lemma_to_cluster = defaultdict(int)
        clusters = []

        for sample in samples:
            clusters.append(lemma_to_cluster[sample.lemma])
            lemma_to_cluster[sample.lemma] += 1

        return clusters


if __name__ == '__main__':
    fire.Fire(WSIClustererCLIWrapper(AllToSep()).score)
