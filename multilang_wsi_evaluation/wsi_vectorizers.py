import itertools
import typing as tp

import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

from multilang_wsi_evaluation.interfaces import IWSI, IWSIVectorizer, Sample
from multilang_wsi_evaluation.clusterers_models import Clusterer, BaseClusterer
from multilang_wsi_evaluation.vectorizer_evaluator import generate_clusterers


class VectorizerWSI(IWSI):
    def __init__(self, vectorizer: IWSIVectorizer, clusterer: Clusterer, verbose: bool = True) -> None:
        self.vectorizer = vectorizer
        self.clusterer = clusterer
        self.verbose = verbose

    def predict(self, samples: tp.List[Sample]) -> tp.List[tp.Any]:
        self.vectorizer.fit(samples)
        samples_words = {sample.lemma for sample in samples}
        samples_clusters = []

        for word, word_samples in tqdm(itertools.groupby(samples, key=lambda sample: sample.lemma),
                                       total=len(samples_words), disable=not self.verbose):
            word_vectors = self.vectorizer.predict(word_samples)
            word_clusters = self.clusterer.fit_predict(word_vectors)
            samples_clusters.extend(word_clusters)

        return samples_clusters


class AdaptiveClusterer:
    def __init__(self, base_clusterer: BaseClusterer,
                 unsup_metric: tp.Callable[[tp.Any, tp.Any, tp.Any], float],
                 unsup_metric_name: tp.Optional[str] = None) -> None:
        self.base_clusterer = base_clusterer
        self.unsup_metric = unsup_metric
        self.unsup_metric_name = unsup_metric_name

    def fit_predict(self, matrix: tp.Any) -> tp.Any:
        params_clusters, params_unsup_metric = [], []

        for dist in self.base_clusterer.dists:
            word_dists_matrix = pairwise_distances(matrix, metric=dist) if dist is not None else matrix
            for clusterer, clusterer_params in generate_clusterers([self.base_clusterer], dist):
                clusters = clusterer.fit_predict(word_dists_matrix)
                params_clusters.append(clusters)
                unsup_metric = self.unsup_metric(word_dists_matrix, matrix, clusters)
                params_unsup_metric.append(unsup_metric)

        return params_clusters[np.argmax(params_unsup_metric)]
