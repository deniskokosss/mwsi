from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class SingleClusterer(BaseEstimator, ClusterMixin):
    def __init__(self, stub_param: Any = None):
        """Dummy hard clusterer that combines all instances into the one cluster

        @param stub_param: not used
        """

        self._stub_param = stub_param

    @property
    def stub_param(self):
        return self._stub_param

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        @param X: ndarray of shape (n_samples, n_features) or (n_samples, n_samples) - training instances or distances between them
        @param y: Ignored (Not used, present here for API consistency by convention (from scikit-learn))
        @return: ndarray of shape (n_samples,) - cluster labels
        """

        return np.arange(X.shape[0])


class IndividualClusterer(BaseEstimator, ClusterMixin):
    def __init__(self, eps: float = 0):
        """Dummy hard/soft clusterer that puts all instances into the different clusters

        @param eps: used for label smoothing - we extract 'eps' from the weight of the main cluster and distribute it among
        the other clusters"""

        self._eps = eps

    @property
    def eps(self):
        return self._eps

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        @param X: ndarray of shape (n_samples, n_features) or (n_samples, n_samples) - training instances or distances between them
        @param y: Ignored (Not used, present here for API consistency by convention (from scikit-learn))
        @return: ndarray of shape (n_samples,n_samples) - cluster labels
        """

        n_samples = X.shape[0]
        main_weight = 1 - self.eps
        label_smoothing_weight = self.eps / (n_samples - 1)

        labels1 = np.ones((n_samples, n_samples), dtype=float) * label_smoothing_weight
        labels2 = np.diag(np.ones(n_samples) * (main_weight - label_smoothing_weight))

        return labels1 + labels2
