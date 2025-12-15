import typing as tp
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering

from multilang_wsi_evaluation.subsets_strategy import SubsetsStrategy


@dataclass(frozen=True)
class BaseClusterer:
    clusterer_type: type
    dists: tp.List[tp.Optional[tp.Union[str, tp.Callable]]]
    param_to_values: tp.Dict[str, tp.List[tp.Any]]

    param_to_strategy: tp.Dict[str, SubsetsStrategy]
    dists_strategy: SubsetsStrategy


class Clusterer:  # Actually should be tp.Protocol (python>=3.8)
    def fit_predict(self, matrix: tp.Any) -> tp.Any:
        pass


class AgglomerativeClusteringWrapper(AgglomerativeClustering):
    def fit_predict(self, X, y=None):
        if X.shape[0] < self.n_clusters:
           self.n_clusters = X.shape[0]
        return super().fit_predict(X, y)