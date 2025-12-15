import logging
import typing as tp
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import ClusterMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from hydra.utils import instantiate


#  Code taken from the https://github.com/asafamr/bertwsi/blob/master/wsi/wsi_clustering.py
class AmramiClusterer(ClusterMixin):
    def __init__(
            self,
            hard_clusterer,
            n_represents: tp.Optional[int] = 15,
            n_clusters_override: int = None,
            use_sampling: bool = True,
            use_cluster_merging: bool = True,
            return_soft_clustering: bool = False,
            samples_per_rep: int = 20,
            disable_tfidf: bool = False

    ):
        if n_clusters_override:
            hard_clusterer = hard_clusterer.set_params(n_clusters=n_clusters_override)
        self._hard_clusterer = hard_clusterer
        self._n_represents = n_represents
        self._use_sampling = use_sampling
        self._return_soft = return_soft_clustering
        if self._return_soft:
            logging.warning('Maybe you want to activate sampling with soft clustering')
        self._use_cluster_merging = use_cluster_merging
        self._samples_per_rep = samples_per_rep
        self._disable_tfidf = disable_tfidf

        self._min_sense_instances = 2  # TODO: add to params

    def _apply_hard_clusterer(self, matrix) -> np.ndarray:
        inst_clusters = self._hard_clusterer.fit_predict(matrix)
        return inst_clusters

    def _create_represents(self, matrix: np.ndarray) -> tp.Dict[str, tp.Dict[int, int]]:
        reps = dict()
        for inst_idx in range(matrix.shape[0]):
            new_samples = list(
                np.random.choice(
                    np.arange(matrix.shape[1]),
                    self._n_represents * self._samples_per_rep,
                    p=matrix[inst_idx, :] / matrix[inst_idx, :].sum()
                )
            )
            new_reps = []
            for i in range(self._n_represents):
                new_rep = {}
                for j in range(self._samples_per_rep):
                    new_sample = new_samples.pop()
                    new_rep[new_sample] = 1  # rep.get(new_sample, 0) + 1
                new_reps.append(new_rep)
            reps[f'inst_{inst_idx}'] = new_reps

        return reps

    def _vectorize_represents(
            self,
            represents: tp.Dict[str, tp.List[tp.Dict]]
    ):
        inst_ids_ordered = list(represents.keys())
        representatives = [y for x in inst_ids_ordered for y in represents[x]]

        dict_vectorizer = DictVectorizer(sparse=False)
        rep_mat = dict_vectorizer.fit_transform(representatives)
        if self._disable_tfidf:
            transformed = rep_mat
        else:
            transformed = TfidfTransformer(norm=None).fit_transform(rep_mat).toarray()
        return transformed

    def _create_soft_clustering(self, represents_clustering: np.ndarray) -> np.ndarray:
        _repr_clustering = LabelEncoder().fit_transform(represents_clustering)
        total_insts = _repr_clustering.shape[0] // self._n_represents
        total_clusters = np.unique(_repr_clustering).size

        instance_senses = np.zeros((total_insts, total_clusters))
        for i in range(total_insts):
            inst_id_clusters, cnts = np.unique(
                _repr_clustering[i * self._n_represents: (i + 1) * self._n_represents],
                return_counts=True
            )
            instance_senses[i, inst_id_clusters] = cnts
            instance_senses[i] /= cnts.sum()

        return instance_senses

    def _merge_small_clusters(
            self,
            reps_matrix: np.ndarray,
            reps_hard_clustering: np.ndarray,
    ) -> np.ndarray:
        labels = LabelEncoder().fit_transform(reps_hard_clustering)
        n_senses = np.max(labels) + 1

        senses_n_domminates = Counter()
        instance_senses = {}
        total_insts = reps_matrix.shape[0] // self._n_represents
        for i in range(total_insts):
            inst_id_clusters = Counter(labels[i * self._n_represents:
                                              (i + 1) * self._n_represents])
            instance_senses[f'inst_{i}'] = inst_id_clusters
            senses_n_domminates[inst_id_clusters.most_common()[0][0]] += 1

        big_senses = [x for x in senses_n_domminates if senses_n_domminates[x] >= self._min_sense_instances]

        sense_means = np.zeros((n_senses, reps_matrix.shape[1]))
        for sense_idx in range(n_senses):
            idxs_this_sense = np.where(labels == sense_idx)
            cluster_center = np.mean(np.array(reps_matrix)[idxs_this_sense], 0)
            sense_means[sense_idx] = cluster_center

        sense_remapping = {}
        if self._min_sense_instances > 0:
            dists = cdist(sense_means, sense_means, metric='cosine')
            closest_senses = np.argsort(dists, )[:, ]

            for sense_idx in range(n_senses):
                for closest_sense in closest_senses[sense_idx]:
                    if closest_sense in big_senses:
                        sense_remapping[sense_idx] = closest_sense
                        break
            new_order_of_senses = list(set(sense_remapping.values()))
            sense_remapping = dict((k, new_order_of_senses.index(v)) for k, v in sense_remapping.items())

            labels = np.array([sense_remapping[x] for x in labels])

        return labels

    def _hard_from_soft(self, soft_clustering: np.ndarray) -> np.ndarray:
        return np.argmax(soft_clustering, axis=1)

    def fit_predict(self, matrix, **kwargs) -> np.ndarray:
        _matrix = matrix.copy()
        if self._use_sampling:
            reps = self._create_represents(_matrix)
            _matrix = self._vectorize_represents(reps)
            hard_clustering = self._hard_clusterer.fit_predict(_matrix)
        else:
            hard_clustering = self._hard_clusterer.fit_predict(matrix)

        if self._use_cluster_merging:
            clustering = self._merge_small_clusters(_matrix, hard_clustering)
        else:
            clustering = hard_clustering

        soft_clustering = self._create_soft_clustering(clustering)
        if self._return_soft:
            return soft_clustering
        else:
            return self._hard_from_soft(soft_clustering)
