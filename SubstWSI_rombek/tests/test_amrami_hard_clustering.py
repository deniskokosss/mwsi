import os
import typing as tp
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

import pytest
from SubstWSI_rombek.src.AmramiClusterer import AmramiClusterer


def test_amrami_removal_small_clusts() -> None:
    clusterer = AmramiClusterer(
        hard_clusterer=AgglomerativeClustering(n_clusters=7, affinity='cosine', linkage='average'),
        n_represents=1,
        use_sampling=False,
        use_cluster_merging=True,
        return_soft_clustering=True,
    )
    matrix = np.arange(18).reshape(6, 3)
    clusters = np.array([1, 1, 2, 2, 1, 3])

    new_clusters = clusterer._merge_small_clusters(
        reps_matrix=matrix,
        reps_hard_clustering=clusters
    )
    old2new = {}
    for old_cl, new_cl in zip(clusters, new_clusters):
        if old_cl not in old2new:
            old2new[old_cl] = new_cl
        else:
            cl = old2new[old_cl]
            assert cl == new_cl


@pytest.mark.parametrize('n_rep,given_clust,exp_clust', [
    (1, np.array([1, 1, 2, 2, 1, 3]), np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])),
    (2, np.array([1, 1, 2, 2, 1, 3]), np.array([[1, 0, 0], [0, 1, 0], [0.5, 0, 0.5]])),

])
def test_amrami_soft_clustering(n_rep, given_clust, exp_clust) -> None:
    clusterer = AmramiClusterer(
        hard_clusterer=AgglomerativeClustering(n_clusters=7, affinity='cosine', linkage='average'),
        n_represents=n_rep,
        use_sampling=False,
        use_cluster_merging=True,
    )
    new_clusters = clusterer._create_soft_clustering(
        represents_clustering=given_clust
    )
    assert np.allclose(new_clusters, exp_clust)
