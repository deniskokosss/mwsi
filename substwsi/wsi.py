import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.naive_bayes import BernoulliNB
from joblib import Memory
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import  _TREE_BUILDERS
import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cdist
# import seaborn as sns
from sklearn.metrics import silhouette_score

def print_mfs_feats(vectorizer, vecs, senses, topf=25):
    mfs_ids = senses.value_counts().index[:2]
    mfs_mask = senses.isin(mfs_ids)

    bnb = BernoulliNB()
    bnb.fit(vecs.astype(np.bool).astype(np.int)[mfs_mask], senses[mfs_mask])
    feature_probs = np.exp(bnb.feature_log_prob_)
    assert feature_probs.shape[
               0] == 2, f'Naive Bayes has weight matrix with {feature_probs.shape[1]} rows! Only 2 rows are supported. Check if it is 2-class data.'
    fn = np.array(vectorizer.get_feature_names())
    result = []
    for cls in range(len(feature_probs)):
        top_feats = feature_probs[cls].argsort()[::-1][:topf]
        result.append(' '.join((f'{feat} {p1:.2f}/{p2:.2f}' for feat, p1, p2 in
                                zip(fn[top_feats], feature_probs[cls, top_feats], feature_probs[1 - cls, top_feats]))))
    return result

def get_distances_hist(df, fn1, fn2):
    sns.set(font_scale=2)

    g = sns.FacetGrid(df, col_wrap = 4, col="word", hue='same', height=10, aspect=2,
                      sharex=False, sharey=False )
    g = g.map(sns.distplot, "distances", norm_hist=True)
    g.savefig(fn1)

    g2 = sns.FacetGrid(df, col_wrap=4, col="word", height=10, aspect=2,
                      sharex=False, sharey=False)
    g2 = g2.map(sns.distplot, "distances", norm_hist=True)
    g2.savefig(fn2)

def clusterize_search( word, vecs, gold_sense_ids = None ,ncs=list(range(1, 5, 1)) + list(range(5, 12, 2)),
            affinities=('cosine',), linkages=('average',), print_topf=None,
            generate_pictures_df = False,  corpora_ids = None):
    if linkages is None:
        linkages = _TREE_BUILDERS.keys()
    if affinities is None:
        affinities = ('cosine', 'euclidean', 'manhattan')
    sdfs = []
    mem = Memory('maxari_cache', verbose=0)
    # warn_zero_vecs_words = []
    tmp_dfs = []


    zero_vecs = ((vecs ** 2).sum(axis=-1) == 0)
    if zero_vecs.sum() > 0:
        # warn_zero_vecs_words.append(zero_vecs.mean())
        vecs = np.concatenate((vecs, zero_vecs[:, np.newaxis].astype(vecs.dtype)), axis=-1)

    if generate_pictures_df:
        if gold_sense_ids is not None:
            sense_ids = gold_sense_ids.to_numpy()
            bool_mask = sense_ids[:, None] == sense_ids
        else:
            assert corpora_ids is not None, "gold sense ids and corpora ids are both None"
            w_corpora_ids = corpora_ids
            bool_mask = w_corpora_ids[:, None] == w_corpora_ids


    best_clids = None
    best_silhouette = 0
    distances = []

    for affinity in affinities:
        distance_matrix = cdist(vecs, vecs, metric=affinity)
        distances.append(distance_matrix)
        for nc in ncs:
            for linkage in linkages:
                if linkage == 'ward' and affinity != 'euclidean':
                    continue
                clr = AgglomerativeClustering(affinity='precomputed', linkage=linkage, n_clusters=nc)
                clids = clr.fit_predict(distance_matrix) if nc > 1 else np.zeros(len(vecs))

                ari = ARI(gold_sense_ids, clids) if gold_sense_ids is not None else np.nan
                sil_cosine = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids,metric='cosine')
                sil_euclidean = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids, metric='euclidean')
                vc = '' if gold_sense_ids is None else '/'.join(
                                        np.sort(pd.value_counts(gold_sense_ids).values)[::-1].astype(str))
                if sil_cosine > best_silhouette:
                    best_silhouette = sil_cosine
                    best_clids = clids

                sdf = pd.DataFrame({'ari': ari,
                                    'word': word, 'nc': nc,
                                    'sil_cosine': sil_cosine,
                                    'sil_euclidean': sil_euclidean,
                                    'vc': vc,
                                    'affinity': affinity, 'linkage': linkage}, index=[0])

                sdfs.append(sdf)

        if generate_pictures_df:
            tmp_df = pd.DataFrame()
            tmp_df['distances'] = distance_matrix.flatten()
            tmp_df['same'] = bool_mask.flatten()
            tmp_df['word'] = word
            w_max_ari = max([i['ari'][0] for i in sdfs if i['word'][0] == word and i['affinity'][0] == affinity])
            tmp_df['ari'] = w_max_ari
            w_max_sil_cosine = max([i['sil_cosine'][0] for i in sdfs if i['word'][0] == word and i['affinity'][0] == affinity])
            tmp_df['sil_cosine'] = w_max_sil_cosine
            tmp_dfs.append(tmp_df)

    # if pictures_dir is not None:
    #     assert pictures_prefix is not None
    #     path1 = pictures_dir + '/' + pictures_prefix + 'separate.svg'
    #     path2 = pictures_dir + '/' + pictures_prefix + 'all.svg'
    #
    #     get_distances_hist(big_graph_df, path1, path2)

    picture_df = pd.concat(tmp_dfs) if tmp_dfs else None

    # if len(warn_zero_vecs_words) > 0:
    #     print(
    #         f'WARNING: {len(warn_zero_vecs_words)}/{len(vecs.keys())} words had ~{np.mean(warn_zero_vecs_words)} 0 vectors! Converted them to 1-hot.')
    sdf = pd.concat(sdfs, ignore_index=True)
    # groupby is docuented to preserve inside group order
    # res = sdf.sort_values(by='ari').groupby(by='word').last()
    # # maxari for fixed hypers
    # fixed_hypers = sdf.groupby(['affinity', 'linkage', 'nc']).agg({'ari': np.mean}).reset_index()
    # idxmax = fixed_hypers.ari.idxmax()
    # res_df = fixed_hypers.loc[idxmax:idxmax].copy()
    # res_df = res_df.rename(columns=lambda c: 'fh_maxari' if c == 'ari' else 'fh_' + c)
    # res_df['maxari'] = res.ari.mean()
    #
    # for metric in [c for c in sdf.columns if c.startswith('sil')]:
    #     res_df[metric+'_ari'] = sdf.sort_values(by=metric).groupby(by='word').last().ari.mean()
    return best_clids, sdf, picture_df, distances
