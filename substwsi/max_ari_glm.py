import os
import pandas as pd
import json
import fire
from substwsi.wsi import clusterize_search, get_distances_hist
import numpy as np
from pathlib import Path
import sklearn
from tqdm.auto import tqdm


def load_vectors(vectors_path, samples_path):
    with open(vectors_path) as f:
        vectors_preds = json.load(f)
    with open(samples_path) as f:
        samples = json.load(f)

    id_to_pred = {pred['sample_id']: pred for pred in vectors_preds}
    extended_samples = list(samples)

    for sample in extended_samples:
        current_pred = id_to_pred[sample['id']]
        sample['vector'] = np.array(current_pred['context_output'])

    return pd.DataFrame.from_records(extended_samples)


def max_ari(df, ncs,
            affinities=('cosine',), linkages=('average',), print_topf=None,
            pictures_dir=None, pictures_prefix=None, preds_path=None):
    if pictures_dir is not None:
        os.makedirs(pictures_dir, exist_ok=True)
    sdfs = []
    picture_dfs = []
    for lemma in tqdm(df.lemma.unique()):
        mask = (df.lemma == lemma)
        lemma_vectors = np.stack(df.vector[mask])
        gold_sense_ids = df.gold_sense_id[mask]
        gold_sense_ids = None if gold_sense_ids.isnull().any() else gold_sense_ids

        best_clids, sdf, picture_df, _ = clusterize_search(
            lemma, lemma_vectors, gold_sense_ids,
            ncs=ncs, affinities=affinities, linkages=linkages, print_topf=print_topf,
            generate_pictures_df=(pictures_dir is not None)
        )
        df.loc[mask, 'predict_sense_id'] = best_clids
        sdfs.append(sdf)
        if picture_df is not None:
            picture_dfs.append(picture_df)

    if preds_path is not None:
        Path(preds_path).parent.mkdir(parents=True, exist_ok=True)
        while Path(preds_path).exists():
            preds_path += '_new.tsv'
        df.to_csv(preds_path, sep='\t', index=False)

    picture_df = pd.concat(picture_dfs, ignore_index=True) if picture_dfs else None
    if pictures_dir is not None:
        assert pictures_prefix is not None
        assert not picture_df is not None

        path1 = pictures_dir + '/' + pictures_prefix + 'separate.svg'
        path2 = pictures_dir + '/' + pictures_prefix + 'all.svg'

        get_distances_hist(picture_df, path1, path2)

    sdf = pd.concat(sdfs, ignore_index=True)
    # groupby is docuented to preserve inside group order
    res = sdf.sort_values(by='ari').groupby(by='word').last()
    # maxari for fixed hypers
    fixed_hypers = sdf.groupby(['affinity', 'linkage', 'nc']).agg({'ari': np.mean}).reset_index()
    idxmax = fixed_hypers.ari.idxmax()
    res_df = fixed_hypers.loc[idxmax:idxmax].copy()
    res_df = res_df.rename(columns=lambda c: 'fh_maxari' if c == 'ari' else 'fh_' + c)
    res_df['maxari'] = res.ari.mean()

    for metric in [c for c in sdf.columns if c.startswith('sil')]:
        res_df[metric+'_ari'] = sdf.sort_values(by=metric).groupby(by='word').last().ari.mean()

    return res_df, res, sdf


def norm_dist(vec1, vec2, d):
    vec1 = vec1 / np.linalg.norm(vec1, ord=d)
    vec2 = vec2 / np.linalg.norm(vec2, ord=d)

    return np.linalg.norm(vec1 - vec2, ord=d)


def norm_l1(vec1, vec2):
    return norm_dist(vec1, vec2, d=1)


def norm_l2(vec1, vec2):
    return norm_dist(vec1, vec2, d=2)


def compute_max_ari(vectors_path, samples_path, results_dump_dir=None):
    vectors_df = load_vectors(vectors_path, samples_path)
    linkages = set(sklearn.cluster.hierarchical._TREE_BUILDERS.keys()) - {'ward'}
    affinities = (
        'cosine', 'cityblock', 'euclidean', 'seuclidean', 'correlation', 'hamming', 'jaccard',
        norm_l1
    )
    res_df, res, sdf = max_ari(vectors_df, ncs=range(2, 10), affinities=affinities, linkages=linkages)

    if results_dump_dir is not None:
        res_df.to_csv(os.path.join(results_dump_dir, 'res_df.csv'), index=False)
        res.to_csv(os.path.join(results_dump_dir, 'res.csv'), index=False)
        sdf.to_csv(os.path.join(results_dump_dir, 'sdf.csv'), index=False)


if __name__ == '__main__':
    fire.Fire(compute_max_ari)
