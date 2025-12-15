from substs_loading_old import load_substs
from collections import Counter
import os
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import fire
from wsi import clusterize_search, get_distances_hist
import numpy as np
from pymorphy2 import MorphAnalyzer
from pathlib import Path

print(os.getcwd())
_ma = MorphAnalyzer()
_ma_cache = {}


def ma(s):
    # return [ww(s.strip())]
    s = s.strip()  # get rid of spaces before and after token, pytmorphy2 doesn't work with them correctly
    if s not in _ma_cache:
        _ma_cache[s] = _ma.parse(s)
    return _ma_cache[s]


def get_normal_forms(s, nf_cnt=None):
    hh = ma(s)
    if nf_cnt is not None and len(hh) > 1:  # select most common normal form
        h_weights = [nf_cnt[h.normal_form] for h in hh]
        max_weight = max(h_weights)
        return {h.normal_form for i, h in enumerate(hh) if h_weights[i] == max_weight}
    else:
        return {h.normal_form for h in hh}


def max_ari(df, X, ncs,
            affinities=('cosine',), linkages=('average',), vectorizer=None, print_topf=None,
            pictures_dir = None, pictures_prefix = None, preds_path=None):

    # vectors = {}
    # gold_sense_ids = {}

    if pictures_dir is not None:
        os.makedirs(pictures_dir, exist_ok=True)
    sdfs = []
    picture_dfs = []
    for word in df.word.unique():
        mask = (df.word == word)
        vectors = X[mask] if vectorizer is None else vectorizer.fit_transform(X[mask]).toarray()
        gold_sense_ids = df.gold_sense_id[mask]
        gold_sense_ids = None if gold_sense_ids.isnull().any() else gold_sense_ids

        best_clids, sdf, picture_df, _ = clusterize_search(word, vectors, gold_sense_ids, 
                ncs=ncs,
                affinities=affinities, linkages=linkages, print_topf=print_topf,
                generate_pictures_df = pictures_dir is not None)
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


def get_nf_cnt(substs_probs):
    nf_cnt = Counter(nf for l in substs_probs for p, s in l for nf in {h.normal_form for h in ma(s)})
    print(' '.join('%s:%d' % p for p in nf_cnt.most_common(25)))
    return nf_cnt


def preprocess_substs(r, lemmatize=True, nf_cnt=None, exclude_lemmas={}, ):
    res = [s.strip() for p, s in r]
    if exclude_lemmas:
        res1 = [s for s in res if not set(get_normal_forms(s)).intersection(exclude_lemmas)]
        # print(f'{len(r)}->{len(res)}', set(res).difference(res1))
        res = res1
    if lemmatize:
        res = [nf for s in res for nf in get_normal_forms(s, nf_cnt)]
    return res


def get_nf_cnt(substs_probs):
    nf_cnt = Counter(nf for l in substs_probs for p, s in l for nf in {h.normal_form for h in ma(s)})
    #     print(' '.join('%s:%d' % p for p in nf_cnt.most_common(25)))
    return nf_cnt


def combine(substs_probs1, substs_probs2):
    spdf = substs_probs1.to_frame(name='sp1')
    spdf['sp2'] = substs_probs2
    spdf['s2-dict'] = spdf.sp2.apply(lambda l: {s: p for p, s in l})
    res = spdf.apply(lambda r: sorted([(p * r['s2-dict'][s], s) for p, s in r.sp1 if s in r['s2-dict']], reverse=True,
                                      key=lambda x: x[0]), axis=1)
    return res


def do_run_max_ari(substitutes_dump, data_name, dump_images=False, preds_path=None, 
    vectorizers=['TfidfVectorizer','CountVectorizer'], topks=2**np.arange(8, 3, -1), lemmatize=True, analyzers=['word'], ngram_ranges=[(1,1)], min_dfs=(0.1,0.05, 0.03, 0.02, 0.01, 0.0), max_dfs=(0.98, 0.95, 0.9, 0.8), ncs=(2,10)):

    df = load_substs(substitutes_dump, data_name=data_name)

    dump_directory = substitutes_dump + '_dump' + datetime.now().isoformat().replace(':','-')
    nf_cnt = get_nf_cnt(df['substs_probs'])
    sdfs = []
    exclude = []

    for topk in topks:
        print(topk)
        # substs_texts = substs_probs.str[:topk].apply(preprocess_substs, nf_cnt=nf_cnt, ).str.join(' ')
        substs_texts = df.apply(lambda r: preprocess_substs(r.substs_probs[:topk], nf_cnt=nf_cnt, lemmatize=lemmatize,
                                                            exclude_lemmas=exclude + [r.word]), axis=1).str.join(' ')
        for vectorizer,analyzer,ngram_range in ((a,b,c) for a in vectorizers for b in analyzers for c in ngram_ranges):
            
            local_dump_dir = '/'.join([dump_directory, str(topk), vectorizer])
            os.makedirs(local_dump_dir, exist_ok=True)

            for min_df in min_dfs:
                for max_df in max_dfs:
                    dump_filename_prefix = '/%d_%s_%f_%f' % (topk, vectorizer, min_df, max_df)
                    dump_path_prefix =  local_dump_dir + dump_filename_prefix

                    pictures_dir = local_dump_dir if dump_images else None
                    pictures_prefix = dump_filename_prefix if dump_images else None

                    vec = eval(vectorizer)(token_pattern=r"(?u)\b\w+\b", min_df=min_df, max_df=max_df, analyzer=analyzer, ngram_range=ngram_range)
                    try:
                        res_df, res, sdf = max_ari(df, substs_texts, ncs=range(*ncs), affinities=('cosine',), linkages=('average',),
                                                vectorizer=vec, pictures_dir = pictures_dir, pictures_prefix = pictures_prefix, preds_path=preds_path)
                    except ValueError as e:
                        continue
                    res_df = res_df.assign(topk=topk, vec_cls=vectorizer, min_df=min_df, max_df=max_df, analyzer=analyzer, ngram_range=str(ngram_range),lemmatize=lemmatize)
                    sdfs.append(res_df)

                    print(dump_path_prefix, max(res_df['maxari']) if len(res_df['maxari']) else '-')


                    dump_filename = dump_path_prefix + '.res'
                    # Path(dump_filename).mkdir(parents=True, exist_ok=True)
                    res.to_csv(Path(dump_filename))

                    dump_filename = dump_path_prefix + '.res_df'
                    # Path(dump_filename).mkdir(parents=True, exist_ok=True)
                    res_df.to_csv(Path(dump_filename))

                    dump_filename = dump_path_prefix + '.sdf'
                    # Path(dump_filename).mkdir(parents=True, exist_ok=True)
                    sdf.to_csv(Path(dump_filename))

    sdfs_df = pd.concat(sdfs, ignore_index=True)
    dump_filename = dump_directory + '/dump_general.sdfs'
    for metric in ['maxari','fh_maxari', 'sil_cosine_ari', 'sil_euclidean_ari']:
        if metric not in sdfs_df.columns:
            continue
        res_fname = dump_filename+'.'+metric
        sdfs_df.sort_values(by=metric, ascending=False).to_csv(res_fname, sep='\t')
        print('Saved results to:\n', res_fname)
        print(metric, sdfs_df[metric].describe())
    # pd.concat(sdfs, ignore_index=True).sort_values(by='fh_maxari')
    print(pd.concat(sdfs, ignore_index=True).sort_values(by='maxari', ascending=False).head(25))
    # print(pd.concat(sdfs, ignore_index=True).sort_values(by='maxari', ascending=False)['maxari'])
    # pd.concat(sdfs, ignore_index=True).sort_values(by='fh_maxari').tail(25)

def run_full_evalution():

    directory = './xlm/bts-rnc/train-limitNone/'
    russe_train_path = './xlm/russe-wsi-kit/data/main/bts-rnc/_train.tsv'
    substs_dump = {}
    df = pd.read_csv(russe_train_path, sep='\t')

    files = os.listdir(directory)
    # for fp in files:
    #     substs_probs = pd.read_csv(directory+fp, index_col=0)['0'].apply(eval)
    #     print(fp, len(substs_probs), substs_probs.apply(len).mean())
    #     substs_dump[fp] = substs_probs
    # do_run_max_ari()

    # for filename in files:
    #     do_test_russe_dubstitutes(directory + filename)

if __name__ == '__main__':
    fire.Fire(do_run_max_ari)
