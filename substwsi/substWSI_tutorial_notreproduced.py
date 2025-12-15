import hydra
import mlflow
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import Counter
import os
from datetime import datetime
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from pymorphy2 import MorphAnalyzer
from pymorphy2.analyzer import Parse
from pathlib import Path
from time import time
import regex as re
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB
from joblib import Memory
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import networkx as nx
from transformers import AutoTokenizer, AutoModelForMaskedLM



_ma = MorphAnalyzer()
_ma_cache = {}
def ma(s) -> list[Parse]:
    '''
    Gets a string with one token, deletes spaces before and
    after token and returns grammatical information about it. If it was met
    before, we would get information from the special dictionary _ma_cache;
    if it was not, information would be gotten from pymorphy2.
    '''
    global _ma, _ma_cache
    s = s.strip()  # get rid of spaces before and after token,
                   # pytmorphy2 doesn't work with them correctly
    if s not in _ma_cache:
        _ma_cache[s] = _ma.parse(s)
    return _ma_cache[s]


def load_substs(substs_fname, limit=None, drop_duplicates=False, 
                data_name = None, show=True):
    '''
    If more than one set of substitutes is used, this function combines them.
    '''
    if substs_fname.endswith('&'):
        split = substs_fname.strip('&').split('&')
        # if show:
        #     print(f'Combining:', split)
        dfinps = [load_substs_(p, limit, drop_duplicates, data_name) \
                  for p in split]
        res = dfinps[0]
        nm = len(split[0].split('<mask>'))-1
        for dfinp in dfinps[1:]:
            res = res.merge(dfinp, on=['context','positions'], \
                            how='inner', suffixes=('','_y'))
            res.substs_probs = intersect_sparse(res.substs_probs, \
                                                res.substs_probs_y, \
                                                nmasks=nm, s=0.0)        
            res.drop(columns=[c for c in res.columns if c.endswith('_y')], \
                     inplace=True)
        # if show:
        #     print('\nExamples for one meaning:')
        #     ex = res.loc[(res['word'] == 'балка') & \
        #                   (res['gold_sense_id'] == 1)].head(2)
        #     print('\nExample #1\nword:', tuple(ex.word)[0],
        #       '\ngold_sense_id:', tuple(ex.gold_sense_id)[0],
        #       '\ncontext:', tuple(ex.context)[0], '\nsubstitutes & probabilities:',
        #       tuple(ex.substs_probs)[0])
        #     print('\nExample #2\nword:', tuple(ex.word)[1],
        #       '\ngold_sense_id:', tuple(ex.gold_sense_id)[1],
        #       '\ncontext:', tuple(ex.context)[1], '\nsubstitutes & probabilities:',
        #       tuple(ex.substs_probs)[1])
        #
        #
        #     print('\nExamples for differrent menings:')
        #     ex1 = res.loc[(res['word'] == 'балка') & \
        #                   (res['gold_sense_id'] == 1)].head(1)
        #     ex2 = res.loc[(res['word'] == 'балка') & \
        #                   (res['gold_sense_id'] == 2)].head(1)
        #     print('\nExample #1\nword:', tuple(ex1.word)[0],
        #       '\ngold_sense_id:', tuple(ex1.gold_sense_id)[0],
        #       '\ncontext:', tuple(ex1.context)[0], '\nsubstitutes & probabilities:',
        #       tuple(ex1.substs_probs)[0])
        #     print('\nExample #2\nword:', tuple(ex2.word)[0],
        #       '\ngold_sense_id:', tuple(ex2.gold_sense_id)[0],
        #       '\ncontext:', tuple(ex2.context)[0], '\nsubstitutes & probabilities:',
        #       tuple(ex2.substs_probs)[0])
        return res
    elif substs_fname.endswith('+'): 
        split = substs_fname.strip('+').split('+') # комментарий
        p1 = '+'.join(split[:-1])
        s = float(split[-1]) 
        p2 = re.sub(r'((<mask>)+)(.*?)T',r'T\3\1',p1)
        if p2==p1:
            p2 =  re.sub(r'T(.*?)((<mask>)+)',r'\2\1T',p1)
        print(f'Combining {p1} and {p2}')
        if p1==p2:
            raise Exception('Cannot conver fname to symmetric one:', p1)
        dfinp1, dfinp2 = (load_substs_(p, limit, drop_duplicates, data_name) \
                          for p in (p1,p2))
        dfinp = dfinp1.merge(dfinp2, on=['context','positions'], how='inner', \
                             suffixes=('','_y'))
        dfinp.substs_probs = intersect_sparse(dfinp.substs_probs, \
                                              dfinp.substs_probs_y, \
                                              nmasks=len(substs_fname.split('<mask>'))-1, \
                                              s=s)
        dfinp.drop(columns=[c for c in dfinp.columns if c.endswith('_y')], \
                   inplace=True)
        return dfinp
    else:
        return load_substs_(substs_fname, limit, drop_duplicates, data_name)


def metrics(sdfs):
    # all metrics for each unique word
    sdf = pd.concat(sdfs, ignore_index=True)
    # groupby is docuented to preserve inside group order
    res = sdf.sort_values(by='ari').groupby(by='word').last()
    # maxari for fixed hypers
    fixed_hypers = sdf.groupby(['affinity',
                                'linkage',
                                'nc']).agg({'ari': np.mean}).reset_index()
    idxmax = fixed_hypers.ari.idxmax()
    res_df = fixed_hypers.loc[idxmax:idxmax].copy()
    res_df = res_df.rename(columns=lambda c: 'fh_maxari' if c == 'ari' \
                           else 'fh_' + c)
    res_df['maxari'] = res.ari.mean()

    for metric in [c for c in sdf.columns if c.startswith('fair')]:
        res_df[metric+'_ari'] = sdf.sort_values(by=metric).groupby(by='word').last().ari.mean()

    return res_df, res, sdf


def load_substs_(substs_fname, limit=None, drop_duplicates=True, data_name = None):
    '''
    Collects substitutions, probabilities and examples to one main DataFrame.
    '''
    st = time()
    p = Path(substs_fname).resolve()
    print(p)
    npz_filename_to_save = None
    # print(time()-st, 'Loading substs from ', p)

    # substitutions and probabilities
    if substs_fname.endswith('.npz'): 
        arr_dict = np.load(substs_fname, allow_pickle=True)
        # separate dictionaries for substitutions and probabilities
        ss,pp = arr_dict['substs'], arr_dict['probs'] 
        ss,pp = [list(s) for s in ss], [list(p) for p in pp]
        # creating a DataFrame
        substs_probs = pd.DataFrame({'substs':ss, 'probs':pp}) 
        substs_probs = substs_probs.apply(lambda r: [(p,s) for s,p in zip(r.substs, r.probs)], axis=1) 

    # examword usages and contexts
    p_ex = p.parent / (p.name+'.input')
    if os.path.isfile(p_ex):
        # print(time()-st,'Loading examples from ', p_ex)
        dfinp = pd.read_csv(p_ex, nrows=limit)
        dfinp['positions'] = dfinp['positions'].apply(pd.eval).apply(tuple)
        dfinp['word_at'] = dfinp.apply(lambda r: r.context[slice(*r.positions)], 
                                       axis=1)  # word_at stores wordform as it occured in text

    dfinp.positions = dfinp.positions.apply(tuple)
    # adding substitutions and probabilities to main DF
    dfinp['substs_probs'] = substs_probs 
    if drop_duplicates: # deleting duplicates
        dfinp = dfinp.drop_duplicates('context')
    dfinp.reset_index(inplace = True)
    # display(dfinp)
    dfinp['positions'] = dfinp.positions.apply(tuple)
    return dfinp


# In[5]:


def intersect_sparse(substs_probs, substs_probs_y, nmasks=1, s=0, debug=False):
    '''
    Combines different sets of substitutes (for different tamplates) using
    product of smoothed probability distributions.
    '''
    
    vec = DictVectorizer(sparse=True)
    f1=substs_probs.apply(lambda l: {s:p for p,s in l})
    f2=substs_probs_y.apply(lambda l: {s:p for p,s in l})
    vec.fit(list(f1)+list(f2))
    f1,f2 = (vec.transform(list(f)) for f in (f1,f2)) # sparse matrix

    alpha1, alpha2 = ((1. - f.sum(axis=-1).reshape(-1,1)) / 250000**nmasks \
                      for f in (f1, f2))
    prod = f1.multiply(f2) + f1.multiply(alpha2) + f2.multiply(alpha1)
    # + alpha1*alpha2 is ignored to preserve sparsity 
    # finally, we don't want substs with 0 
    # probs before smoothing in both distribs
    fn = np.array(vec.feature_names_)
    maxlen=(substs_probs_y.apply(len)+substs_probs.apply(len)).max()
    m = prod
    n_texts = m.shape[0]
    
    def reverse_argsort(mdata):
        return np.argsort(mdata)[::-1]
    
    idx = list()
    for text_ix in range(n_texts):
      # sparce matrices are used to preserve high performance
      # refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
      # to learn the sparse matrices indexing (i.e. what is `indptr` and `data`)
      text_sparse_indices = m.indices[m.indptr[text_ix]:m.indptr[text_ix+1]]
      text_sparse_data = m.data[m.indptr[text_ix]:m.indptr[text_ix+1]]
      text_sp_data_revsorted_ixes = reverse_argsort(text_sparse_data)
      smth = text_sparse_indices[text_sp_data_revsorted_ixes]
      idx.append(smth)

    l = list()
    for text_ix, text_sparse_ixes_sorted in enumerate(idx):
      probas = m[text_ix].toarray()[0,text_sparse_ixes_sorted]
      substs = fn[text_sparse_ixes_sorted]
      good_substs = list()
      for proba, subst in zip(probas, substs):
      
        if subst.startswith(' ') and ' ' not in subst.strip():
          good_substs.append((proba, subst))
      l.append(good_substs)
 
    print('Combination: ', l[0][:10])
    return l


def max_ari(df, X, ncs,
            affinity='cosine', linkage='average', vectorizer=None):
    '''
    Gets data and substitutions (and some parameters
    like vectorizer and parameters for clusterization).
    Here for each unique word substitutions
    are vectorized and senses are clusterized.
    It returns metrics of clusterization.
    '''
    sdfs = []
    for word in df.word.unique():
        # collecting examples for the word
        mask = (df.word == word)
        # vectors for substitutions
        vectors = X[mask] if vectorizer is None \
            else vectorizer.fit_transform(X[mask]).toarray()
        # ids of senses of the examples
        gold_sense_ids = df.gold_sense_id[mask]
        gold_sense_ids = None if gold_sense_ids.isnull().any() \
            else gold_sense_ids

        # clusterization (ari is kept in sdf along with other info)
        best_clids, sdf, _ = clusterize_search(word, vectors, gold_sense_ids,
                                               ncs=ncs,
                                               affinity=affinity, linkage=linkage)
        df.loc[mask, 'predict_sense_id'] = best_clids  # result ids of clusters
        sdfs.append(sdf)

    return sdfs




from enum import Enum

class Lemmatize(Enum):
    TOKENIZER = 4
    SUBSTFREQ = 1
    CORPUSFREQ = 2
    GRAPH = 3
    NONE = 5

def get_nf_cnt(substs_probs, mode: Lemmatize):
    '''
    Gets substitutes and returns normal
    forms of substitutes and count of substitutes that coresponds to
    each normal form.
    '''
    if mode == Lemmatize.SUBSTFREQ:
        nf_cnt = Counter(nf
                        for l in substs_probs
                        for p, s in l
                        for nf in {h.normal_form for h in ma(s)})
    elif mode == Lemmatize.CORPUSFREQ:
        nf_cnt = pd.read_csv(r'../notebooks/content/freqrnc2011.csv', sep='\t')
        nf_cnt = nf_cnt.groupby('Lemma')['Freq(ipm)'].max().to_dict()
    elif mode == Lemmatize.GRAPH:
        G = nx.DiGraph()
        for wordform in [t[1] for substs in substs_probs for t in substs]:
            lemmas = [('W_' + t.word, 'L_' + t.normal_form, t.score) for t in ma(wordform) if t.score > 0.45][:1]
            # if not lemmas:
            #     lemmas = [('W_' + t.word, 'L_' + t.normal_form, t.score) for t in ma(wordform)][:1]
            G.add_weighted_edges_from(lemmas)
        nf_cnt = {}
        for i, comp in enumerate(nx.weakly_connected_components(G)):
            cluster_name = [t for t in comp if t.startswith('L_')][0].strip('_L')
            for wordform in [t for t in comp if t.startswith('W_')]:
                nf_cnt[wordform.strip('W_')] = cluster_name
    elif mode == Lemmatize.TOKENIZER:
        nf_cnt = AutoTokenizer.from_pretrained('xlm-roberta-large')
    # print('\n'.join('%s: %d' % p for p in nf_cnt.most_common(10)))
    return nf_cnt

def get_normal_forms(s: str, mode: Lemmatize, nf_cnt):
    '''
    Gets string with one token and returns set of most possible lemmas,
    all lemmas or one possible lemma.
    '''
    if mode == Lemmatize.SUBSTFREQ or mode == Lemmatize.CORPUSFREQ:
        hh = ma(s)
        if nf_cnt is not None:  # select most common normal form
            hhf = [t.normal_form for t in hh if t.normal_form in nf_cnt]
            if hhf:
                idx = np.argmax([nf_cnt[t] for t in hhf])
                return {*hhf}
            else:
                return set()
    elif mode == Lemmatize.GRAPH:
        if s in nf_cnt:
            return {nf_cnt[s]}
        else:
            return {s}
    elif mode == Lemmatize.TOKENIZER:
        return {t for t in nf_cnt.convert_ids_to_tokens(nf_cnt.encode(s))[1:-1] if len(t) >= 3}


def preprocess_substs(r, nf_cnt, mode: Lemmatize, exclude_lemmas={}):
    '''
    For preprocessing of substitutions. It gets Series of substitutions
    and probabilities, exclude lemmas from exclude_lemmas
    if it is not empty and lemmatize them if it is needed.
    '''
    res = [s.strip() for p, s in r]
    if exclude_lemmas:
        res1 = [s for s in res
                if not set(get_normal_forms(s, mode, nf_cnt)).intersection(exclude_lemmas)]
        res = res1
    if mode != Lemmatize.NONE:
        res = [nf for s in res for nf in get_normal_forms(s, mode, nf_cnt)]
    return res


def clusterize_search(word, vecs, gold_sense_ids=None,
                      ncs=list(range(1, 5, 1)) + list(range(5, 12, 2)),
                      affinity='cosine', linkage='average', print_topf=None,
                      generate_pictures_df=False, corpora_ids=None):
    '''
    Gets word, vectors, gold_sense_ids and provides AgglomerativeClustering.
    '''

    sdfs = []
    tmp_dfs = []

    # adding 1 to zero vectors, because there will be problems (all-zero vectorized entries), if they remain zero
    # this introduces a new dimension with possible values
    # 1 -- all the other coords are zeros
    # 0 -- other coords are not all-zeros
    zero_vecs = ((vecs ** 2).sum(axis=-1) == 0)
    if zero_vecs.sum() > 0:
        vecs = np.concatenate((vecs,
                               zero_vecs[:, np.newaxis].astype(vecs.dtype)),
                              axis=-1)

    best_clids = None
    best_silhouette = 0
    distances = []

    # matrix with computed distances between each pair of the two collections of inputs
    distance_matrix = cdist(vecs, vecs, metric=affinity)
    distances.append(distance_matrix)

    for nc in ncs:
        # clusterization
        clr = AgglomerativeClustering(affinity='precomputed',
                                      linkage=linkage,
                                      n_clusters=nc)
        clids = clr.fit_predict(distance_matrix) if nc > 1 else np.zeros(len(vecs))

        # computing metrics
        ari = ARI(gold_sense_ids, clids) if gold_sense_ids is not None else np.nan
        sil_cosine = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids, metric='cosine')
        sil_euclidean = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids, metric='euclidean')

        # vc like 5/4/3 says that
        # there are 5 examples w golden_id=id1;
        # there are 4 examples w golden_id=id2;
        # there are 3 examples w golden_id=id3;
        # e.g. вид 4/3/2 means that there were
        # 4 examples with вид==view; 3 examples .==type; 2 examples .==specie
        vc = '' if gold_sense_ids is None else '/'.join(
            np.sort(pd.value_counts(gold_sense_ids).values)[::-1].astype(str))

        if sil_cosine > best_silhouette:
            best_silhouette = sil_cosine
            best_clids = clids

        # metrics for each word
        sdf = pd.DataFrame({'ari': ari,
                            'word': word, 'nc': nc,
                            'fair': sil_cosine,
                            'vc': vc,
                            'affinity': affinity, 'linkage': linkage},
                           index=[0])

        sdfs.append(sdf)

    sdf = pd.concat(sdfs, ignore_index=True)

    return best_clids, sdf, distances


def clusterize_graph(df, X, min_df=0.0, max_df=1.0):
    sdfs = []
    for word in df.word.unique():
        mask = (df.word == word)
        vecs = X[mask]
        gold_sense_ids = df.gold_sense_id[mask]

        G = nx.Graph()
        for sent in vecs:
            sent = sent.split(' ')
            for i, subst1 in enumerate(sent[:-1]):
                for subst2 in sent[i + 1:]:
                    if not G.has_edge(subst1, subst2):
                        G.add_edge(subst1, subst2, weight=1)
                    else:
                        G[subst1][subst2]['weight'] += 1
        degree_min, degree_max = np.quantile(list(dict(G.degree()).values()), q=[min_df, max_df]) + np.array([-1, 1])
        remove_vertices = [node for node, degree in dict(G.degree()).items() if not(degree_min < degree < degree_max)]
        G.remove_nodes_from(remove_vertices)
        remove_vertices = [node for node, degree in dict(G.degree()).items() if degree == 0]
        G.remove_nodes_from(remove_vertices)

        communities = nx.community.louvain_communities(G)
        subst_clusters = {s: i + 1 for i, comm in enumerate(communities) for s in comm}
        clids = []
        for sent in vecs:
            sent = sent.split(' ')
            cnt = Counter([subst_clusters[subst] for subst in sent if subst in subst_clusters])
            clids.append(cnt.most_common()[0][0])
        ari = ARI(gold_sense_ids, clids) if gold_sense_ids is not None else np.nan
        sdf = pd.DataFrame({
            'affinity': ['co-ocurence'],
            'linkage': ['louvain'],
            'word': [word],
            'nc': [len(communities)],
            'fair': [0],
            'ari': [ari]
        })
        nx.write_gexf(G, f"{word}.gexf")
        sdfs.append(sdf)
    return sdfs


@hydra.main(config_path="config", config_name="config")
def my_app(cfg : DictConfig) -> None:


    mlflow.set_experiment(experiment_name='substwsi2')
    with mlflow.start_run():

        substitutes_dump = '&'.join(cfg.substitutes_dump) + '&'
        lemma_mode = eval(cfg.lemma_mode)
        params = dict(
            clustering=cfg.clustering,
            data=cfg.data_name.split(r'/')[-1],
            substs='&'.join([subst.split('/')[-1].split('-2l')[0] for subst in substitutes_dump.split('&')]).replace('_mask_', 'M'),
            topk=cfg.topk,
            lemma_mode=lemma_mode,
            min_df=cfg.min_df,
            max_df=cfg.max_df,
            ngram_range=cfg.ngram_range,
            ncs=cfg.ncs,
        )
        mlflow.log_params(params)
        df = load_substs(substitutes_dump, data_name=cfg.data_name)

        nf_cnt = get_nf_cnt(df.substs_probs, lemma_mode)
        substs_texts = df.apply(lambda r: preprocess_substs(r.substs_probs[:cfg.topk],
                                                           nf_cnt=nf_cnt,
                                                            mode=lemma_mode),
                                axis=1).str.join(' ')
        vec = instantiate(cfg.vectorizer,
                          token_pattern=r"(?u)\b\w+\b",
                          min_df=cfg.min_df, max_df=cfg.max_df,
                          analyzer=cfg.analyzer, ngram_range=cfg.ngram_range)
        if cfg.clustering == 'agglomerative':
            sdfs = max_ari(df,
                           substs_texts,
                           ncs=range(*cfg.ncs),
                           affinity='cosine',
                           linkage='average',
                           vectorizer=vec)
        elif cfg.clustering == 'louvain':
            sdfs = clusterize_graph(df, substs_texts, cfg.min_df, cfg.max_df)

        res_df, res, sdf = metrics(sdfs)

        metrs = res_df[np.intersect1d(['fh_nc', 'fh_maxari', 'maxari', 'fair_ari'],
                                      res_df.columns)].iloc[0].to_dict()
        mlflow.log_metrics(metrs)
        print(params)
        print(metrs)

if __name__ == "__main__":
    my_app()

#
# clustering=agglomerative
# lemma_mode=Lemmatize.CORPUSFREQ,Lemmatize.SUBSTFREQ,Lemmatize.GRAPH,Lemmatize.TOKENIZER
# topk=20,60,128,256
# min_df=0.00,0.05,0.25
# max_df=0.75,0.95,0.99
# --multirun
