import json
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import string
import typing as tp
from typing import List, Tuple, Dict

from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering
from Levenshtein import ratio

from SubstWSI_rombek.src.LemmatizerBase import (
    LemmatizerPymorphy,
    LemmatizerSpacy,
    LemmatizerWordNet,
    LemmatizerStanza,
    LemmatizerSnowballStemmer, LemmatizerNone
)


class Lemmatizer:
    def __init__(
            self,
            mode='disable',
            unite_lemmas=True,
            lemmatizer_name='spacy', lang='ru',
            lemmas_similarity_threshold=1.0,
            cache_path: str = None
    ):
        self.mode = mode
        self.unite_lemmas = unite_lemmas
        self.lemmas_similarity_threshold = lemmas_similarity_threshold
        if lemmatizer_name == 'spacy':
            self._lemmatizer = LemmatizerSpacy(mode=mode, lang=lang)
        elif lemmatizer_name == 'pymorphy':
            self._lemmatizer = LemmatizerPymorphy(mode=mode, lang=lang)
        elif lemmatizer_name == 'wordnet':
            self._lemmatizer = LemmatizerWordNet(mode=mode, lang=lang)
        elif lemmatizer_name == 'stanza':
            self._lemmatizer = LemmatizerStanza(mode=mode, lang=lang)
        elif lemmatizer_name == 'snowball':
            self._lemmatizer = LemmatizerSnowballStemmer(mode=mode, lang=lang)
        elif lemmatizer_name == 'none':
            self._lemmatizer = LemmatizerNone(mode=mode, lang=lang)
        if cache_path:
            self.cache_path: tp.Optional[Path] = Path(f'{cache_path}/{lemmatizer_name}_{mode}')
            os.makedirs(self.cache_path, exist_ok=True)
        else:
            self.cache_path = None

    def fit(self, dfs: List[pd.DataFrame]):
        pass

    def _transform_df(self, df):
        tmp_df = df.copy()
        tmp_df.substs_probs = tmp_df.progress_apply(self.lemmatize_row, axis=1)
        return tmp_df

    def search(self, input_df: pd.DataFrame, key: tuple):
        if self.cache_path:
            _df_cache_path = self.cache_path / ('-'.join(map(str, key)) + '.pkl')
        if self.cache_path and _df_cache_path.is_file():
            from_cache = pd.read_pickle(_df_cache_path).drop_duplicates(subset=['word', 'context', 'positions'])
            tmp_input_df = input_df.merge(from_cache, on=['word', 'context', 'positions'], how='left',
                                          suffixes=[None, "_lem"])
            if tmp_input_df['substs_probs_lem'].isna().any():
                mask = tmp_input_df['substs_probs_lem'].isna()

                input_df_lemmatized = pd.concat([
                    self._transform_df(input_df[mask]),
                    tmp_input_df[~mask][
                        ['context', 'word', 'lang', 'positions', 'word_at', 'substs_probs_lem']
                    ].rename(columns={'substs_probs_lem': 'substs_probs'})
                ]).sort_values(by=['word'])
                input_df_lemmatized = input_df_lemmatized[
                    ['context', 'word', 'lang', 'positions', 'word_at', 'substs_probs']
                ]
                to_cache = pd.concat([from_cache, input_df_lemmatized], axis=0)
                to_cache.to_pickle(str(_df_cache_path))
            else:
                input_df_lemmatized = tmp_input_df[
                    ['context', 'word', 'lang', 'positions', 'word_at', 'substs_probs_lem']
                ].rename(columns={'substs_probs_lem': 'substs_probs'})
        else:
            input_df_lemmatized = self._transform_df(input_df)
            if self.cache_path:
                input_df_lemmatized[
                    ['context', 'word', 'lang', 'positions', 'word_at', 'substs_probs']
                ].to_pickle(str(_df_cache_path))
        return input_df_lemmatized

    def merge_similar_lemmas(self, dfs: Dict[tuple, pd.DataFrame]) -> Dict[tuple, pd.DataFrame]:
        '''
        merge (apply agglomerative clustering) for lemmas based on pairwise levenshtein distance
        Args:
            dfs: dict where values dataframes with substs_probs column, keys are ignored
        Returns:
            dfs with some lemmas merged in substs_probs
        '''
        def cluster_lemmas(lemmas: list[str]):
            dists = np.zeros([len(lemmas), len(lemmas)], dtype=np.float16)
            levenshtein_dist = lambda x, y: 1 - ratio(x, y, score_cutoff=self.lemmas_similarity_threshold)
            for i, l1 in enumerate(tqdm(lemmas, desc='Levenshtein lemma merging')):
                for j, l2 in enumerate(lemmas[i + 1:]):
                    sim = levenshtein_dist(l1, l2)
                    dists[i, i + 1 + j] = sim
            dists = (dists + dists.T) / 2

            clusterer = AgglomerativeClustering(
                n_clusters=None, linkage='average', affinity='precomputed',
                distance_threshold=1 - self.lemmas_similarity_threshold - 1e-7
            )
            lemma_clusters = clusterer.fit_predict(dists)
            lemma_cluster_names = (
                pd.DataFrame({'lemmas': lemmas, 'clusters': lemma_clusters}).groupby('clusters').lemmas.first()
            )
            lemma_clusters = {
                lemma: lemma_cluster_names[lemma_cluster]
                for lemma, lemma_cluster in zip(lemmas, lemma_clusters)
            }
            return lemma_clusters, dists

        all_lemmas = list(set([t[1] for df in dfs.values() for substs in df['substs_probs'] for t in substs]))
        if self.cache_path:
            _df_cache_path = (
                self.cache_path / f'lemma_clustering_{sum(map(len, all_lemmas))}_{self.lemmas_similarity_threshold}.json'
            )
            if _df_cache_path.is_file():
                with open(_df_cache_path, 'r', encoding='UTF-8') as f:
                    clusters = json.load(f)
            else:
                clusters, _ = cluster_lemmas(all_lemmas)
                with open(_df_cache_path, 'w', encoding='UTF-8') as f:
                    json.dump(clusters, f, ensure_ascii=False, indent=2)
        else:
            clusters, _ = cluster_lemmas(all_lemmas)

        print(f'Reducing number of lemmas from {len(all_lemmas)} to {len(set(clusters.values()))}')
        try:
            replace_lemma_with_cluster = lambda substs_probs: [(t[0], clusters[t[1]]) for t in substs_probs]
            res = {k: df.assign(substs_probs=df.substs_probs.apply(replace_lemma_with_cluster)) for k, df in dfs.items()}
        except KeyError:
            clusters, _ = cluster_lemmas(all_lemmas)
            replace_lemma_with_cluster = lambda substs_probs: [(t[0], clusters[t[1]]) for t in substs_probs]
            res = {k: df.assign(substs_probs=df.substs_probs.apply(replace_lemma_with_cluster)) for k, df in
                   dfs.items()}
        return res

    def predict(
            self,
            dfs: Dict[tuple, pd.DataFrame] = None,
            word_inst_subst: Dict[str, List[List[str]]] = None
    ):
        print("LEMMATIZATION STARTED:")
        data = dfs or word_inst_subst
        if self.mode == 'disable':
            return dfs or word_inst_subst
        if dfs:
            lemmatized_dfs = {}
            for key, df in dfs.items():
                tmp_df = self.search(df, key)
                lemmatized_dfs[key] = tmp_df
            if self.lemmas_similarity_threshold < 0.999:
                res = {k: [] for k in lemmatized_dfs}
                for word in lemmatized_dfs[key].word.unique():
                    wres = (self.merge_similar_lemmas(
                        {key: t[t.word == word] for key, t in lemmatized_dfs.items()}))
                    for k in wres:
                        res[k].append(wres[k])
                res = {k: pd.concat(v) for k,v in res.items()}
            else:
                res = lemmatized_dfs
            return res
        else:
            lemmatized_substs = {}
            for lemma, lemma_ctxs_substs in tqdm(word_inst_subst.items(), total=len(word_inst_subst)):
                lemmatized_substs[lemma] = []
                for substs in lemma_ctxs_substs:
                    lemmatized_substs[lemma].append(self._lemmatize_row_list(substs))
            return lemmatized_substs

    def _lemmatize_row_list(self, substs: List[str]):
        lemmatized_substs = [
            self._lemmatizer.parse(word)
            for word in substs
        ]
        return lemmatized_substs

    def lemmatize_row_word(self, df_row):
        if self.unite_lemmas:
            total_prob = defaultdict(float)
            for prob, word in df_row['substs_probs']:
                word_lemma = self._lemmatizer.parse(str(word))
                if word_lemma is not None:
                    total_prob[word_lemma] += prob
            lemmatized_substs = [(prob, lemma) for lemma, prob in total_prob.items()]
        else:
            lemmatized_substs = []
            for prob, word in df_row['substs_probs']:
                word_lemma = self._lemmatizer.parse(str(word))
                if word_lemma is not None:
                    lemmatized_substs.append((prob, word_lemma))

        return lemmatized_substs

    def lemmatize_row_context(self, df_row):
        context = df_row['context']
        positions = df_row['positions']

        total_prob = defaultdict(int)
        for prob, word in df_row['substs_probs']:
            word_lemma = self._lemmatizer.parse(word=word, context=context, positions=positions)
            for word in word_lemma:
                total_prob[word] += prob

        norm_coef = sum(prob for word, prob in total_prob.items())
        new_prob_list = sorted([(prob / norm_coef, word) for word, prob in total_prob.items()], reverse=True)

        return new_prob_list

    def lemmatize_row(self, df_row):
        if self.mode == 'word':
            return self.lemmatize_row_word(df_row)
        elif self.mode == 'context':
            raise NotImplementedError
            # return self.lemmatize_row_context(df_row)

# --config-name
# eval_v1.yaml
# model_vec=subst_rombek
# hydra.run.dir=outputs/${model_vec.lemma_config.ru.lemmatizer_name}_${model_vec.lemma_config.ru.lemmas_similarity_threshold}
# --multirun
# model_vec.lemma_config.ru.lemmatizer_name=spacy,pymorphy,stanza,snowball
# model_vec.lemma_config.ru.lemmas_similarity_threshold=0.7,0.8,0.9,1.0

# --config-name
# eval_v1.yaml
# model_vec=subst_rombek
# model_vec.lemma_config.lemmatizer_name=none,stanza
# model_vec.lemma_config.lemmas_similarity_threshold=0.7,0.8,0.9,1.0
# +runname=russe
# --multirun