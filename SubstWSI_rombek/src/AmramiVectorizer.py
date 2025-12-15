from itertools import chain
from typing import List, Tuple, Optional
import typing as tp

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
)
import pandas as pd


class AmramiVectorizer:
    """
        Take all texts with probs from all documents and vectorize is with some method.
    """

    def __init__(
            self,
            ood_mode='res_mean', vocab_size=30522,
            ood_prob=None, norm_probs=False,
            comb_mode='amrami',
            vec_transform: Optional[str] = None, min_df: float = 0.03, max_df: float = 0.95,
            top_k=200,
    ):
        """
            args:
                vec_mode:           type of Vectorizer from sklearn (or mb custom)
                with_probs:         multiply vectorizer numbers by prob. of subst
                comb_mode:          mode of combining probs from files
                ood_prob:           probability for words appearing only in one file
                vectorizer_conf:    settings for vectorizer

        """
        self.norm_probs = norm_probs
        self.ood_mode = ood_mode
        self.ood_prob = ood_prob
        self.vocab_size = vocab_size
        if self.ood_mode == 'res_mean':
            assert self.vocab_size is not None, "Set vocab size for res_mean ood_prob"
        self.top_k = top_k
        self.comb_mode = comb_mode
        self.comb_probs = {
            'amrami': self.comb_probs_amrami,
            'prod': np.prod,
            'mean': np.mean,
        }
        self.vec_transform = vec_transform
        self.transform_min_df = min_df
        self.transform_max_df = max_df

    def comb_probs_amrami(self, probs: Tuple[int]):
        tmp_probs = np.array(probs)
        return np.sum(np.log(tmp_probs) * 0.4)

    def unite_n_rows(self, probs_rows_df: Tuple[List[Tuple[float, str]]]):
        """
            Unite n rows from different files into one row and leaves only top_k substs.
        """
        probs_rows = tuple(probs_rows_df)

        if self.ood_mode == 'res_mean':
            ood_probs = [
                max(
                    np.finfo(np.float64).eps,
                    (1 - sum(prob for prob, word in row)) / (self.vocab_size - len(row))
                )
                for row in probs_rows
            ]  # probs if subst from other file not presented
        else:
            ood_probs = [self.ood_prob for row in probs_rows]

        all_substs = set(word for prob, word in chain(*probs_rows) if word != '')
        file_word_probs = [
            {
                file_row_word: file_row_prob
                for file_row_prob, file_row_word in p_s_row
            }
            for p_s_row in probs_rows
        ]
        substs_probs = {
            word: tuple(file_probs.get(word, ood_probs[ind]) for ind, file_probs in enumerate(file_word_probs))
            for word in all_substs
        }
        new_substs_probs = {
            word: self.comb_probs[self.comb_mode](w_probs)
            for word, w_probs in substs_probs.items()
        }
        substs = np.array(list(new_substs_probs.keys()))
        probs = np.array(list(new_substs_probs.values()))

        idxs = (-probs).argsort()
        substs = substs[idxs][:self.top_k]
        probs = probs[idxs][:self.top_k]

        probs = softmax(probs)
        new_probs_substs = list(zip(probs, substs))
        return new_probs_substs

    def unite_dfs(self, lemma: str, lemma_dfs: tp.Dict[tuple, pd.DataFrame]) -> np.ndarray:
        # Merge many dfs into one, to perform apply later
        dfs_list = list(lemma_dfs.values())
        res: pd.DataFrame = dfs_list[0]
        for ind, df in enumerate(dfs_list[1:]):
            res = pd.merge(res, df, left_index=True, right_index=True, suffixes=(None, f'{ind + 1}'))
        res = res.drop(columns=list(res.filter(regex='word\d+')))

        # Merging probs from different files
        new_probs = res.filter(regex='probs').apply(self.unite_n_rows, axis=1)
        res = res.drop(columns=list(res.filter(regex='substs_probs')))
        res['substs_probs'] = new_probs

        return res

    def vectorize_lemma_df(self, lemma: str, lemma_df: pd.DataFrame):
        insts_substs = [
            [s for (p, s) in inst_sp]
            for inst_sp in lemma_df['substs_probs']
        ]
        base_vec = CountVectorizer(
            preprocessor=lambda x: x, tokenizer=lambda x: x, dtype=np.float64,
            min_df=self.transform_min_df, max_df=self.transform_max_df,
        )
        bow_vectors = base_vec.fit_transform(insts_substs)
        filtered_substs_probs = [
            [(p, s) for (p, s) in inst_sp if s in base_vec.vocabulary_]
            for inst_sp in lemma_df['substs_probs']
        ]

        insts_substs = [
            [s for (p, s) in inst_sp]
            for inst_sp in filtered_substs_probs
        ]
        insts_probs = [
            [p if (s is not None and s != lemma) else 0.0 for (p, s) in inst_sp]
            for inst_sp in filtered_substs_probs
        ]
        for i, inst_probs in enumerate(insts_probs):
            insts_probs[i] /= sum(inst_probs)
        substs_idxs = [[base_vec.vocabulary_[s] for s in inst_subs] for inst_subs in insts_substs]
        for i, probs in enumerate(insts_probs):
            bow_vectors[i, substs_idxs[i]] = insts_probs[i]

        bow_vectors = bow_vectors.toarray()
        if self.vec_transform == 'binary':
            bow_vectors = (bow_vectors > 0).astype(np.float32)
        if self.vec_transform == 'tfidf':
            bow_vectors = TfidfTransformer().fit_transform(bow_vectors).toarray()
        return bow_vectors

    def transform(
            self, lemma: str, files_dfs: tp.Dict[tuple, pd.DataFrame],
    ) -> np.ndarray:
        united_lemma_df = self.unite_dfs(lemma=lemma, lemma_dfs=files_dfs)
        lemma_vecs = self.vectorize_lemma_df(lemma=lemma, lemma_df=united_lemma_df)
        return lemma_vecs
