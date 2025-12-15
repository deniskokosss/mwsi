import time
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import gmean, hmean
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.feature_selection import (
    chi2,
    mutual_info_classif,
)
import pandas as pd
from tqdm import tqdm
import pickle
import os


class Vectorizer:
    """
        Take all texts with probs from all documents and vectorize is with some method.
    """

    def __init__(self,
                 base_vec='count',
                 comb_mode='harm', ood_mode='eps', ood_prob=1e-7, top_k=150, vocab_size=None,
                 norm_probs=False,
                 feature_select_mode=None, keep_imp_proc=0.5,
                 patterns_probs=None,
                 save_probs_path=None,
                 **vectorizer_conf):
        """
            args:
                base_vec:           type of Vectorizer from sklearn (or mb custom)
                with_probs:         multiply vectorizer numbers by prob. of subst
                comb_mode:          mode of combining probs from files
                ood_prob:           probability for words appearing only in one file
                vectorizer_conf:    settings for vectorizer
                patterns_probs:       list of probabilities for each dynamic patter (substs_params.lang_substs)

        """
        self.norm_probs = norm_probs
        if base_vec == 'count':
            self.vectorizer = CountVectorizer(
                preprocessor=lambda x: x,
                **vectorizer_conf
            )
        elif base_vec == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                preprocessor=lambda x: x,
                **vectorizer_conf
            )
        self.comb_mode = comb_mode
        self.ood_mode = ood_mode
        self.ood_prob = ood_prob
        self.patterns_probs = patterns_probs
        self.vocab_size = vocab_size
        if self.ood_mode == 'res_mean':
            assert self.vocab_size is not None, "Set vocab size for res_mean ood_prob"
        self.top_k = top_k
        self.comb_mapping = {
            'geom': gmean,
            'harm': hmean,
            'mean': np.mean,
            'prod': np.prod,
        }
        self.important_words = None
        self.feature_selection_mode = feature_select_mode
        self.keep_important_proc = keep_imp_proc
        self.save_probs_path = save_probs_path

    def comb_prob(self, probs: Tuple[float]):
        return self.comb_mapping[self.comb_mode](probs)

    def unite_n_rows(self, probs_rows_df: Tuple[List[Tuple[float, str]]]):
        """
            Unite n rows from different files into one row and leaves only top_k substs.
        """
        probs_rows = tuple(probs_rows_df)
        XML_DICT_SIZE = 2.5 * 10 ** 6

        if self.ood_mode == 'res_mean':
            ood_probs = [
                (1 - sum(prob for prob, word in row)) / (self.vocab_size - len(row))
                for row in probs_rows
            ]  # probs if subst from other file not presented
        else:
            ood_probs = [self.ood_prob for row in probs_rows]

        if not self.patterns_probs:
            self.patterns_probs = [1 / len(probs_rows_df) for _ in probs_rows_df]

        all_substs = set(word for prob, word in chain(*probs_rows) if word != '')
        file_word_probs = [
            {
                file_row_word: file_row_prob
                for file_row_prob, file_row_word in p_s_row
            }
            for p_s_row in probs_rows
        ]
        substs_probs = {
            word: tuple(self.patterns_probs[ind] * file_probs.get(word, ood_probs[ind])
                        for ind, file_probs in enumerate(file_word_probs))
            for word in all_substs
        }
        new_probs_substs = sorted([(self.comb_prob(w_probs), word) for word, w_probs in substs_probs.items()],
                                  reverse=True)[:self.top_k]
        # Norm probs to sum 1 to sample later
        norm_coef = sum(prob for (prob, _) in new_probs_substs)
        new_probs_substs = [
            (prob / norm_coef, word) for (prob, word) in new_probs_substs
        ]
        return new_probs_substs[:self.top_k]

    def unite_words_dfs(self, splitted_word_dfs: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        words_substs_probs = dict()
        for word, word_dfs in tqdm(splitted_word_dfs.items(), total=len(splitted_word_dfs)):
            # Merge many dfs into one, to perform apply later
            res: pd.DataFrame = word_dfs[0]
            for ind, df in enumerate(word_dfs[1:]):
                res = pd.merge(res, df, left_index=True, right_index=True, suffixes=(None, f'{ind + 1}'))
            res = res.drop(columns=list(res.filter(regex='word\d+')))

            # Merging probs from different files
            new_probs = res.filter(regex='probs').apply(self.unite_n_rows, axis=1)
            res = res.drop(columns=list(res.filter(regex='substs_probs')))
            res['substs_probs'] = new_probs

            words_substs_probs[word] = res

        return words_substs_probs

    def vectorize_word_df(self, word_df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        if self.important_words is not None:
            text_from_row = lambda row: ' '.join([word for prob, word in row if word in self.important_words])
        else:
            text_from_row = lambda row: ' '.join([word for prob, word in row])
        df_texts = list(word_df.substs_probs.apply(text_from_row))
        vectorized_texts = self.vectorizer.fit_transform(df_texts).toarray()
        return vectorized_texts, self.vectorizer.vocabulary_

    def feature_selection(self, word_dfs: Dict[str, pd.DataFrame]):
        """
            Check chi2/mutual_inf for every subst in dataset and keeps set of some percent of most important words
        """
        word_dfs_lists = list(word_dfs.values())
        all_words_df = pd.concat(word_dfs_lists)
        all_words_vectorized, all_words_vocab = self.vectorize_word_df(all_words_df)
        all_words_labels = np.concatenate([np.repeat(ind, df.shape[0]) for ind, df in enumerate(word_dfs_lists)])

        print("FEATURE SELECTION makes brrrr")

        ind2word = {ind: word for word, ind in all_words_vocab.items()}
        num_remove = int(len(ind2word) * (1 - self.keep_important_proc))
        if self.feature_selection_mode == 'chi2':
            feature_imp = chi2(all_words_vectorized, all_words_labels)[0]
        else:
            feature_imp = mutual_info_classif(all_words_vectorized, all_words_labels, discrete_features=True)
        imp_order = np.argsort(feature_imp)
        self.important_words = {ind2word[ind] for ind in imp_order[num_remove:]}

        print("SELECTION FINISHED")

    def transform(
            self, words_dfs_list: Dict[tuple, List[pd.DataFrame]],
    ) -> Dict[str, np.ndarray]:
        print("VECTORIZER TRANSFORM STARTED: ")
        st = time.time()
        words_dfs_list: Dict[str, pd.DataFrame] = self.unite_words_dfs(words_dfs_list)
        if self.feature_selection_mode is not None:
            self.feature_selection(words_dfs_list)
        words_vecs = {
            word: self.vectorize_word_df(df)
            for word, df in words_dfs_list.items()
        }
        if self.save_probs_path:
            os.makedirs(self.save_probs_path, exist_ok=True)
            for word in words_vecs:
                with open(os.path.join(self.save_probs_path, word + '.pkl'), 'wb') as f:
                    pickle.dump(words_vecs[word], f)
        words_vecs = {w: v[0] for w,v in words_vecs.items()}
        print(f"Vectorizing finished in {time.time() - st} sec.")
        return words_vecs
