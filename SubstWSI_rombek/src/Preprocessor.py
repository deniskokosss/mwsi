import string
from itertools import chain

import pandas as pd


class Preprocessor:
    def __init__(self, disable=False):
        self.disable = disable

    def _prepare_row_word(self, df_row):
        # TODO: rework with some stanza/spacy tokenizer
        """
        Remove punctuation from substs
        and split multiple words substs into multiple substs with same prob
        """

        probs_row = df_row['substs_probs']
        remove_punc = lambda word: word.translate(str.maketrans('', '', string.punctuation))
        prep_row = [(prob, remove_punc(word)) for prob, word in probs_row]

        split_nwords_substs = lambda row: list(chain(
            *[
                [(prob, spl_word) for spl_word in subst.split()]
                for prob, subst in row
            ]
        ))
        prep_row = split_nwords_substs(prep_row)
        df_row['substs_probs'] = prep_row
        return df_row

    def fit_predict(self, df: pd.DataFrame):
        if self.disable:
            return df
        tmp_df = df.copy()
        tmp_df = tmp_df.apply(lambda row: self._prepare_row_word(row), axis=1)
        return tmp_df