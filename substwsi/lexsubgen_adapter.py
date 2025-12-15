import typing as tp
import json
from pathlib import Path

import lexsubgen.prob_estimators.combiner
import nltk
import sklearn.base
from nltk.tokenize import word_tokenize
from lexsubgen import SubstituteGenerator
from lexsubgen.utils.batch_reader import BatchReader
from omegaconf import OmegaConf, DictConfig, ListConfig
from tqdm import tqdm
from hydra.utils import instantiate
import numpy as np
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from interfaces import Sample
from multilang_wsi_evaluation.interfaces import IWSIVectorizer

import logging


class LexSubGenAdapter(IWSIVectorizer):
    def __init__(self,
                 base_substgen: tp.Dict,
                 lang_substgen_overrides: tp.Dict,
                 subst_vectorizer: tp.Union[TfidfVectorizer, CountVectorizer],
                 batch_size: int=1,
                 cache_path: str='default'
    ):
        nltk.download("punkt")
        self.base_substgen_params = base_substgen
        self.lang_substgen_params = lang_substgen_overrides
        self.subst_vectorizer = subst_vectorizer
        self.batch_size = batch_size
        self.subst_generator = None
        self.cur_lang = ''
        self.cache_path = cache_path
        self._prepare_cache(cache_path)

    def _prepare_cache(self, cache_path):
        if Path(cache_path).exists():
            logging.info(f"Loading cache from {cache_path}")
            with open(cache_path) as f:
                self.cache = json.load(f)
        else:
            Path(cache_path).parent.mkdir(exist_ok=True)
            self.cache = {}


    def _from_samples_to_tokens_lists(self, samples: tp.List[Sample]) \
            -> tp.List[tp.Tuple[tp.List[str], tp.List[int], str]]:
        res = []
        for sample in samples:
            left_context, target, right_context = (
                sample.context[:sample.begin],
                sample.context[sample.begin:sample.end],
                sample.context[sample.end:]
            )
            left_context = word_tokenize(left_context)
            right_context = word_tokenize(right_context)
            context = left_context + [target] + right_context
            target_id = len(left_context)
            if f"_{target_id}_" + " ".join(context) in self.cache:
                cached_substs = " ".join(
                    [t for t in self.cache[f"_{target_id}_" + " ".join(context)][0].split(' ')]
                    [:self.base_substgen_params.top_k]
                )
            else:
                cached_substs = None
            res.append((context, target_id, sample.lang, cached_substs))
        return res

    def _prepare_conf_from_instantiate(self, d: DictConfig):
        for k, v in d.items():
            if k == 'class':
                d['_target_'] = v
                del d['class']
            elif isinstance(v, DictConfig):
                self._prepare_conf_from_instantiate(v)
            elif isinstance(v, ListConfig):
                for el in v:
                    self._prepare_conf_from_instantiate(el)
        return d

    def _merge_lang_param(self, base, k_new, v_new):
        for k, v in base.items():
            if k == k_new:
                base[k] = v_new
            elif isinstance(v, DictConfig):
                self._merge_lang_param(v, k_new, v_new)
            elif isinstance(v, ListConfig):
                for el in v:
                    self._merge_lang_param(el, k_new, v_new)
        return base

    def _reload_subst_generator(self, lang: str):
        for k, v in self.lang_substgen_params[lang].items():
            params = self._merge_lang_param(self.base_substgen_params, k, v)
        params = self._prepare_conf_from_instantiate(params)
        self.subst_generator = instantiate(params)
        self.cur_lang = lang
        logging.info(f"Loaded model for {lang}")

    def predict(self, samples: tp.List[Sample]) -> tp.Any:
        data_reader = BatchReader(samples, batch_size=self.batch_size)

        substs = []
        for batch in tqdm(data_reader):
            batch = batch[0]
            sentences, target_ids, langs, cached_substs = zip(*self._from_samples_to_tokens_lists(batch))
            if all(cached_substs):
                substs += cached_substs
            else:
                if langs[0] != self.cur_lang:
                    self._reload_subst_generator(langs[0])
                # Computing probability distribution over possible substitutes
                probs, word2id = self.subst_generator.get_probs(
                    sentences, target_ids
                )

                # Selecting most probable substitutes from the obtained distribution
                pred_substitutes = self.subst_generator.substitutes_from_probs(
                    probs, word2id, sentences,
                )
                pred_substitutes = [" ".join(t) for t in pred_substitutes]
                for sent, target in zip(sentences, target_ids):
                    self.cache[f"_{target}_" + " ".join(sent)] = pred_substitutes
                substs += pred_substitutes

        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, ensure_ascii=False)
        vectors = self.subst_vectorizer.fit_transform(substs).toarray()
        return vectors
