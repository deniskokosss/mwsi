import datetime
from pathlib import Path
import time
import typing as tp

import numpy as np
import omegaconf
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from SubstWSI_rombek.src.Lemmatizer import Lemmatizer
from SubstWSI_rombek.src.Vectorizer import Vectorizer
from SubstWSI_rombek.src.Preprocessor import Preprocessor
from SubstWSI_rombek.src.AmramiVectorizer import AmramiVectorizer
from substwsi.PrimarySubst import PrimarySubst

from multilang_wsi_evaluation.interfaces import IWSIVectorizer, Sample


class VectorizingPipeline(IWSIVectorizer):
    def __init__(
            self,
            subst_params: tp.Dict,  # subst_params for every language
            substs_overrides: tp.Dict,  # overriding params for substs
            preprocessor_config: tp.Dict,
            lemma_config: tp.Dict,
            vec_config: tp.Dict,
            mode='test', verbose=1
    ):

        self._subst_params = self._instantiate_substs_cfgs(subst_params)
        self._substs_overrides = substs_overrides
        self.lang_primary_subst = dict()  # init PrimarySubst for each lang and set of subst params

        self._lemma_config = lemma_config
        self.lang2lemmatizer = None

        self.preprocessor: Preprocessor = Preprocessor(**preprocessor_config)
        self.vec_mode = vec_config.pop('vec_mode')
        if self.vec_mode == 'amrami':
            self.vectorizer: AmramiVectorizer = AmramiVectorizer(**vec_config)
            print("CHECK FOR USING AMRAMI CLUSTERER WITH AMRAMI VECTORIZER")
            self.word_inst_substs = dict()
            self.word_inst_probs = dict()
        else:
            self.vectorizer: Vectorizer = Vectorizer(**vec_config)
        self.mode = mode
        self.verbose = verbose

        # Result vecs
        self.lang_vectorized_words_substs: tp.Dict[str, tp.Dict[str, np.ndarray]] = dict()

    def _instantiate_substs_cfgs(self, subst_params: tp.Dict):
        lang2cfgs = dict()
        path = Path(subst_params.pop('path_to_configs'))
        for lang, cfgs in subst_params['lang_substs'].items():
            lang2cfgs[lang] = dict()

            for cfg in cfgs:
                if 'mask_count' in subst_params:
                    for cnt in subst_params['mask_count']:
                        d = dict(OmegaConf.load(path / lang / f'{cfg}.yaml'))
                        key = list(d.keys())[0]
                        d[key]['fill_masks'] = cnt
                        d[key + str(cnt)] = d[key]
                        del d[key]
                        lang2cfgs[lang].update(d)
                else:
                    d = OmegaConf.load(path / lang / f'{cfg}.yaml')
                    OmegaConf.resolve(d)
                    lang2cfgs[lang].update(dict(d))

        return lang2cfgs

    def _create_lang_overrides(self, lang: str, overrides: tp.Dict):
        base_overrides = {
            k: v
            for k, v in overrides.items() if not isinstance(v, (dict, omegaconf.DictConfig))
            # loading overrides for all languages
        }
        lang_overrides = overrides.get(lang, {})
        base_overrides.update(lang_overrides)
        return base_overrides

    def _prepare_substs(self, lang: str, subst_params: tp.Dict, substs_overrides: tp.Dict):
        data_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        lang_params = subst_params[lang]
        overrides = self._create_lang_overrides(lang=lang, overrides=substs_overrides)
        substs = dict()
        for key, params_set in lang_params.items():
            tmp_params = params_set
            tmp_params.update(overrides)
            assert 'path_to_substs' in tmp_params
            tmp_params['path_to_substs'] = str(tmp_params['path_to_substs']) + f"/{lang}"
            substs[key] = PrimarySubst(**tmp_params, data_name=data_name)
        return substs

    def _prepare_lemmatizers(self, langs: tp.Set[str], lemma_config: tp.Dict) -> tp.Dict[str, Lemmatizer]:
        default_cfg = {
            k: v for k, v in lemma_config.items() if not isinstance(v, (dict, omegaconf.DictConfig))
        }
        # init default lemmatizer for non overrided languages
        default_langs = {lang for lang in langs if lang not in lemma_config.keys()}
        if 'lang' in default_cfg:
            default_langs = {default_cfg['lang'] for _ in default_langs}
            del default_cfg['lang']
        lang2lemmatizer = {
            lang: Lemmatizer(lang=lang, **default_cfg)
            # lang: Lemmatizer(lang='en', **default_cfg)
            for lang in default_langs
        }
        # apply overrides and init other lemmatizers
        non_default_langs = langs - default_langs
        for lang in non_default_langs:
            # Special config for language
            lang_config = self._create_lang_overrides(lang=lang, overrides=lemma_config)
            lang2lemmatizer[lang] = Lemmatizer(**lang_config)
        return lang2lemmatizer

    def _load_files_dfs(self, lang: str, samples: tp.List[Sample]):
        """Load all files matches substs_config"""

        keys: tp.List[str] = list(self.lang_primary_subst[lang].keys())
        files_dfs: tp.Dict[str, pd.DataFrame] = {
            (key, self.lang_primary_subst[lang][key].fill_masks): self.lang_primary_subst[lang][key].search(samples)
            for key in keys
        }
        ind2word = dict(enumerate(list(files_dfs.values())[0].word.unique()))
        word2ind = {word: ind for ind, word in ind2word.items()}
        return files_dfs, word2ind



    def lemmatize_probs_dfs(self, lang: str, files_dfs: tp.Dict[tuple, pd.DataFrame]):
        """
            Lemmatize all probs from dataset df
        """
        print("Lemmatize started")
        st = time.time()

        if lang not in self.lang2lemmatizer:
            lang = 'default'
        lemmatized_file_dfs = self.lang2lemmatizer[lang].predict(dfs=files_dfs)

        print(f"Lemmatize finished in {time.time() - st} sec.")
        return lemmatized_file_dfs

    def lemmatize_substs(self, lang: str, word_inst_subst: tp.Dict[str, tp.List[tp.List[str]]]):
        print("Lemmatize started")
        st = time.time()

        if lang not in self.lang2lemmatizer:
            lang = 'default'
        lemmatized_file_dfs = self.lang2lemmatizer[lang].predict(word_inst_subst=word_inst_subst)

        print(f"Lemmatize finished in {time.time() - st} sec.")
        return lemmatized_file_dfs

    def vectorize_probs(self,files_dfs: tp.Dict[tuple, pd.DataFrame], word2ind: tp.Dict):
        vecs = self.vectorizer.transform({
            uniq_word: [df[df.word == uniq_word] for key, df in files_dfs.items()]
            for uniq_word in word2ind
        })
        return vecs

    def _vectorize_amrami(self, files_dfs: tp.Dict[tuple, pd.DataFrame], word2ind: tp.Dict):
        print("VECTORIZER TRANSFORM STARTED: ")
        lemma_files_dfs: tp.Dict[str, tp.List[pd.DataFrame]] = {
            uniq_word: {key: df[df.word == uniq_word] for key, df in files_dfs.items()}
            for uniq_word in word2ind
        }
        lemma_vectors = dict()
        for lemma, lemma_dfs in tqdm(lemma_files_dfs.items()):
            lemma_vectors[lemma] = self.vectorizer.transform(
                lemma=lemma, files_dfs=lemma_dfs
            )
        return lemma_vectors

    def fit(self, samples: tp.List[Sample]) -> None:
        used_langs = {samp.lang for samp in samples}
        # Setting up lemmatizers
        self.lang2lemmatizer = self._prepare_lemmatizers(langs=used_langs, lemma_config=self._lemma_config)
        # Loading substs
        print(f"Used languages are: {used_langs}")
        lang_to_samples = {
            lang: [sample for sample in samples if sample.lang == lang]
            for lang in used_langs
        }
        for lang, lang_samples in lang_to_samples.items():
            print(f"Solving for lang {lang}")
            self.lang_primary_subst[lang] = self._prepare_substs(
                lang=lang,
                subst_params=self._subst_params,
                substs_overrides=self._substs_overrides,
            )
            print("Loading substs started")
            st = time.time()
            files_dfs, word2ind = self._load_files_dfs(lang=lang, samples=lang_samples)
            files_dfs = {
                key: self.preprocessor.fit_predict(df)
                for key, df in files_dfs.items()
            }

            print(f"Loading substs finished in {time.time() - st}")
            files_dfs = self.lemmatize_probs_dfs(lang=lang, files_dfs=files_dfs)
            if self.vec_mode == 'amrami':
                self.lang_vectorized_words_substs[lang] = self._vectorize_amrami(
                    files_dfs=files_dfs,
                    word2ind=word2ind
                )
            else:
                self.lang_vectorized_words_substs[lang] = self.vectorize_probs(
                    files_dfs=files_dfs,
                    word2ind=word2ind
                )

    def predict(self, samples: tp.List[Sample]) -> tp.Any:
        """
            samples: all samples for one distinct word from dataset
        """
        lang = samples[0].lang
        lemma = samples[0].lemma
        samples_vectors = self.lang_vectorized_words_substs[lang][lemma]
        return samples_vectors
