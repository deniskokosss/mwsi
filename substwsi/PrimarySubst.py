import os.path
import typing as tp
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from multilang_wsi_evaluation.interfaces import Sample
from substwsi.sgen_xlm_opt import generate_substitutes as generate_substitutes_xlm
from substwsi.sgen_bert_opt import generate_substitutes as generate_substitutes_bert
from substwsi.substs_loading_old import load_substs


@dataclass
class PrimarySubst:
    """ Tasks and features:
        - Not associated with certain dataset but certain set of params
        - Save information about all subst with the specified params
        - Search necessary subst for specified samples
            * search(tp.List[Sample]) -> pd.DataFrame
        - Generate new subst if there is not suitable
            * __generate(pd.DataFrame) - for inner usage
    """

    topk: int = 500
    templ: str = '<mask>'
    maxlen: int = 510
    version: int = 3
    fill_masks: int = 2
    model_path: str = None
    fix_spaces: bool = True
    cont_greedy: bool = True
    beam_search: bool = True
    max_ex_per_word: int = None
    skip_last_nmasks: int = 0

    limit: int = None
    skip_rows: int = None

    debug: bool = True
    rewrite_existing: bool = True

    path_to_substs: str = './'
    data_name: str = 'new'  # for generation new substs
    weight: float = 0  # for weighted average combination

    def __post_init__(self) -> None:
        """ __post_init__ starts automatically immediately after __init__
        Save in self.__s_dir directory name that stores necessary substs (current and future generated) - cash
        """

        self.__s_dir: Path = Path(self.path_to_substs)
        self.__s_dir.mkdir(parents=True, exist_ok=True)

        self.__file_name: str = self.__get_file_name()  # ends with .npz.input

    @property
    def s_dir(self) -> Path:
        return self.__s_dir

    def search(self, all_samples: tp.List[Sample]) -> pd.DataFrame:
        """ Return pd.DataFrame with necessary substs for all_samples"""

        # load all .npz.input files as pd.DataFrame's

        list_tmp = [self.__load_df(sub_path) for sub_path in self.s_dir.rglob('*' + self.__file_name)
                    if os.path.exists(sub_path.with_suffix(''))]

        subst_input = pd.concat(list_tmp, axis=0, ignore_index=True) if len(list_tmp) > 0 \
            else pd.DataFrame(columns=['context_id', 'word', 'gold_sense_id', 'predict_sense_id', 'positions',
                                       'context', 'word_at', 'from_file'])

        subst_input = subst_input.drop_duplicates(subset=['word','context', 'positions'])
        all_samples: pd.DataFrame = self.__samples_to_df(all_samples)

        # Context with positions fully describes each sample
        # i.e. 'context' + 'positions' is a unique key for each sample
        samples_with_subst = subst_input.merge(all_samples, how='inner', on=['word', 'context', 'positions'])
        searched_files = set(samples_with_subst['from_file'].to_list())
        del subst_input

        # Marked samples (which have subst) in all_samples pd.DataFrame with True value
        mask = all_samples['context'].isin(samples_with_subst['context'])
        mask = mask & all_samples['positions'].isin(samples_with_subst['positions'])

        # Select samples without generated substs and call __generate for them
        samples_without_subst = all_samples[~mask]
        if not samples_without_subst.empty:
            file_path = self.__generate(samples_without_subst)
            searched_files.add(file_path)

        # load .npz files and form pd.DataFrame with substs
        df = pd.concat(
            [load_substs(fn)[['word', 'positions', 'word_at', 'context', 'substs_probs']] for fn in searched_files],
            axis=0,
            ignore_index=True
        )
        df = df.drop_duplicates(subset=['word', 'context', 'positions'])
        df = all_samples.merge(
            df, how='inner', on=['word', 'context', 'positions'], validate='many_to_one',
        )  # get only necessary subst

        return df

    def __generate(self, samples: pd.DataFrame) -> str:
        """ Generate new files (.npz and .npz.input) with subst for samples, call update_subst_input,
        return file name (: str) (.npz)"""
        print(samples.head(5))

        file_path: Path = self.s_dir / f'{self.data_name}-{self.__file_name}'  # .npz.input
        file_path = file_path.with_suffix('')  # .npz

        dd: Path = file_path.parent
        dd.mkdir(parents=True, exist_ok=True)

        # turn on simple logging
        origin_stdout = sys.stdout
        log_file = open(str(file_path.with_suffix('').with_suffix('.log')), 'w')
        sys.stdout = log_file

        # generate_substitutes returns value (: Path) which equals file_path
        if 'bert' in self.model_path:
            generate_substitutes_bert(data_name=[dd, file_path, samples], topk=self.topk, templ=self.templ,
                                      fill_masks=self.fill_masks, fix_spaces=self.fix_spaces, debug=self.debug,
                                      model_path=self.model_path, maxlen=self.maxlen, beam_search=self.beam_search,
                                      max_ex_per_word=self.max_ex_per_word, version=self.version,
                                      cont_greedy=self.cont_greedy,
                                      rewrite_existing=self.rewrite_existing, skip_last_nmasks=self.skip_last_nmasks
                                      )
        else:
            generate_substitutes_xlm(data_name=[dd, file_path, samples], topk=self.topk, templ=self.templ,
                                     fill_masks=self.fill_masks, fix_spaces=self.fix_spaces, debug=self.debug,
                                     model_path=self.model_path, maxlen=self.maxlen, beam_search=self.beam_search,
                                     max_ex_per_word=self.max_ex_per_word, version=self.version,
                                     cont_greedy=self.cont_greedy,
                                     rewrite_existing=self.rewrite_existing, skip_last_nmasks=self.skip_last_nmasks)

        # turn off logging
        sys.stdout = origin_stdout
        log_file.close()

        return str(file_path)

    def __get_file_name(self) -> str:
        model_path = Path(self.model_path) if self.model_path is not None else None
        if model_path is None:
            model_name = 'None'
        elif '/' in str(model_path) or '\\' in str(model_path):
            model_name = model_path.parent.name + '_' + model_path.name
        else:
            #  If we want to use pretrained model from net
            model_name = model_path

        templ = self.templ.replace('_', ' ').replace('<mask>', '_mask_' * self.fill_masks)

        # now
        name = f'limit{self.limit}-maxexperword{self.max_ex_per_word}-maxlen{self.maxlen}'
        name += f'/model{model_name}-beamsearch{self.beam_search}-contgreedy{self.cont_greedy}-version{self.version}'
        name += f'/{templ.replace(" ", "-")}-2ltr{self.fill_masks}f{self.skip_last_nmasks}s_topk{self.topk}' \
                f'_fixspaces{self.fix_spaces}.npz.input'

        return name

    @staticmethod
    def __load_df(sub_path: Path) -> pd.DataFrame:
        df: pd.DataFrame = load_input(sub_path)
        df = df[['word', 'positions', 'context']]
        df['from_file'] = str(sub_path.with_suffix(''))  # (.npz files)
        return df

    @staticmethod
    def __samples_to_df(all_samples: tp.List[Sample]) -> pd.DataFrame:
        """ Form pd.DataFrame ['word', 'context', 'positions']"""
        df = pd.DataFrame(data=all_samples).rename(columns={'lemma': 'word'})
        df['positions'] = df.apply(lambda r: (r.begin, r.end), axis=1)
        df = df.drop(['begin', 'end'], axis=1)
        return df


def load_input(sub_path: Path):
    df = pd.read_csv(sub_path)
    df['positions'] = df['positions'].apply(pd.eval).apply(tuple)
    df['word_at'] = df.apply(lambda r: r.context[slice(*r.positions)], axis=1)
    df.reset_index(inplace=True)
    return df
