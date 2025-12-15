from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from multilang_wsi_evaluation.vectorizer_evaluator import evaluate_vectorizer


class GeneratingMode(str, Enum):
    LEFT: str = 'left'
    RIGHT: str = 'right'
    SYMM: str = 'symm'
    LEFT_AMRAMI: str = 'amr_left'
    RIGHT_AMRAMI: str = 'amr_right'
    LEFT_FAKE_AMRAMI: str = 'fakeamr_left'
    RIGHT_FAKE_AMRAMI: str = 'fakeamr_right'


@dataclass
class Template:
    left_mask: Optional[str] = None
    right_mask: Optional[str] = None
    conj: Optional[str] = None
    par: bool = False

    def __init__(self, lang, left_mask=None, right_mask=None, conjunct=None, parenthesis=False):
        if not any([conjunct, left_mask, right_mask]):
            raise RuntimeError("Provide conjunction or left and right mask templates.")
        if conjunct and (left_mask or right_mask):
            raise RuntimeError("Cannot use conjunction with ready templates")
        if conjunct:
            self.conj = '_'.join(conjunct.split())
            self.par = parenthesis
            if parenthesis:
                self.left_mask = f'{lang}_({self.conj})_left'
                self.right_mask = f'{lang}_({self.conj})_right'
            else:
                self.left_mask = f'{lang}_{self.conj}_left'
                self.right_mask = f'{lang}_{self.conj}_right'
        else:
            assert left_mask and right_mask, "Provide both left and right masks"
            self.left_mask = left_mask
            self.right_mask = right_mask

    @property
    def left(self):
        return [self.left_mask]

    @property
    def right(self):
        return [self.right_mask]

    @property
    def symmetrical(self):
        return [self.left_mask, self.right_mask]

    @property
    def left_fake_amrami(self):
        return ['m', self.left_mask]

    @property
    def right_fake_amrami(self):
        return ['m', self.right_mask]

    @property
    def left_amrami(self):
        return ['T', self.left_mask]

    @property
    def right_amrami(self):
        return ['T', self.right_mask]

    def templ_by_mode(self, mode: int):
        __mode_mapping = {
            GeneratingMode.LEFT: self.left,
            GeneratingMode.RIGHT: self.right,
            GeneratingMode.SYMM: self.symmetrical,
            # GeneratingMode.LEFT_AMRAMI: self.left_amrami,
            # GeneratingMode.RIGHT_AMRAMI: self.right_amrami,
            # GeneratingMode.LEFT_FAKE_AMRAMI: self.left_fake_amrami,
            # GeneratingMode.RIGHT_FAKE_AMRAMI: self.right_fake_amrami,
        }
        if mode in __mode_mapping:
            return __mode_mapping[mode]
        else:
            raise RuntimeError("Unknown template generating mode")


repo_path = Path(r'/home/rombek/MSU_study/summer-wsi')
eval_config_path = repo_path / 'multilang_wsi_evaluation/conf/'
model_vec_config_path = eval_config_path / 'model_vec'
substs_config_path = model_vec_config_path / 'subst_params'

lang = 'ru'
lang_to_templates = {
    'ru': [
        Template(lang='ru', conjunct='or'),
        # Template(lang='ru', conjunct='and'),
        # Template(lang='ru', conjunct='and also', parenthesis=True),
    ],
    'en': [
        Template(lang='en', conjunct='or'),
        Template(lang='en', conjunct='and'),
        Template(lang='en', conjunct='and also'),
        Template(lang='en', conjunct='and also', parenthesis=True),
        # Template(lang='en', conjunct='or even'),
        # Template(lang='en', conjunct='or even', parenthesis=True),
    ]
}
path_to_substs = repo_path / 'SubstWSI_rombek/substs'
template_modes = [
    # GeneratingMode.LEFT,
    # GeneratingMode.RIGHT,
    GeneratingMode.SYMM,
]
subst_params = {
    'path_to_configs': substs_config_path,
    'lang_substs': {
        'en': ['en_or_left', 'en_or_right']
    }
}
subst_overrides = {
    'path_to_substs': path_to_substs,
    'model_path': "fairseq-xlmr-large/model.pt"
}


@hydra.main(config_path='../multilang_wsi_evaluation/conf/lang_eval', config_name=f'eval_ru')
def run(cfg: DictConfig):
    """
        --config-path path_to_conf --config-name config
    """
    print(f'Config:\n {OmegaConf.to_yaml(cfg)}')

    model_vec_cfg = cfg.model_vec
    lemmatizer = model_vec_cfg['lemma_config']['lemmatizer_name']
    mode = model_vec_cfg['lemma_config']['mode']
    masks_cnt = model_vec_cfg['substs_overrides']['fill_masks']

    for template in lang_to_templates[lang]:
        for templ_mode in template_modes:
            templs = template.templ_by_mode(templ_mode)
            subst_params['lang_substs'][lang] = templs

            model_vec_cfg['subst_params'] = DictConfig(subst_params)
            model_vec_cfg['substs_overrides'] = DictConfig(subst_overrides)

            # model_vec_cfg['subst_params'] = DictConfig({
            #     temp: OmegaConf.load(substs_config_path / lang / f'{temp}.yaml')
            #     for temp in templs
            # })

            model_vectorizer = instantiate(model_vec_cfg)

            eval_params = cfg.eval_params
            list_paths_datasets = eval_params.list_paths_datasets
            metrics = eval_params.get('metrics')
            callable_metrics = eval_params.get('callable_metrics')
            verbose = cfg.run_settings.get('verbose', True)
            vectorizer_name = cfg.run_settings.get('vectorizer_name')
            use_fast_semeval_metrics = eval_params.use_fast_semeval_metrics
            ret_metric = cfg.return_value['metric']
            ret_dataset = cfg.return_value['dataset']
            cfg_clusterers = cfg.model_clusterers

            if template.conj:
                save_name = \
                    f"conj{template.conj}_par{template.par}_fillmasks{masks_cnt}_templmode{templ_mode}_" \
                    f"lemmatizer{lemmatizer}_lemmamode{mode}"
            else:
                save_name = \
                    f"templs{';'.join(templs).replace('m', 'm' * masks_cnt)}_" \
                    f"lemmatizer{lemmatizer}_lemmamode{mode}"
            res = vec_eval(
                model_vectorizer,
                cfg_clusterers=cfg_clusterers,
                list_paths_datasets=list_paths_datasets,
                metrics=metrics,
                callable_metrics=callable_metrics,
                opt_metric=ret_metric,
                opt_dataset=ret_dataset,
                verbose=verbose,
                vectorizer_name=save_name,
                use_fast_semeval_metrics=use_fast_semeval_metrics
            )


if __name__ == "__main__":
    run()
