import argparse
from datetime import datetime
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
import typing as tp

from multilang_wsi_evaluation.utils import load_parts
from substwsi.PrimarySubst import PrimarySubst

parser = argparse.ArgumentParser(description='Generate substitutions for datasets')
parser.add_argument('--dataset_path', type=str, help='Path to target datasets')
parser.add_argument('--substs_configs_path', type=Path, help='Path to folder with base (1 mask) configs')
parser.add_argument('--mask_number', type=int, default=1, help='Number of mask to use in substitutions')
parser.add_argument('--out_path', type=str, help='Path to result substs')
parser.add_argument('--model_path', type=str, help='Path to fairseq model checkpoint', default=None)
parser.add_argument('--lang', type=str, help='Filters dataset for samples with given lang', default=None)

if __name__ == "__main__":
    args = parser.parse_args()

    cur_dir = Path.cwd()
    substs_params_paths = list(args.substs_configs_path.glob('*.yaml'))
    subst_params = dict()

    for param_path in substs_params_paths:
        templ = param_path.stem
        param_dict = OmegaConf.load(param_path)[templ]
        param_dict['fill_masks'] = args.mask_number
        if args.model_path and args.model_path != "None":
            param_dict['model_path'] = args.model_path
        subst_params[templ] = param_dict

    primary_subst = dict()  # init PrimarySubst for each set of subst params
    data_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    for part in load_parts(args.dataset_path):
        print("DATASET: ", part.id)
        part_out_path = Path(args.out_path) / part.id
        for key, params_set in subst_params.items():
            primary_subst[key] = PrimarySubst(**params_set, path_to_substs=str(part_out_path), data_name=data_name)

        group_ids, groups_samples_ids, groups_samples = part.groups_samples()
        all_samples = [sample for group_samples in groups_samples for sample in group_samples]
        if args.lang:
            all_samples = [sample for sample in all_samples if sample.lang == args.lang]
        print("TOTAL SAMPLES: ", len(all_samples))

        files_dfs: tp.Dict[str, pd.DataFrame] = dict()
        for key in primary_subst:
            print(f"GENERATING FOR PATTERN {key}")
            files_dfs[key] = primary_subst[key].search(all_samples)
