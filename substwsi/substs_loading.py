import csv
import os
import typing as tp
from collections import Counter
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import regex as re
from fire import Fire
from sklearn.feature_extraction import DictVectorizer


def intersect_sparse(substs_probs: pd.Series, substs_probs_y: pd.Series, nmasks: int = 1, s: float = 0):
    print('intersect_sparse, arg1:', substs_probs.iloc[0][:10],
          'intersect_sparse, arg2:', substs_probs_y.iloc[0][:10], sep='\n')

    vec = DictVectorizer(sparse=True)
    f1 = substs_probs.apply(lambda r: {subst: prob for prob, subst in r})
    f2 = substs_probs_y.apply(lambda r: {subst: prob for prob, subst in r})

    vec.fit(list(f1) + list(f2))
    f1, f2 = (vec.transform(list(f)) for f in (f1, f2))  # shape (num_of_ex, num_of_all_substs_from_both_fs)

    # 250000 - the num of potential subwords which XLM-R can generate; 250000**nmasks - the num of potential substitutes
    # alpha1 and alpha2 like a probability for 'missing' words for f1 and f2 respectively; shape (num_of_ex, 1)
    alpha1, alpha2 = ((1. - f.sum(axis=-1).reshape(-1, 1)) / 250000**nmasks for f in (f1, f2))

    # + alpha1*alpha2 is ignored to preserve sparsity
    # finally, we don't want substs with 0 probs before smoothing in both distributes
    prod = f1.multiply(f2) + f1.multiply(alpha2) + f2.multiply(alpha1)  # element-wise multiply; get 'common' probs

    fn = np.array(vec.feature_names_)  # get all substs from both f1 and f2
    maxlen = (substs_probs_y.apply(len) + substs_probs.apply(len)).max()

    # print(prod.shape, type(prod))
    # idx = ( row.toarray()[0].argsort(axis=-1)[:-maxlen-1:-1] for row in prod )

    # todo: understand
    idx = ([prod.indices[prod.indptr[i]:prod.indptr[i+1]][np.argsort(prod.data[prod.indptr[i]: prod.indptr[i+1]])[::-1]]
            for i in range(prod.shape[0])])
    # l = [[(p, s) for p, s in zip(prod[i].toarray()[0, jj], fn[jj]) if s.startswith(' ') and ' ' not in s.strip()]
    #     for i, jj in enumerate(idx)]
    l = [[(prob, subst) for prob, subst in zip(prod[i].toarray()[0, jj], fn[jj])] for i, jj in enumerate(idx)]
    print('Combination: ', l[0][:10])
    return l


def bcomb3(df: pd.DataFrame, nmasks: int = 1, s: float = 0) -> pd.DataFrame:
    vec = DictVectorizer(sparse=True)
    f1 = df.substs_probs.apply(lambda r: {subst: prob for prob, subst in r})
    f2 = df.substs_probs_y.apply(lambda r: {subst: prob for prob, subst in r})

    vec.fit(list(f1)+list(f2))
    f1, f2 = (vec.transform(list(f)) for f in (f1, f2))  # shape (num_of_ex, num_of_all_substs_from_both_fs)

    # (1. - ...) / 25... is like probability for 'missing' words; shape (num_of_ex, 1)
    for f in (f1, f2):
        f += (1. - f.sum(axis=-1).reshape(-1, 1)) / 250000**nmasks  # add this prob to each f

    log_prod = np.log(f1) + np.log(f2)  # np.log(f1.multiply(f2))
    log_prior = np.log((f1.mean(axis=0) + f2.mean(axis=0)) / 2)  # axis=0 -> mean for each column (i.e. for each word)
    fpmi = log_prod - s * log_prior

    fn = np.array(vec.feature_names_)
    maxlen = (df.substs_probs_y.apply(len) + df.substs_probs.apply(len)).max()
    idx = fpmi.argsort(axis=-1)[:, :-maxlen-1:-1]
    df['substs_probs'] = [[(p, s) for p, s in zip(fpmi[i, jj], fn[jj]) if s.startswith(' ') and ' ' not in s.strip()]
                          for i, jj in enumerate(idx)]  # there is a hidden filter; should turn off it?
    return df


def comb_mask(paths_name, probs, limit=None, drop_duplicates=True, data_name=None, filter_cnt_subword=False) -> pd.DataFrame:
    if filter_cnt_subword:
        from substwsi.sgen_xlm_opt import load_model  # from xlm.sgen_xlm import load_model
        def cnt_subword(w):
            return len(model_xlm.bpe.encode(f'и {w}').split()) - 1
        def filter(w, cnt_mask):
            return cnt_subword(w) != cnt_subword
        model_xlm = load_model()
    else:
        def filter(w, cnt_mask): return False

    cnt_mask = paths_name[0].count('<mask>')
    print(f'Weight: {probs[cnt_mask]}, {paths_name[0]}')
    dfinp = load_substs(paths_name[0], limit, drop_duplicates, data_name)
    # отсев подстановок с другим количеством subword и домножение вероятности на вероятностью маски
    dfinp['substs_probs'] = dfinp.substs_probs.apply(lambda sp: [(prob * probs[cnt_mask], s) for prob, s in sp if not filter(s, cnt_mask) ])

    for path in paths_name[1:]:
        cnt_mask = path.count('<mask>')
        print(f'Weight: {probs[cnt_mask]}, {path}')
        dfinp_new = load_substs(path, limit, drop_duplicates, data_name)
        # отсев подстановок с другим количеством subword и домножение вероятности на вероятностью маски
        dfinp_new['substs_probs'] = dfinp_new.substs_probs.apply(lambda sp: [(prob * probs[cnt_mask], s) for prob, s in sp if not filter(s, cnt_mask)])

        dfinp = dfinp.merge(dfinp_new, on=['context','positions'], how='inner', suffixes=('','_y'))
        dfinp['substs_probs'] = dfinp.apply(lambda r: sorted(r.substs_probs+r.substs_probs_y, key=lambda pair: pair[0], reverse=True), axis=1)
        #delete unnecessary column
        dfinp = dfinp[['context_id', 'word', 'gold_sense_id', 'predict_sense_id', 'positions', 'context', 'word_at', 'substs_probs']]

    if filter_cnt_subword:
        slens = dfinp.substs_probs.apply(lambda ps: [cnt_subword(s) for p,s in ps])
        for l in range(1,10):
            print(f'{l} subwords', ' '.join( f'top {top}: {slens.apply(lambda lens: (np.array(lens[:top])==l).mean()).mean()}' for top in (10, 30, 100) ))

    return dfinp


def load_substs(substs_fname: str, limit=None, drop_duplicates=False, data_name=None) -> pd.DataFrame:

    if substs_fname.endswith('&'):  # something like this 'file_1&file_2&file_3&...&'
        split: tp.List = substs_fname.strip('&').split('&')

        print(f'Combining:', split)
        dfinps: tp.List[pd.DataFrame] = [load_substs_(p, limit, drop_duplicates, data_name) for p in split]

        res: pd.DataFrame = dfinps[0]
        nm = len(split[0].split('<mask>')) - 1  # num of masks

        for dfinp in dfinps[1:]:
            res: pd.DataFrame = res.merge(dfinp, on=['context', 'positions'], how='inner', suffixes=('', '_y'))
            res.substs_probs = intersect_sparse(res.substs_probs, res.substs_probs_y, nmasks=nm, s=0.0)
            res.drop(columns=[c for c in res.columns if c.endswith('_y')], inplace=True)
        return res

    elif substs_fname.endswith('+'):  # something like this 'file_1+0.0+' or 'file_1+0.5+'
        split = substs_fname.strip('+').split('+')

        s = float(split[-1])
        p1 = '+'.join(split[:-1])
        p2 = re.sub(r'((<mask>)+)(.*?)\bT\b', r'T\3\1', p1)  # try to form symmetrical name

        if p2 == p1:
            p2 = re.sub(r'\bT\b(.*?)((<mask>)+)', r'\2\1T', p1)  # another try
        print(f'Combining {p1} and {p2}')
        if p1 == p2:
            raise Exception('Cannot convert fname to symmetric one:', p1)

        nm = len(substs_fname.split('<mask>'))-1  # num_of masks
        dfinp1, dfinp2 = (load_substs_(p, limit, drop_duplicates, data_name) for p in (p1, p2))

        dfinp = dfinp1.merge(dfinp2, on=['context', 'positions'], how='inner', suffixes=('', '_y'))
        dfinp.substs_probs = intersect_sparse(dfinp.substs_probs, dfinp.substs_probs_y, nmasks=nm, s=s)
        dfinp.drop(columns=[c for c in dfinp.columns if c.endswith('_y')], inplace=True)
        return dfinp

    # something like this T-<mask>-2ltr...@1-3:0.3:0.4:0.3@
    # masks 1-3 with probs 0.3, 0.4, 0.3 respectively
    elif substs_fname.endswith('@'):
        split = substs_fname[:-1].split('@')
        path_name = split[0]
        idx_mask = path_name.find('<mask>')
        assert idx_mask != -1, f"load_substs: not found '<mask>' in path_name = {path_name}"
        path_name = path_name.replace('<mask>', '')
        path_name = path_name[:idx_mask] + '<mask>' + path_name[idx_mask:]
        assert path_name.count('<mask>') == 1, f"load_substs: directory path contain '<mask>', path_name = {path_name}"

        config = split[1].split(':')
        cnt_mask = config[0].split('-')
        cnt_mask = [cnt for cnt in range(int(cnt_mask[0]), int(cnt_mask[1])+1)]
        assert len(cnt_mask) == len(config[1:]), f'load_substs: wrong amount of mask or probs. ' \
                                                 f'masks = {config[0]}, probs = {config[1]}'
        assert 0 <= int(cnt_mask[0]) < 10, f"load_substs: don't support cnt_mask[0] = {cnt_mask[0]} > 9, " \
                                           f"create paths_name will not work"

        probs = {cnt: float(prob) for cnt, prob in zip(cnt_mask, config[1:])}
        assert 1.0001 >= sum(probs.values()) >= 0.9999, f'load_substs: wrong probs = {probs}, sum = {sum(probs.values())}'
        idx_f = path_name.find('ltr') + 3
        paths_name = [path_name[:idx_f].replace('<mask>', '<mask>' * cnt) + str(cnt) + path_name[idx_f+1:] for cnt in cnt_mask]

        print(f'Combining:')
        for p in paths_name:
            print(p)
        return comb_mask(paths_name, probs, limit, drop_duplicates, data_name)

    else:
        return load_substs_(substs_fname, limit, drop_duplicates, data_name)


def load_substs_(substs_fname, limit=None, drop_duplicates=True, data_name=None,
                 filter_subwords=True, filter_mwes=True) -> pd.DataFrame:

    st = time()
    p = Path(substs_fname)
    npz_filename_to_save = None
    print(time()-st, 'Loading substs from ', p)
    if substs_fname.endswith('.npz'):
        arr_dict = np.load(substs_fname, allow_pickle=True)
        ss, pp = arr_dict['substs'], arr_dict['probs']  # ss and pp: np.ndarray
        print(ss.shape, ss.dtype, pp.shape, pp.dtype)
        ss, pp = [list(s) for s in ss], [list(p) for p in pp]  # ss and pp: tp.List[tp.List[str]]
        substs_probs = pd.DataFrame({'substs': ss, 'probs': pp})
        substs_probs = substs_probs.apply(lambda r: [(prob, subst) for subst, prob in zip(r.substs, r.probs)], axis=1)
        # print(substs_probs.head(3))
    else:
        substs_probs = pd.read_csv(p, index_col=0, nrows=limit)['0']

        print(time()-st, 'Eval... ', p)
        substs_probs = substs_probs.apply(pd.eval)
        print(time()-st, 'Reindexing... ', p)
        substs_probs.reset_index(inplace = True, drop = True)

        szip = substs_probs.apply(lambda l: zip(*l)).apply(list)
        res_probs, res_substs = szip.str[0].apply(list), szip.str[1].apply(list)
        # print(type(res_probs))

        npz_filename_to_save = p.parent/(p.name.replace('.bz2', '.npz'))
        if not os.path.isfile(npz_filename_to_save):
            print('saving npz to %s' % npz_filename_to_save)
            np.savez_compressed(p.parent/(p.name.replace('.bz2', '.npz')), probs=res_probs, substs=res_substs)

        # pd.DataFrame({'probs':res_probs, 'substs':res_substs}).to_csv(p.parent/(p.name.replace('.bz2', '.npz')),sep='\t')

    p_ex = p.parent / (p.name+'.input')
    if os.path.isfile(p_ex):
        print(time()-st, 'Loading examples from ', p_ex)
        dfinp = pd.read_csv(p_ex, nrows=limit)
        dfinp['positions'] = dfinp['positions'].apply(pd.eval).apply(tuple)
        dfinp['word_at'] = dfinp.apply(lambda r: r.context[slice(*r.positions)], axis=1)
    else:
        assert data_name is not None, "no input file %s and no data name provided" % p_ex

        from xlm.data_loading import load_data
        dfinp, _ = load_data(data_name)
        if npz_filename_to_save is not None:
            input_filename = npz_filename_to_save.parent / (npz_filename_to_save.name+'.input')
        else:
            input_filename = p_ex
        print('saving input to %s' % input_filename)
        dfinp.to_csv(input_filename, index=False)

    dfinp.positions = dfinp.positions.apply(tuple)
    dfinp['substs_probs'] = substs_probs
    if drop_duplicates:
        dfinp = dfinp.drop_duplicates('context')
    dfinp.reset_index(inplace = True)
#    print(dfinp.head())
    dfinp['positions'] = dfinp.positions.apply(tuple)
    # dfinp.substs_probs = dfinp.substs_probs.apply(lambda l: [(p,subst) for p,subst in l
    #                                                  if (not filter_subwords or subst.startswith(' '))
    #                                                      and (not filter_mwes or not ' ' in subst.strip())])
    #                                                    #and (not filter_mwes or not re.findall(r'\W',subst.strip()))])
    # print(dfinp[dfinp.substs_probs.apply(lambda x: len(x) == 0)])
    return dfinp


def sstat(substs_fname, limit=None, dataset_path=None, dfinp=None) -> None:
    if dfinp is None:
        dfinp = load_substs(substs_fname, limit)
        if 'gold_sense_id' not in dfinp.columns and dataset_path is not None:
            dataset_df = pd.read_csv(dataset_path, sep='\t', quoting=csv.QUOTE_MINIMAL)
            dataset_df['positions'] = dataset_df['positions'].apply(lambda p: tuple(map(int, p.split('-'))))
            dfinp = dfinp.merge(dataset_df, how='inner', on=['word', 'context', 'positions'])
    print('Counting stat')
    print(dfinp.head(3))
    dfinp['word_at'] = dfinp.apply(lambda r: r.context[r.positions[0]: r.positions[1]], axis=1)
    topks = (10, 30, 100)
    for topk in topks:
        dfinp[f'substs_probs_{topk}'] = dfinp.substs_probs.str[:topk]
    key = ['word', 'gold_sense_id'] if 'gold_sense_id' in dfinp and dfinp['gold_sense_id'].isnull().sum() == 0 else 'word'
    rdf = dfinp.groupby(key).agg({
                'word_at': lambda x: Counter(x).most_common(),
                **{f'substs_probs_{topk}': lambda x: Counter(s for l in x for s in set(s1 for p, s1 in l)).most_common() for topk in topks}
    }).reset_index()
    rdf['cnt'] = rdf.word_at.apply(lambda l: sum(c for w, c in l))
    topk2subststat = {topk:  pd.DataFrame.from_records(((r.word, s, cnt) for _, r in rdf.iterrows() for s, cnt in r[f'substs_probs_{topk}']), columns=['word','subst','cnt']) for topk in topks}
    for topk in topks:
        sdf = topk2subststat[topk]
        wordsubst2cnt = sdf.groupby(['word', 'subst']).agg({'cnt': 'sum'}).cnt.to_dict()
        subst2cnt = sdf.groupby('subst').agg({'cnt': 'sum'}).cnt.to_dict()
        word2cnt = rdf.groupby('word').agg({'cnt': 'sum'}).cnt.to_dict()
#        import pdb; pdb.set_trace()
        rdf[f'substs_probs_{topk}'] = rdf.apply(lambda r: [(s, c, c/r.cnt, (wordsubst2cnt[(r.word, s)]-c)/(word2cnt[r.word]-r.cnt) if word2cnt[r.word]-r.cnt != 0 else np.nan,
                                                           (subst2cnt[s]-wordsubst2cnt[(r.word, s)])/(len(dfinp)-word2cnt[r.word]) if len(dfinp)-word2cnt[r.word] != 0 else np.nan)
                                                          for s, c in r[f'substs_probs_{topk}']], axis=1)

    stats_fname = substs_fname + '.sstat.tsv'
    with open(stats_fname, 'w') as outp:
        for i, r in rdf.iterrows():
            print('->', r.word, f'_{r.gold_sense_id}' if "gold_sense_id" in r else "", r.cnt, ','.join('%s %d' % p for p in r.word_at), file=outp)
            for topk in topks:
                print(f'in-topk-{topk}->', ','.join('%s %d-%.2f/%.2f/%.2f' % t for t in r[f'substs_probs_{topk}']), file=outp)
    print('Stats saved to ', stats_fname)


if __name__ == '__main__':
    Fire(sstat)
