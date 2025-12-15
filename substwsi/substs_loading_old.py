from pathlib import Path
import pandas as pd
import numpy as np
from time import time
from collections import Counter
import os
import regex as re
from sklearn.feature_extraction import DictVectorizer


def intersect_sparse(substs_probs, substs_probs_y, nmasks=1, s=0):
    print('intersect_sparse, arg1:', substs_probs.iloc[0][:10], 'intersect_sparse, arg2:', substs_probs_y.iloc[0][:10],
          sep='\n')
    vec = DictVectorizer(sparse=True)
    f1 = substs_probs.apply(lambda l: {s: p for p, s in l})
    f2 = substs_probs_y.apply(lambda l: {s: p for p, s in l})
    vec.fit(list(f1) + list(f2))
    f1, f2 = (vec.transform(list(f)) for f in (f1, f2))
    alpha1, alpha2 = ((1. - f.sum(axis=-1).reshape(-1, 1)) / 250000 ** min(nmasks, 3) for f in (f1, f2))
    prod = f1.multiply(f2) + f1.multiply(alpha2) + f2.multiply(
        alpha1)  # + alpha1*alpha2 is ignored to preserve sparsity; finally, we don't want substs with 0 probs before smoothing in both distribs

    fn = np.array(vec.feature_names_)
    maxlen = (substs_probs_y.apply(len) + substs_probs.apply(len)).max()
    # print(prod.shape, type(prod))
    #    idx = ( row.toarray()[0].argsort(axis=-1)[:-maxlen-1:-1] for row in prod )
    m = prod
    idx = ([m.indices[m.indptr[i]:m.indptr[i + 1]][np.argsort(m.data[m.indptr[i]:m.indptr[i + 1]])[::-1]] for i in
            range(m.shape[0])])
    l = [[(p, s) for p, s in zip(prod[i].toarray()[0, jj], fn[jj]) if s.startswith(' ') and ' ' not in s.strip()] for
         i, jj in enumerate(idx)]
    print('Combination: ', l[0][:10])
    return l


def bcomb3(df, nmasks=1, s=0):
    vec = DictVectorizer(sparse=True)
    f1 = df.substs_probs.apply(lambda l: {s: p for p, s in l})
    f2 = df.substs_probs_y.apply(lambda l: {s: p for p, s in l})
    vec.fit(list(f1) + list(f2))
    f1, f2 = (vec.transform(list(f)) for f in (f1, f2))
    for f in (f1, f2):
        f += (1. - f.sum(axis=-1).reshape(-1, 1)) / 250000 ** nmasks
    log_prod = np.log(f1) + np.log(f2)
    log_prior = np.log((f1.mean(axis=0) + f2.mean(axis=0)) / 2)
    fpmi = log_prod - s * log_prior

    fn = np.array(vec.feature_names_)
    maxlen = (df.substs_probs_y.apply(len) + df.substs_probs.apply(len)).max()
    idx = fpmi.argsort(axis=-1)[:, :-maxlen - 1:-1]
    l = [[(p, s) for p, s in zip(fpmi[i, jj], fn[jj]) if s.startswith(' ') and ' ' not in s.strip()] for i, jj in
         enumerate(idx)]
    df['substs_probs'] = l
    return df


def comb_mask(paths_name, probs, limit=None, drop_duplicates=True, data_name=None, filter_cnt_subword=False):
    if filter_cnt_subword:
        from xlm.sgen_xlm import load_model
        def cnt_subword(w):
            return len(model_xlm.bpe.encode(f'и {w}').split()) - 1

        def filter(w, cnt_mask):
            return cnt_subword(w) != cnt_subword

        model_xlm = load_model()
    else:
        def filter(w, cnt_mask):
            return False

    cnt_mask = paths_name[0].count('_mask_')
    print(f'Weight: {probs[cnt_mask]}, {paths_name[0]}')
    dfinp = load_substs(paths_name[0], limit, drop_duplicates, data_name)
    # отсев подстановок с другим количеством subword и домножение вероятности на вероятностью маски
    dfinp['substs_probs'] = dfinp.substs_probs.apply(
        lambda sp: [(prob * probs[cnt_mask], s) for prob, s in sp if not filter(s, cnt_mask)])

    for path in paths_name[1:]:
        cnt_mask = path.count('_mask_')
        print(f'Weight: {probs[cnt_mask]}, {path}')
        dfinp_new = load_substs(path, limit, drop_duplicates, data_name)
        # отсев подстановок с другим количеством subword и домножение вероятности на вероятностью маски
        dfinp_new['substs_probs'] = dfinp_new.substs_probs.apply(
            lambda sp: [(prob * probs[cnt_mask], s) for prob, s in sp if not filter(s, cnt_mask)])

        dfinp = dfinp.merge(dfinp_new, on=['context', 'positions'], how='inner', suffixes=('', '_y'))
        dfinp['substs_probs'] = dfinp.apply(
            lambda r: sorted(r.substs_probs + r.substs_probs_y, key=lambda pair: pair[0], reverse=True), axis=1)
        # delete unnecessary column
        dfinp = dfinp[['word', 'positions', 'context', 'word_at', 'substs_probs', 'gold_sense_id']]

    if filter_cnt_subword:
        slens = dfinp.substs_probs.apply(lambda ps: [cnt_subword(s) for p, s in ps])
        for l in range(1, 10):
            print(f'{l} subwords', ' '.join(
                f'top {top}: {slens.apply(lambda lens: (np.array(lens[:top]) == l).mean()).mean()}' for top in
                (10, 30, 100)))

    return dfinp


def load_substs(substs_fname, limit=None, drop_duplicates=False, data_name=None):
    if substs_fname.endswith('&'):
        split = substs_fname.strip('&').split('&')
        print(f'Combining:', split)
        dfinps = [load_substs_(p, limit, drop_duplicates, data_name) for p in split]
        res = dfinps[0]
        nm = len(split[0].split('_mask_')) - 1
        for dfinp in dfinps[1:]:
            res = res.merge(dfinp, on=['context', 'positions'], how='inner', suffixes=('', '_y'))
            res.substs_probs = intersect_sparse(res.substs_probs, res.substs_probs_y, nmasks=nm, s=0.0)
            res.drop(columns=[c for c in res.columns if c.endswith('_y')], inplace=True)
        return res
    elif substs_fname.endswith('+'):
        split = substs_fname.strip('+').split('+')
        p1 = '+'.join(split[:-1])
        p2 = re.sub(r'((_mask_)+)(.*?)\bT\b', r'T\3\1', p1)  # try to form symmetrical name

        if p2 == p1:
            p2 = re.sub(r'\bT\b(.*?)((_mask_)+)', r'\2\1T', p1)  # another try
        print(f'Combining {p1} and {p2}')
        if p1 == p2:
            raise Exception('Cannot conver fname to symmetric one:', p1)
        dfinp1, dfinp2 = (load_substs_(p, limit, drop_duplicates, data_name) for p in (p1, p2))
        dfinp = dfinp1.merge(dfinp2, on=['context', 'positions'], how='inner', suffixes=('', '_y'))
        dfinp.substs_probs = intersect_sparse(dfinp.substs_probs, dfinp.substs_probs_y,
                                              nmasks=len(substs_fname.split('_mask_')) - 1)
        dfinp.drop(columns=[c for c in dfinp.columns if c.endswith('_y')], inplace=True)
        return dfinp
    elif substs_fname.endswith('@'):  # ./T-_mask_-2ltr...@1-3:0.3:0.4:0.3@ #маски 1-3 с вер-тями 0.3, 0.4, 0.3 соотв.
        split = substs_fname[:-1].split('@')
        path_name = split[0]
        idx_mask = path_name.find('_mask_')
        assert idx_mask != -1, f"load_substs: not found '_mask_' in path_name = {path_name}"
        path_name = path_name.replace('_mask_', '')
        path_name = path_name[:idx_mask] + '_mask_' + path_name[idx_mask:]
        assert path_name.count('_mask_') == 1, f"load_substs: directory path contain '_mask_', path_name = {path_name}"

        config = split[1].split('=')
        cnt_mask = config[0].split('-')
        cnt_mask = [cnt for cnt in range(int(cnt_mask[0]), int(cnt_mask[1]) + 1)]
        assert len(cnt_mask) == len(
            config[1:]), f'load_substs: wrong amount of mask or probs. masks = {config[0]}, probs = {config[1]}'
        assert 0 <= int(cnt_mask[
                            0]) < 10, f"load_substs: don't support cnt_mask[0] = {cnt_mask[0]} > 9, create paths_name will not work"

        probs = {cnt: float(prob) for cnt, prob in zip(cnt_mask, config[1:])}
        assert 1.0001 >= sum(
            probs.values()) >= 0.9999, f'load_substs: wrong probs = {probs}, sum = {sum(probs.values())}'
        idx_f = path_name.find('ltr') + 3
        paths_name = [path_name[:idx_f].replace('_mask_', '_mask_' * cnt) + str(cnt) + path_name[idx_f + 1:] for cnt in
                      cnt_mask]
        print(f'Combining:')
        for p in paths_name:
            print(p)
        res = comb_mask(paths_name, probs, limit, drop_duplicates, data_name)
        return res
    else:
        return load_substs_(substs_fname, limit, drop_duplicates, data_name)


def load_substs_(substs_fname, limit=None, drop_duplicates=True, data_name=None):
    st = time()
    p = Path(substs_fname)
    npz_filename_to_save = None
    print(time() - st, 'Loading substs from ', p)
    if substs_fname.endswith('.npz'):
        arr_dict = np.load(substs_fname, allow_pickle=True)
        ss, pp = arr_dict['substs'], arr_dict['probs']
        print(ss.shape, ss.dtype, pp.shape, pp.dtype)
        ss, pp = [list(s) for s in ss], [list(p) for p in pp]
        substs_probs = pd.DataFrame({'substs': ss, 'probs': pp})
        substs_probs = substs_probs.apply(lambda r: [(p, s) for s, p in zip(r.substs, r.probs)], axis=1)
        # print(substs_probs.head(3))
    else:
        substs_probs = pd.read_csv(p, index_col=0, nrows=limit)['0']

        print(time() - st, 'Eval... ', p)
        substs_probs = substs_probs.apply(pd.eval)
        print(time() - st, 'Reindexing... ', p)
        substs_probs.reset_index(inplace=True, drop=True)

        szip = substs_probs.apply(lambda l: zip(*l)).apply(list)
        res_probs, res_substs = szip.str[0].apply(list), szip.str[1].apply(list)
        # print(type(res_probs))

        npz_filename_to_save = p.parent / (p.name.replace('.bz2', '.npz'))
        if not os.path.isfile(npz_filename_to_save):
            print('saving npz to %s' % npz_filename_to_save)
            np.savez_compressed(p.parent / (p.name.replace('.bz2', '.npz')), probs=res_probs, substs=res_substs)

        # pd.DataFrame({'probs':res_probs, 'substs':res_substs}).to_csv(p.parent/(p.name.replace('.bz2', '.npz')),sep='\t')

    p_ex = p.parent / (p.name + '.input')
    if os.path.isfile(p_ex):
        print(time() - st, 'Loading examples from ', p_ex)
        dfinp = pd.read_csv(p_ex, nrows=limit)
        dfinp['positions'] = dfinp['positions'].apply(pd.eval).apply(tuple)
        dfinp['word_at'] = dfinp.apply(lambda r: r.context[slice(*r.positions)], axis=1)
    else:
        assert data_name is not None, "no input file %s and no data name provided" % p_ex

        from xlm.data_loading import load_data
        dfinp, _ = load_data(data_name)
        if npz_filename_to_save is not None:
            input_filename = npz_filename_to_save.parent / (npz_filename_to_save.name + '.input')
        else:
            input_filename = p_ex
        print('saving input to %s' % input_filename)
        dfinp.to_csv(input_filename, index=False)

    dfinp.positions = dfinp.positions.apply(tuple)
    dfinp['substs_probs'] = substs_probs
    if drop_duplicates:
        dfinp = dfinp.drop_duplicates('context')
    dfinp.reset_index(inplace=True)
    #    print(dfinp.head())
    dfinp['positions'] = dfinp.positions.apply(tuple)
    return dfinp


def sstat(substs_fname, limit=None):
    dfinp = load_substs(substs_fname, limit)
    print('Counting stat')
    print(dfinp.head(3))
    dfinp['word_at'] = dfinp.apply(lambda r: r.context[r.positions[0]: r.positions[1]], axis=1)
    key = ['word', 'gold_sense_id'] if 'gold_sense_id' in dfinp else 'word'
    rdf = dfinp.groupby(key).agg({
        'word_at': lambda x: Counter(x).most_common(),
        'substs_probs': lambda x: Counter(s for l in x for s in set(s1 for p, s1 in l[:30])).most_common()
    }).reset_index()
    rdf['cnt'] = rdf.word_at.apply(lambda l: sum(c for w, c in l))
    rdf.substs_probs = rdf.apply(lambda r: [(s, c / r.cnt) for s, c in r.substs_probs], axis=1)

    stats_fname = substs_fname + '.sstat.tsv'
    with open(stats_fname, 'w') as outp:
        for i, r in rdf.iterrows():
            print(r.word, f'_{r.gold_sense_id}' if "gold_sense_id" in r else "", r.cnt,
                  ','.join('%s %d' % p for p in r.word_at), file=outp)
            print('->', ','.join('%s %.2f' % p for p in r.substs_probs), file=outp)
    print('Stats saved to ', stats_fname)


from fire import Fire

if __name__ == '__main__':
    Fire(sstat)