from contextlib import contextmanager

import fire
import numpy as np
import os
import pandas as pd
import torch
from pathlib import Path
from time import time
import typing as tp

from transformers import BertTokenizer
from pytorch_pretrained_bert import BertForMaskedLM


@contextmanager
def model_eval(model):
    '''Temporarily switch to evaluation mode.'''
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


def load_model_and_tokenizer(model_path_or_name='bert-large-uncased'):
    p = Path(model_path_or_name)
    if '/' in str(p) or p.is_file():
        print('Loading BERT from', model_path_or_name)
    else:
        print("Loading BERT from HUB: model = ", model_path_or_name)

    model = BertForMaskedLM.from_pretrained(model_path_or_name)
    tokenizer = BertTokenizer.from_pretrained(model_path_or_name)

    model.cls.predictions = model.cls.predictions.transform

    model = model.cuda() if torch.cuda.is_available() else model
    print('Model is running on device:', next(model.parameters()).device)
    model.eval()

    return model, tokenizer


def prepare_inputs(
        tokenizer,
        context, positions,
        template, fill_masks,
        debug, maxlen=510
):
    template = template.replace('_', ' ')
    pre, target, post = context[:positions[0]], context[slice(*positions)], context[positions[1]:]
    assert '<mask>' in template or 'T' in template, ''
    if '<mask>' in template:
        ctx_with_template = pre + template.replace('T', target).replace('<mask>', tokenizer.mask_token) + post
        # recalculate positions for mask
        mask_idx = ctx_with_template.index(tokenizer.mask_token)
        new_positions = mask_idx, mask_idx + len(tokenizer.mask_token)
    elif 'T' in template:
        new_st = len(pre) + template.index('T')
        new_positions = new_st, new_st + len(target)
        ctx_with_template = pre + template.replace('T', target) + post
    else:
        raise RuntimeError("Need to put mask or target(T) in template")

    pre = ctx_with_template[:new_positions[0]]
    target = ctx_with_template[slice(*new_positions)]
    post = ctx_with_template[new_positions[1]:]

    before_pred = ['[CLS]'] + tokenizer.tokenize(pre)
    after_pred = tokenizer.tokenize(post) + ['[SEP]']
    cnt_tokens_pred = len(before_pred)

    if '{mask}' in template:
        # {mask_predict}
        target_prediction_idx = torch.arange(cnt_tokens_pred, cnt_tokens_pred + fill_masks)
        target_tokens = [tokenizer.mask_token for _ in range(fill_masks)]
    else:
        # {target_predict}
        assert fill_masks == 1, "Set fill masks == 1 for non masked input"
        target_prediction_idx = torch.tensor([cnt_tokens_pred])
        target_tokens = tokenizer.tokenize(target)

    return torch.tensor(
        tokenizer.convert_tokens_to_ids(before_pred + target_tokens + after_pred)).view(1, -1), target_prediction_idx


def _fill_one_mask_batch3(model, tokens, masked_index, topk, opt=0):
    assert tokens.dim() == 2 and masked_index.dim() == 0
    if opt == 0:
        features = model(tokens.long())
        logits = features[:, masked_index, :]
        print(logits.shape)
        logits = torch.matmul(logits, model.bert.embeddings.word_embeddings.weight.transpose(0, 1))
    elif opt == 1:
        # TODO: how to return only last hidden state output from BERT ???
        raise NotImplementedError("Check for sgen_xlm_opt ")
    else:
        raise ValueError('Unknown optimization level: ', opt)
    prob = logits.softmax(dim=-1)
    values, indexes = prob.topk(k=topk, dim=-1)
    return indexes.detach(), values.detach()


def fill_mask_mwe_ltr_batch_beamsearch3(
        model, tokenizer,
        context: str, positions: tp.Tuple[int, int],
        template: str, fill_masks: int,
        topk: int, beam_size: int, cont_greedy: bool,
        fix_spaces=False, bpe_tokenize=True,
        batch_size_tokens=4000, maxlen=510,
        debug=True
):
    tokens, masked_index = prepare_inputs(tokenizer, context, positions, template, fill_masks, debug, maxlen)
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    masked_index = masked_index.to(device)
    assert tokens.dim() == 2 and masked_index.dim() == 1, f'{tokens.shape}, {masked_index.shape}'
    cont_topk = 1 if cont_greedy else beam_size
    if debug:
        print('Filling mask 0...')

    indexes, values = _fill_one_mask_batch3(model, tokens, masked_index[0], beam_size)
    indexes, values = indexes.T, values.T
    assert indexes.shape == (beam_size, 1) and values.shape == (beam_size, 1)
    if debug:
        print("Warning: print_hypos3 not implemented")
        # print_hypos3(self, indexes, values)

    batch_size = batch_size_tokens // tokens.size()[-1]
    tokens = tokens.repeat(beam_size, 1)
    if debug:
        print('Dynamic batch size:', batch_size, 'Tokens.shape / masked_index.shape:', tokens.shape, masked_index.shape)

    for mask_num in range(1, len(masked_index) if fill_masks is None else fill_masks):
        tokens[:, masked_index[:mask_num]] = indexes.type(torch.long)
        if debug: print(f'Filling mask {mask_num}/{len(masked_index)}')
        indexes2, values2 = (torch.cat(m, dim=0) for m in zip(*(
            _fill_one_mask_batch3(model, tokens[st:st + batch_size], masked_index[mask_num], topk=cont_topk)
            for st in range(0, len(tokens), batch_size))))
        if debug:
            pass
            # print('Hypos shape / continuations shape:', indexes.shape, indexes2.shape)
            # for i in range(len(indexes)):
            #     print(self.task.source_dictionary.string(indexes[i, :]), '->',
            #           self.task.source_dictionary.string(indexes2[i][:10]),
            #           np.array2string(values2[i][:10].cpu().detach().numpy(), formatter=fff))

        assert values.shape == (beam_size, mask_num) and values2.shape == (beam_size, cont_topk), \
            f'Incorrect shapes: {values.shape} and {values2.shape}'
        # multiply probs of each previous hypothesis and beam_size possible continuations:
        # P(hypo_i)*P(cont_ij|hypo_i)
        logprobs = (torch.log(values).sum(axis=-1, keepdims=True) + torch.log(values2)).reshape(-1)
        ids = logprobs.argsort().flip(dims=(0,))[:beam_size]  # take beam_size most probably hypothesis
        ii, jj = ids // cont_topk, ids % cont_topk
        indexes = torch.cat([indexes[ii], indexes2[ii, jj].reshape(-1, 1)], dim=-1)
        values = torch.cat([values[ii], values2[ii, jj].reshape(-1, 1)], dim=-1)
        if debug:
            print("Warning: print_hypos3 not implemented")
            # print_hypos3(self, indexes, values)

    res_substs = [tokenizer.decode(indexes[i]) for i in range(indexes.shape[0])]
    res_probs = np.exp(np.log(values.cpu().detach().numpy()).sum(axis=-1))[:topk]
    if debug:
        print('-->', ' '.join(f"'{s}' {p:.3f}" for p, s in zip(res_probs[:15], res_substs[:15])))

    return res_substs, res_probs


def process_target(target):
    # required for Semeval 2020 Task 1 English corpus where targets are appended with POS tags
    return target.split('_')[0]


def generate_substitutes(
        data_name=None, topk=500, templ='<mask>', fill_masks=2, fix_spaces=True, debug=True,
        skip_rows=None, limit=None, model_path=None, max_ex_per_word=None, maxlen=510,
        drop_duplicates=True, beam_search=True, rewrite_existing=True, cont_greedy=True,
        version=3, skip_last_nmasks=0
):
    if beam_search is False and cont_greedy is False:
        raise ValueError(f'Incompatible combination: beam_search={beam_search}, cont_greedy={cont_greedy}')
    model_path = Path(model_path) if model_path is not None else None
    model, tokenizer = load_model_and_tokenizer(str(model_path))

    if isinstance(data_name, list):
        dirs, fname, df = data_name
        data_name = None
    else:
        raise NotImplementedError("Check sgen_xlm_opt")

    if templ == '<mask>T':
        templ = 'T'

    if not rewrite_existing and os.path.isfile(fname):
        print('templates already exist in the location %s' % fname)
        return fname

    print(
        f'Dataset {data_name}: topk={topk}, templ="{templ}", fill_masks={fill_masks} '
        f'with limit {limit}, max_ex_per_word {max_ex_per_word}, maxlen {maxlen}, '
        f'beam_search {beam_search}, contgreedy {cont_greedy}, version {version}'
    )
    print('Saving results to:\n', fname)
    if max_ex_per_word is not None:
        df = df.sample(frac=1).groupby('word').head(max_ex_per_word)
    df.to_csv(dirs / (fname.name + '.input'), index=False)
    wordat2cnt = df.apply(lambda r: r.context[r.positions[0]:r.positions[1]].strip(), axis=1).value_counts().to_dict()
    print('Targets and counts: ', wordat2cnt)
    print('Target counts: ', pd.Series(list(wordat2cnt.values())).describe())
    print('Target lengths: ', pd.Series(list(wordat2cnt.keys())).apply(len).describe())
    assert all(len(w) > 0 for w in wordat2cnt.keys()), 'some positions are incorrect!'
    print('Context lengths in words: ', df.context.str.split(' ').apply(len).describe())
    print('Context lengths in lines: ', df.context.str.split('\n').apply(len).describe())
    df['lctx'], df['rctx'] = df.apply(lambda r: r.context[:r.positions[0]], axis=1), \
                             df.apply(lambda r: r.context[r.positions[1]:], axis=1),
    df['target_word'] = df.apply(lambda r: r.context[r.positions[0]:r.positions[1]], axis=1)
    for sample in df.head(3), df.tail(3):
        print('Left contexts:\n', '\n'.join(sample.lctx))
        print('Right contexts:\n', '\n'.join(sample.rctx))
        print('Target words:\n', '\n'.join(sample.target_word))

    if version in [1, 2]:
        raise NotImplementedError("check sgen_xlm_opt")
    elif version == 3:
        beam_size = topk * 5 if beam_search else topk
        with model_eval(model):
            with torch.no_grad():
                res_ser = df.progress_apply(
                    lambda row: fill_mask_mwe_ltr_batch_beamsearch3(
                        model=model, tokenizer=tokenizer,
                        context=row.context, positions=row.positions,
                        template=templ,
                        topk=topk, beam_size=beam_size,
                        cont_greedy=cont_greedy,
                        fill_masks=fill_masks - skip_last_nmasks,
                        fix_spaces=fix_spaces, debug=debug, maxlen=maxlen),
                    axis=1
                )
    else:
        raise ValueError('Version %d is currently not implemented!' % version)

    res_substs, res_probs = (np.array(list(res_ser.str[i])) for i in (0, 1))
    np.savez_compressed(fname, substs=res_substs, probs=res_probs)
    print(f'Substs {res_substs.shape} and probs {res_probs.shape} saved to {fname}\n'.encode('utf-8', errors='ignore'))
    print(f'Substs {res_substs[:3]} and probs {res_probs[:3]}'.encode('utf-8', errors='ignore'))

    return fname


if __name__ == '__main__':
    fire.Fire(generate_substitutes)
