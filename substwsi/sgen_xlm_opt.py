import fire
import numpy as np
import os
import pandas as pd
import sys
import traceback
import torch
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from time import time
from tqdm import tqdm
import logging

logging.basicConfig(level = logging.INFO)

tqdm.pandas()
fff = {'float_kind':lambda x: "%.3f" % x}


def load_model(model_path=None):
    if model_path is None:
        logging.info('Loading XLM-R from torch hub ...')
        model = torch.hub.load('pytorch/fairseq', 'xlmr.base')
    else:
        logging.info('Loading XLM-R from' + str(model_path))
        from fairseq.models.roberta import XLMRModel
        p = Path(model_path)
        model = XLMRModel.from_pretrained(str(p.parent), checkpoint_file=p.name)

    model = model.cuda() if torch.cuda.is_available() else model
    logging.info(f'Device XLM-R is running on: {model.device}')
    model.eval()
    return model

#from data_loading import load_data
import sys
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import numpy as np
from collections import Counter
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pymorphy2 import MorphAnalyzer

_ma = MorphAnalyzer()
_ma_cache = {}
def lemmatize(s):
  s = s.strip() # get rid of spaces before and after token, pytmorphy2 doesn't work with them correctly
  if s not in _ma_cache:
    _ma_cache[s] = _ma.parse(s)[0].normal_form
  return _ma_cache[s]
#
# def get_normal_forms(s, nf_cnt=None):
#   hh = ma(s)
#   if nf_cnt is not None and len(hh)>1: # select most common normal form
#     h_weights = [nf_cnt[h.normal_form] for h in hh]
#     max_weight = max(h_weights)
#     return {h.normal_form for i,h in enumerate(hh) if h_weights[i]==max_weight}
#   else:
#     return {h.normal_form for h in hh}


def prepare_inputs(self, masked_input, fix_spaces, bpe_tokenize, debug, maxlen=510):
    if debug: logging.info('--> ', masked_input)
    masked_token = '<mask>'
    assert masked_token in masked_input, \
        "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(masked_token)
    if bpe_tokenize:
        text_spans = masked_input.split(masked_token)
        text_spans_bpe = (' {0} '.format(masked_token)).join(
          [self.bpe.encode(text_span.rstrip()) if not fix_spaces or si==0 else
           ' '.join(self.bpe.encode('&'+text_span.rstrip()).split(' ')[1:])
           for si, text_span in enumerate(text_spans)]
        ).strip()
    else:
        text_spans_bpe = masked_input

    subwords = text_spans_bpe.split(' ')
    if len(subwords) > maxlen:
        center = subwords.index(masked_token)
        if center < maxlen//2:
            text_spans_bpe = ' '.join(subwords[:maxlen])
        else:
            text_spans_bpe = ' '.join(subwords[center-maxlen//2:center+maxlen//2])

    tokens = self.task.source_dictionary.encode_line(
        '<s> ' + text_spans_bpe + ' </s>',
        append_eos=False,
        add_if_not_exist=False,
    )
    # if debug:
    #     logging.info('/'.join(self.task.source_dictionary[i] for i in tokens))

    masked_index, = (tokens == self.task.mask_idx).nonzero(as_tuple=True)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if debug:
        logging.info(f'Input shape: {tokens.size()}, masked_index: {masked_index}')
    return tokens, masked_index


def fill_mask(self, masked_input: str, topk: int = 5, fix_spaces=True, bpe_tokenize=True, debug=True, maxlen=510):
    tokens, masked_index = prepare_inputs(self, masked_input, fix_spaces, bpe_tokenize, debug, maxlen)
    assert tokens.dim() == 2 and masked_index.dim() == 1, f'{tokens.shape}, {masked_index.shape}'
    masked_index = masked_index[0]
    from fairseq import utils

    with utils.model_eval(self.model):
        features, extra = self.model(
            tokens.long().to(device=self.device),
            features_only=False,
            return_all_hiddens=False,
        )
    logits = features[0, masked_index, :].squeeze()
    prob = logits.softmax(dim=0)
    values, index = prob.topk(k=topk, dim=0)
    topk_predicted_token_bpe = self.task.source_dictionary.string(index)

    res_substs, res_probs = [], []
    for index, predicted_token_bpe in enumerate(topk_predicted_token_bpe.split(' ')):
        predicted_token = self.bpe.decode(predicted_token_bpe)
        # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
        if predicted_token_bpe.startswith('\u2581'):
            predicted_token = ' ' + predicted_token
        res_probs.append( values[index].item() )
        res_substs.append( predicted_token )
    if debug:
        logging.info('-->', ' '.join(f"'{s}' {p:.3f}" for p,s in zip(res_probs[:15],res_substs[:15])))
    return res_substs, res_probs


def _fill_one_mask_batch(self, tokens, masked_index, topk):
    assert tokens.dim() == 2 and masked_index.dim() == 0
    # logging.info('_fill_one_mask_batch, tokens:', tokens.shape)
    from fairseq import utils

    with utils.model_eval(self.model):
        features, extra = self.model(
            tokens.long().to(device=self.device),
            features_only=False,
            return_all_hiddens=False,
        )
    # logging.info('_fill_one_mask_batch done, features:', features.shape)
    # logging.info('features:',features.size())
    logits = features[:, masked_index, :]
    # logging.info(logits.size())
    prob = logits.softmax(dim=-1)
    values, indexes = prob.topk(k=topk, dim=-1)
    # logging.info('indexes:',indexes.size(), values.size())
    return indexes.detach().cpu().numpy(), values.detach().cpu().numpy()


def _fill_one_mask_batch3(self, tokens, masked_index, topk, opt=1):
    assert tokens.dim() == 2 and masked_index.dim() == 0
    # logging.info('_fill_one_mask_batch, tokens:', tokens.shape)
#    logging.info(torch.cuda.get_device_properties(0).total_memory/2**30, torch.cuda.memory_reserved(0)/2**30, torch.cuda.memory_allocated(0)/2**30)
        #toks = tokens.long().to(device=self.device)
#    import ipdb; ipdb.set_trace()
    st = time()
    if opt == 0:
        features, extra = self.model(tokens.long(), features_only=False, return_all_hiddens=False)
#        logging.info(1, tokens.shape, time()-st)
        logits = features[:, masked_index, :]
#        logging.info(2, tokens.shape, time()-st)
    elif opt == 1:
        features, extra = self.model(tokens.long(), features_only=True, return_all_hiddens=False)
#        logging.info(1, tokens.shape, time()-st)
        features_at_mask = features[:, masked_index, :]
#        logging.info(11, tokens.shape, time()-st)
        logits = self.model.encoder.lm_head(features_at_mask)
#        logging.info(2, tokens.shape, time()-st)
    else:
        raise ValueError('Unknown optimization level: ', opt)
#    logging.info(3, tokens.shape, time()-st)
    # logging.info('_fill_one_mask_batch done, features:', features.shape)
    # logging.info('features:',features.size())
   # logging.info(logits.size())
    prob = logits.softmax(dim=-1)
#    logging.info(4, tokens.shape, time()-st)
    values, indexes = prob.topk(k=topk, dim=-1)
#    logging.info(5, tokens.shape, time()-st)
    # logging.info('indexes:',indexes.size(), values.size())
    return indexes.detach(), values.detach()


def fill_mask_mwe_ltr_batch(self, masked_input: str, topk: int = 5, fill_masks=2, fix_spaces=True,
                            bpe_tokenize=True, debug=True, batch_size_tokens=4000, maxlen=510):
    """
    Buggy version 1: actually fills only (I[fill_masks > 1] + 1) first masks
    """
    tokens, masked_index = prepare_inputs(self, masked_input, fix_spaces, bpe_tokenize, debug, maxlen)
    assert tokens.dim() == 2 and masked_index.dim() == 1, f'{tokens.shape}, {masked_index.shape}'

    indexes, values = _fill_one_mask_batch(self, tokens, masked_index[0], topk)
    indexes, values = indexes[0], values[0]
    # logging.info(indexes)
    topk_predicted_token_bpe = self.task.source_dictionary.string(indexes)
    if debug:
        logging.info(topk_predicted_token_bpe)

    res_substs, res_probs = [], []
    batch_size = batch_size_tokens // tokens.size()[-1]
    if debug:
        logging.info('Dynamic batch size: ', batch_size)
    if fill_masks > 1:
        tokens = tokens.repeat(topk, 1)
        tokens[:, masked_index[0]] = torch.tensor(indexes)
        indexes2, values2 = (np.vstack(m) for m in zip(*(
          _fill_one_mask_batch(self, tokens[st:st+batch_size], masked_index[1], topk=3)
          for st in range(0, len(tokens), batch_size))))

    for (i, predicted_token_bpe), index in zip(enumerate(topk_predicted_token_bpe.split(' ')), indexes):
        if fill_masks > 1:
            topk_predicted_token_bpe2 = self.task.source_dictionary.string(indexes2[i])
            if debug and i < 10:
                logging.info(predicted_token_bpe, '->', topk_predicted_token_bpe2, values[i],'*',
                      np.array2string(values2[i], formatter=fff),'=',np.array2string(values[i]*values2[i], formatter=fff))

            topk_predicted_token_bpe2 = self.task.source_dictionary.string(indexes2[i, :1])
            predicted_token = self.bpe.decode(' '.join([predicted_token_bpe, topk_predicted_token_bpe2]))
        else:
            predicted_token = self.bpe.decode(predicted_token_bpe)

        # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
        if predicted_token_bpe.startswith('\u2581'):
            predicted_token = ' ' + predicted_token
        res_probs.append( values2[i,0]*values[i] if fill_masks>1 else values[i] )
        res_substs.append( predicted_token )

    res_substs, res_probs = np.array(res_substs), np.array(res_probs)
    sort_idx = res_probs.argsort()[::-1]
    res_substs, res_probs = res_substs[sort_idx], res_probs[sort_idx]
    if debug:
        logging.info('-->', ' '.join(f"'{s}' {p:.3f}" for p,s in zip(res_probs[:15],res_substs[:15])))

    return res_substs, res_probs


def print_hypos(self, indexes, values):
    assert len(indexes.shape) == 2 and len(values.shape) == 2 and indexes.shape == values.shape, \
        f'Incorrect shapes: {indexes.shape} and {values.shape}; shall be (bs,hypo_len)'
    hypos = self.task.source_dictionary.string(torch.as_tensor(indexes)).split('\n')
    weights = np.array2string(np.exp(np.log(values).sum(axis=-1)), formatter=fff, separator=' ').strip('[]').split(' ')
    logging.info('>' + ' '.join(f"'{h}' {w}" for h, w in zip(hypos, weights)))


def print_hypos3(self, indexes, values):
    assert len(indexes.shape) == 2 and len(values.shape) == 2 and indexes.shape==values.shape, \
        f'Incorrect shapes: {indexes.shape} and {values.shape}; shall be (bs,hypo_len)'
    hypos = self.task.source_dictionary.string(torch.as_tensor(indexes)).split('\n')
    weights = np.array2string(np.exp(np.log(values.cpu().detach().numpy()).sum(axis=-1)), formatter=fff, separator=' ').strip('[]').split(' ')
    logging.info(' '.join(f"'{h}' {w}" for h, w in zip(hypos, weights)))


def fill_mask_mwe_ltr_batch_beamsearch(self, masked_input: str, topk: int, beam_size: int,
                                       fill_masks=None, fix_spaces=False, bpe_tokenize=True,
                                       batch_size_tokens=4000, maxlen=510, debug=True):

    tokens, masked_index = prepare_inputs(self, masked_input, fix_spaces, bpe_tokenize, debug, maxlen)
    assert tokens.dim() == 2 and masked_index.dim() == 1, f'{tokens.shape}, {masked_index.shape}'

    if debug: print('Filling mask 0...')
    indexes, values = _fill_one_mask_batch(self, tokens, masked_index[0], beam_size)
    indexes, values = indexes.T, values.T
    assert indexes.shape == (beam_size, 1) and values.shape == (beam_size, 1)
    if debug:
        print_hypos(self, indexes, values)

    batch_size = batch_size_tokens // tokens.size()[-1]
    tokens = tokens.repeat(beam_size, 1)
    if debug: logging.info('Dynamic batch size:', batch_size, 'Tokens.shape / masked_index.shape:', tokens.shape, masked_index.shape)
    for mask_num in range(1, len(masked_index) if fill_masks is None else fill_masks):
        tokens[:, masked_index[:mask_num]] = torch.tensor(indexes, dtype=torch.int32)
        if debug:
            logging.info(f'Filling mask {mask_num}/{len(masked_index)}')
        indexes2, values2 = (np.vstack(m) for m in zip(*(
            _fill_one_mask_batch(self, tokens[st:st+batch_size], masked_index[mask_num], topk=beam_size)
            for st in range(0, len(tokens), batch_size))))
        if debug:
            logging.info('Hypos shape / continuations shape:', indexes.shape, indexes2.shape)
            for i in range(len(indexes)):
                logging.info(self.task.source_dictionary.string(indexes[i, :]), '->',
                      self.task.source_dictionary.string(indexes2[i][:10]),
                      np.array2string(values2[i][:10], formatter=fff))
        assert values.shape == (beam_size,mask_num) and values2.shape == (beam_size, beam_size), \
            f'Incorrect shapes: {values.shape} and {values2.shape}'
        # multiply probs of each previous hypothesis and beam_size possible continuations:
        # P(hypo_i)*P(cont_ij|hypo_i)
        logprobs = (np.log(values).sum(axis=-1, keepdims=True) + np.log(values2)).ravel()
        ids = logprobs.argsort()[::-1][:beam_size]  # take beam_size most probably hypothesis
        # logging.info(np.exp(logprobs[ids]))
        ii, jj = ids//beam_size, ids%beam_size
        # logging.info(indexes[ii].shape, indexes2[ii,jj].shape)
        indexes = np.hstack([indexes[ii], indexes2[ii, jj].reshape(-1, 1)])
        values = np.hstack([values[ii], values2[ii, jj].reshape(-1, 1)])
        if debug:
            print_hypos(self, indexes, values)

    bpe_strings = self.task.source_dictionary.string(torch.as_tensor(indexes)).split('\n')
    # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
    res_substs = np.array([(' ' if x.startswith('\u2581') else '') + self.bpe.decode(x) for x in bpe_strings])[:topk]
    res_probs = np.exp(np.log(values).sum(axis=-1))[:topk]
    if debug:
        logging.info('-->', ' '.join(f"'{s}' {p:.3f}" for p, s in zip(res_probs[:15], res_substs[:15])))

    return res_substs, res_probs


def fill_mask_mwe_ltr_batch_beamsearch2(self, masked_input: str, topk: int, beam_size: int, cont_greedy: bool,
                                        fill_masks=None, fix_spaces=False, bpe_tokenize=True,
                                        batch_size_tokens=4000, maxlen=510, debug=True):

    tokens, masked_index = prepare_inputs(self, masked_input, fix_spaces, bpe_tokenize, debug, maxlen)
    assert tokens.dim() == 2 and masked_index.dim() == 1, f'{tokens.shape}, {masked_index.shape}'
    cont_topk = 1 if cont_greedy else beam_size
    if debug:
        logging.info('Filling mask 0...')
    indexes, values = _fill_one_mask_batch(self, tokens, masked_index[0], beam_size)
    indexes, values = indexes.T, values.T
    assert indexes.shape == (beam_size, 1) and values.shape == (beam_size, 1)
    if debug:
        print_hypos(self, indexes, values)

    batch_size = batch_size_tokens // tokens.size()[-1]
    tokens = tokens.repeat(beam_size, 1)
    if debug:
        logging.info('Dynamic batch size:', batch_size, 'Tokens.shape / masked_index.shape:', tokens.shape, masked_index.shape)
    for mask_num in range(1, len(masked_index) if fill_masks is None else fill_masks):
        tokens[:, masked_index[:mask_num]] = torch.tensor(indexes, dtype=torch.int32)
        if debug:
            logging.info(f'Filling mask {mask_num}/{len(masked_index)}')
        indexes2, values2 = (np.vstack(m) for m in zip(*(
            _fill_one_mask_batch(self, tokens[st:st+batch_size], masked_index[mask_num], topk=cont_topk)
            for st in range(0, len(tokens), batch_size))))
        if debug:
            logging.info('Hypos shape / continuations shape:', indexes.shape, indexes2.shape)
            for i in range(len(indexes)):
                logging.info(self.task.source_dictionary.string(indexes[i, :]), '->',
                      self.task.source_dictionary.string(indexes2[i][:10]),
                      np.array2string(values2[i][:10], formatter=fff))
        assert values.shape == (beam_size,mask_num) and values2.shape == (beam_size, cont_topk), \
            f'Incorrect shapes: {values.shape} and {values2.shape}'
        # multiply probs of each previous hypothesis and beam_size possible continuations:
        # P(hypo_i)*P(cont_ij|hypo_i)
        logprobs = (np.log(values).sum(axis=-1, keepdims=True) + np.log(values2)).ravel()
        ids = logprobs.argsort()[::-1][:beam_size]  # take beam_size most probably hypothesis
        # logging.info(np.exp(logprobs[ids]))
        ii, jj = ids//cont_topk, ids%cont_topk
        # logging.info(indexes[ii].shape, indexes2[ii,jj].shape)
        indexes = np.hstack([indexes[ii], indexes2[ii, jj].reshape(-1, 1)])
        values = np.hstack([values[ii], values2[ii, jj].reshape(-1, 1)])
        if debug:
            print_hypos(self, indexes, values)

    bpe_strings = self.task.source_dictionary.string(torch.as_tensor(indexes)).split('\n')
    # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
    res_substs = np.array([(' ' if x.startswith('\u2581') else '') + self.bpe.decode(x) for x in bpe_strings])[:topk]
    res_probs = np.exp(np.log(values).sum(axis=-1))[:topk]
    if debug:
        logging.info('-->', ' '.join(f"'{s}' {p:.3f}" for p, s in zip(res_probs[:15], res_substs[:15])))

    return res_substs, res_probs


def fill_mask_mwe_ltr_batch_beamsearch3(self, masked_input: str, topk: int, beam_size: int, cont_greedy: bool,
                                        fill_masks=None, fix_spaces=False, bpe_tokenize=True,
                                        batch_size_tokens=4000, maxlen=510, debug=True, remove_masks=False):

    tokens, masked_index = prepare_inputs(self, masked_input, fix_spaces, bpe_tokenize, debug, maxlen)
    if remove_masks:
        tokens = tokens[tokens != self.task.mask_idx]
        tokens = tokens.view(1, -1)
    tokens = tokens.to(self.device)
    masked_index = masked_index.to(self.device)
    assert tokens.dim() == 2 and masked_index.dim() == 1, f'{tokens.shape}, {masked_index.shape}'
    cont_topk = 1 if cont_greedy else beam_size
    if debug:
        logging.info('Filling mask 0...')
    indexes, values = _fill_one_mask_batch3(self, tokens, masked_index[0], beam_size)
    indexes, values = indexes.T, values.T
    assert indexes.shape == (beam_size, 1) and values.shape == (beam_size, 1)
    if debug:
        print_hypos3(self, indexes, values)

    batch_size = batch_size_tokens // tokens.size()[-1]
    tokens = tokens.repeat(beam_size, 1)
    if debug:
        print('Dynamic batch size:', batch_size, 'Tokens.shape / masked_index.shape:', tokens.shape, masked_index.shape)
    for mask_num in range(1, len(masked_index) if fill_masks is None else fill_masks):
        tokens[:, masked_index[:mask_num]] = indexes.type(torch.int32)
        if debug: print(f'Filling mask {mask_num}/{len(masked_index)}')
        indexes2, values2 = (torch.cat(m, dim=0) for m in zip(*(
            _fill_one_mask_batch3(self, tokens[st:st + batch_size], masked_index[mask_num], topk=cont_topk)
            for st in range(0, len(tokens), batch_size))))
        if debug:
            print('Hypos shape / continuations shape:', indexes.shape, indexes2.shape)
            for i in range(len(indexes)):
                print(self.task.source_dictionary.string(indexes[i,:]), '->',
                      self.task.source_dictionary.string(indexes2[i][:10]),
                      np.array2string(values2[i][:10].cpu().detach().numpy(), formatter=fff) )
        assert values.shape == (beam_size, mask_num) and values2.shape == (beam_size, cont_topk), \
            f'Incorrect shapes: {values.shape} and {values2.shape}'
        # multiply probs of each previous hypothesis and beam_size possible continuations:
        # P(hypo_i)*P(cont_ij|hypo_i)
        logprobs = (torch.log(values).sum(axis=-1, keepdims=True) + torch.log(values2)).reshape(-1)
        ids = logprobs.argsort().flip(dims=(0,))[:beam_size]  # take beam_size most probably hypothesis
        # logging.info(np.exp(logprobs[ids]))
        ii, jj = ids // cont_topk, ids % cont_topk
        # logging.info(indexes[ii].shape, indexes2[ii,jj].shape)
        indexes = torch.cat([indexes[ii], indexes2[ii, jj].reshape(-1, 1)], dim=-1)
        values = torch.cat([values[ii], values2[ii, jj].reshape(-1, 1)], dim=-1)
        if debug: print_hypos3(self, indexes, values)

    bpe_strings = self.task.source_dictionary.string(indexes).split('\n')
    # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
    res_substs = np.array([(' ' if x.startswith('\u2581') else '') + self.bpe.decode(x) for x in bpe_strings])[:topk]
    res_probs = np.exp(np.log(values.cpu().detach().numpy()).sum(axis=-1))[:topk]
    if debug:
        logging.info('-->' + ' '.join(f"'{s}' {p:.3f}" for p, s in zip(res_probs[:15], res_substs[:15])))

    return res_substs, res_probs


def fill_mask_mwe_ltr_batch_beamsearch4(self, masked_input: str, topk: int, beam_size: int, cont_greedy: bool,
                                        fill_masks=None, fix_spaces=False, bpe_tokenize=True,
                                        batch_size_tokens=4000, maxlen=510, debug=True, remove_masks=False):

    splitted_masked_input = masked_input.split('<mask>')
    num_masks = len(splitted_masked_input) - 1
    masked_input = splitted_masked_input[0] + '<mask>' + splitted_masked_input[-1]
    tokens, masked_index = prepare_inputs(self, masked_input, fix_spaces, bpe_tokenize, debug, maxlen)
    if remove_masks:
        tokens = tokens[tokens != self.task.mask_idx]
        tokens = tokens.view(1, -1)
    tokens = tokens.to(self.device)
    masked_index = masked_index.to(self.device) + torch.arange(0, num_masks, device=self.device)
    assert tokens.dim() == 2 and masked_index.dim() == 1, f'{tokens.shape}, {masked_index.shape}'
    cont_topk = 1 if cont_greedy else beam_size
    if debug:
        logging.info('Filling mask 0...')
    indexes, values = _fill_one_mask_batch3(self, tokens, masked_index[0], beam_size)
    indexes, values = indexes.T, values.T
    assert indexes.shape == (beam_size, 1) and values.shape == (beam_size, 1)
    if debug:
        print_hypos3(self, indexes, values)
    # У меня нет лука (а также <mask><mask><mask>) для салата
    # --model_path="../xlmr_prefixmask/model.pt"
    batch_size = batch_size_tokens // tokens.size()[-1]
    tokens = tokens.repeat(beam_size, 1)
    if debug:
        print('Dynamic batch size:', batch_size, 'Tokens.shape / masked_index.shape:', tokens.shape, masked_index.shape)
    for mask_num in range(1, num_masks):
        # tokens[:, masked_index[:mask_num]] = indexes.type(torch.int32)
        tokens = torch.cat([
            tokens[:, :masked_index[mask_num - 1]],
            indexes[:, [-1]],
            tokens[:, masked_index[mask_num - 1]:]
        ], dim=1)

        if debug: print(f'Filling mask {mask_num}/{len(masked_index)}')
        assert (tokens[:, masked_index[mask_num]] == 250_001).all()
        indexes2, values2 = (torch.cat(m, dim=0) for m in zip(*(
            _fill_one_mask_batch3(self, tokens[st:st + batch_size], masked_index[mask_num], topk=cont_topk)
            for st in range(0, len(tokens), batch_size))))
        if debug:
            print('Hypos shape / continuations shape:', indexes.shape, indexes2.shape)
            for i in range(len(indexes)):
                print(self.task.source_dictionary.string(indexes[i,:]), '->',
                      self.task.source_dictionary.string(indexes2[i][:10]),
                      np.array2string(values2[i][:10].cpu().detach().numpy(), formatter=fff) )
        assert values.shape == (beam_size, mask_num) and values2.shape == (beam_size, cont_topk), \
            f'Incorrect shapes: {values.shape} and {values2.shape}'
        # multiply probs of each previous hypothesis and beam_size possible continuations:
        # P(hypo_i)*P(cont_ij|hypo_i)
        # logprobs = (torch.log(values).sum(axis=-1, keepdims=True) + torch.log(values2)).reshape(-1)
        # ids = logprobs.argsort().flip(dims=(0,))[:beam_size]  # take beam_size most probably hypothesis
        # # logging.info(np.exp(logprobs[ids]))
        # ii, jj = ids // cont_topk, ids % cont_topk
        # logging.info(indexes[ii].shape, indexes2[ii,jj].shape)
        indexes = torch.cat([indexes, indexes2], dim=-1)
        values = torch.cat([values, values2], dim=-1)
        if debug: print_hypos3(self, indexes, values)

    bpe_strings = self.task.source_dictionary.string(indexes).split('\n')
    new_lens = [t.split('\u2581')[1].strip().count(' ') + 1 if '\u2581' in t else num_masks for t in bpe_strings]
    res_probs = [v[:new_lens[i]].prod().item() for i, v in enumerate(values)]
    # res_probs = values[:, 0].cpu().numpy()
    new_ordering = np.argsort(res_probs)[::-1][:topk]
    # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
    res_substs = np.array([" " + self.bpe.decode(x).split(' ')[0] for x in bpe_strings])
    res_substs = res_substs[new_ordering]
    res_probs = np.array(res_probs)[new_ordering]

    # bpe_strings = self.task.source_dictionary.string(indexes).split('\n')
    # # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
    # res_substs = np.array([(' ' if x.startswith('\u2581') else '') + self.bpe.decode(x) for x in bpe_strings])[:topk]
    # res_probs = np.exp(np.log(values.cpu().detach().numpy()).sum(axis=-1))[:topk]
    if debug:
        print('-->', ' '.join(f"'{s}' {p:.3f}" for p, s in zip(res_probs[:15], res_substs[:15])))

    return res_substs, res_probs


# def fill_mask_mwe_ltr_batch_beamsearch4(self, masked_input: str, topk: int, beam_size: int, cont_greedy: bool,
#                                         fill_masks=None, fix_spaces=False, bpe_tokenize=True,
#                                         batch_size_tokens=4000, maxlen=510, debug=True, remove_masks=False):
#     splitted_input = masked_input.split('<mask>')
#
#     assert sum([len(t) for t in splitted_input[1:-1]]) == 0, "<mask> tokens are not placed continuosly"
#     context_l, context_r = splitted_input[0], splitted_input[-1]
#
#     num_masks = len(splitted_input) - 1
#
#


def interactive(model):
    topk = 10
    # try:
    for s in sys.stdin:
        ss, pp = fill_mask_mwe_ltr_batch_beamsearch4(model, s, topk=topk, beam_size=topk * 2, fill_masks=None,
                                                     fix_spaces=True, cont_greedy=True, debug=True)
        # logging.info(ss.shape, pp.shape, ss.dtype, pp.dtype, ss[0], pp[0])
        # if len(s.split('<mask>')) == 2:
        #     ss, pp = fill_mask(model, s, topk=topk, debug=True)
        # else:
        #     ss, pp = fill_mask_mwe_ltr_batch(model, s, topk=topk, fill_masks=2, debug=True)
        print(ss, pp)

            # logging.info(ss.shape, pp.shape, ss.dtype, pp.dtype, ss[0], pp[0])
            # except:  # catch *all* exceptions
            #     e = sys.exc_info()[0]
            #     traceback.print_exc()
            #     logging.info(e)
            #     interactive(model)


def process_target(target):
    # required for Semeval 2020 Task 1 English corpus where targets are appended with POS tags
    return target.split('_')[0]


def generate_substitutes(data_name=None, topk=500, templ='<mask>', fill_masks=2, fix_spaces=True, debug=True,
                         skip_rows=None, limit=None, model_path=None, max_ex_per_word=None, maxlen=510,
                         drop_duplicates=True, beam_search=True, rewrite_existing=True, cont_greedy=True,
                         version=3, skip_last_nmasks=0):
    '''
        version:
        0 (modelNone/ in path): only greedy search, seems correct, TODO: recover code if v2 is worse!
        1 (modelNone-beamsearch*/ in path): beam_search=False with >3 masks fills only 2 first masks
        2 (modelNone-beamsearch*-version2):
    '''
    logging.info(f"DEBUG is {debug}")
    if beam_search is False and cont_greedy is False:
        raise ValueError(f'Incompatible combination: beam_search={beam_search}, cont_greedy={cont_greedy}')

    model_path = Path(model_path) if model_path is not None else None
    model = load_model(model_path)

#    df = None
#    if dataframe is not None:
#        print('dataframe is given manually with data_name = %s' % data_name)
#        df = dataframe
#    else:

    if data_name is None:
        interactive(model)
        return

    templ = templ.replace('_', ' ').replace('<mask>', '<mask>' * fill_masks)
    if isinstance(data_name, list) is True:
        dirs, fname, df = data_name
        data_name = None
    else:

        df = pd.read_csv(f'../datasets/russe_bts-rnc/ru/{data_name}.tsv', sep='\t')
        df['positions'] = df.positions.str.split('-').apply(lambda x: (int(x[0]), int(x[1])))
        if drop_duplicates:
            df1 = df.drop_duplicates(subset=['context', 'positions'])
            logging.info(f'Duplicates: #ex before/after dropping duplicates: {len(df)}/{len(df1)}')
            df = df1
        else:
            duplicates = df.duplicated(subset=['context', 'positions']).mean()
            logging.info('Duplicates: {duplicates.sum()}/{len(df)} ({duplicates.mean()*100}%)')

        model_name = 'None' if model_path is None else (model_path.parent.name + '_' + model_path.name)
        dirs = Path(f'./{data_name}-limit{limit}-maxexperword{max_ex_per_word}-maxlen{maxlen}')/f'model{model_name}-beamsearch{beam_search}-contgreedy{cont_greedy}-version{version}'
        dirs.mkdir(parents=True, exist_ok=True)
        fname = dirs / f'{templ.replace(" ", "-").replace("<", "_").replace(">", "_")}-2ltr{fill_masks}f{skip_last_nmasks}s_topk{topk}_fixspaces{fix_spaces}.npz'

    if not rewrite_existing and os.path.isfile(fname):
        logging.info('templates already exist in the location %s' % fname)
        return fname

    logging.info(f'Dataset {data_name}: topk={topk}, templ="{templ}", fill_masks={fill_masks} with limit {limit}, max_ex_per_word {max_ex_per_word}, maxlen {maxlen}, beam_search {beam_search}, contgreedy {cont_greedy}, version {version}')
    logging.info(f'Saving results to:\n' + str(fname))

    if max_ex_per_word is not None:
        df = df.sample(frac=1).groupby('word').head(max_ex_per_word)
    df.to_csv(dirs / (fname.name+'.input'), index=False)
    wordat2cnt = df.apply(lambda r: r.context[r.positions[0]:r.positions[1]].strip(), axis=1).value_counts().to_dict()
    logging.info(f'Targets and counts: {wordat2cnt}')
    logging.info(f'Target counts: {pd.Series(list(wordat2cnt.values())).describe()}')
    logging.info(f'Target lengths: {pd.Series(list(wordat2cnt.keys())).apply(len).describe()}')
    assert all(len(w) > 0 for w in wordat2cnt.keys()), 'some positions are incorrect!'
    logging.info(f'Context lengths in words: {df.context.str.split(" ").apply(len).describe()}')
    logging.info(f'Context lengths in lines: ' + str(df.context.str.split("\n").apply(len).describe()))
    df['lctx'], df['rctx'] = df.apply(lambda r: r.context[:r.positions[0]], axis=1), \
                             df.apply(lambda r: r.context[r.positions[1]:], axis=1),
    df['target_word'] = df.apply(lambda r: r.context[r.positions[0]:r.positions[1]], axis=1)
    for sample in df.head(3), df.tail(3):
        logging.info('Left contexts:\n' + '\n'.join(sample.lctx))
        logging.info('Right contexts:\n' + '\n'.join(sample.rctx))
        logging.info('Tearget words:\n' + '\n'.join(sample.target_word))

    df = df.iloc[skip_rows:limit]
    context_masked = df.apply(lambda r:
                              r.context[:r.positions[0]]
                              + templ.replace('T', process_target(r.context[r.positions[0]:r.positions[1]]))
                                .replace('L', lemmatize(process_target(r.context[r.positions[0]:r.positions[1]])))
                              + r.context[r.positions[1]:], axis=1)
    with torch.no_grad():
        from fairseq import utils
        with utils.model_eval(model.model):
            if version == 1:
                    if beam_search:
                        res_ser = context_masked.apply(lambda s:
                                                       fill_mask_mwe_ltr_batch_beamsearch(model, s, topk=topk, beam_size=topk*2,
                                                                                          fill_masks=fill_masks-skip_last_nmasks,
                                                                                          fix_spaces=fix_spaces, debug=debug,
                                                                                          maxlen=maxlen))
                    elif len(templ.split('<mask>')) == 2:
                        res_ser = context_masked.apply(lambda s: fill_mask(model, s, topk=topk, debug=debug,
                                                                           fix_spaces=fix_spaces, maxlen=maxlen))
                    else:
                        res_ser = context_masked.apply(lambda s: fill_mask_mwe_ltr_batch(model, s, topk=topk, debug=debug,
                                                                                         fill_masks=fill_masks-skip_last_nmasks,
                                                                                         fix_spaces=fix_spaces, maxlen=maxlen))
            elif version == 2:
                beam_size = topk*5 if beam_search else topk
                res_ser = context_masked.apply(lambda s:
                                               fill_mask_mwe_ltr_batch_beamsearch2(model, s, topk=topk, beam_size=beam_size,
                                                                                   cont_greedy=cont_greedy,
                                                                                   fill_masks=fill_masks-skip_last_nmasks,
                                                                                   fix_spaces=fix_spaces, debug=debug,
                                                                                   maxlen=maxlen))
            elif version == 3:
                beam_size = topk * 5 if beam_search else topk
                from fairseq import utils
                with utils.model_eval(model.model):
                    with torch.no_grad():
                        res_ser = context_masked.progress_apply(
                            lambda s: fill_mask_mwe_ltr_batch_beamsearch3(model, s, topk=topk, beam_size=beam_size,
                                                                          cont_greedy=cont_greedy,
                                                                          fill_masks=fill_masks - skip_last_nmasks,
                                                                          fix_spaces=fix_spaces, debug=debug, maxlen=maxlen,
                                                                          remove_masks=(templ=='<mask>T')
                                                                          ))
            elif version == 4:
                beam_size = topk if beam_search else topk
                from fairseq import utils
                with utils.model_eval(model.model):
                    with torch.no_grad():
                        res_ser = context_masked.progress_apply(
                            lambda s: fill_mask_mwe_ltr_batch_beamsearch4(model, s, topk=topk, beam_size=beam_size,
                                                                          cont_greedy=True,
                                                                          fill_masks=fill_masks - skip_last_nmasks,
                                                                          fix_spaces=fix_spaces, debug=debug, maxlen=maxlen,
                                                                          remove_masks=(templ=='<mask>T')
                                                                          ))
            else:
                raise ValueError('Version %d is currently not implemented!' % version)

    res_substs, res_probs = (np.array(list(res_ser.str[i])) for i in (0, 1))
    np.savez_compressed(fname, substs=res_substs, probs=res_probs)
    logging.info(f'Substs {res_substs.shape} and probs {res_probs.shape} saved to {fname}\n')
    logging.info(f'Substs {res_substs[:3]} and probs {res_probs[:3]} ')

    return fname


if __name__ == '__main__':
    fire.Fire(generate_substitutes)

# --data_name=train --templ=T_(или_даже_<mask>) --topk=150 --fill_masks=2