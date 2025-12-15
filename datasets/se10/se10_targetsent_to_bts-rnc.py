import csv
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import argparse

import numpy as np
from lexsubgen.datasets.utils import download_dataset
from lexsubgen.datasets.wsi import SemEval2010DatasetReader
from lexsubgen.utils.wsi import SEMEVAL2010URL, SEMEVAL2010TESTURL
import spacy
import pandas as pd


# Code taken from https://github.com/asafamr/bertwsi/blob/master/wsi/semeval_utils.py

def se10_targetsent_amrami(dir_path: Path, out_path: Path):
    logging.info('reading SemEval dataset from %s' % dir_path)
    substs = {
        'context_id': [],
        'context': [],
        'positions': [],
    }
    nlp = spacy.load('en', disable=['ner'])
    additional_mapping = {'stuck': 'stick', 'straightened': 'straighten', 'shaved': 'shave', 'shaving': 'shave',
                          'swam': 'swim', 'figgere': 'figure', 'violating': 'violate', 'lain': 'lie', 'lied': 'lie',
                          'figger': 'figure', 'swore': 'swear', 'swears': 'swear', 'observed': 'observe',
                          'committed': 'commit', 'divided': 'divide', 'lie': 'lay', 'lay': 'lie', 'lah': 'lie',
                          'swimming': 'swim'}

    def basic_stem(w):
        if w[-1] == 's':
            w = w[:-1]
        elif w[-3:] == 'ing':
            w = w[:-3]
        elif w[-2:] == 'ed':
            w = w[:-2]
        return w.lower()

    for root_dir, dirs, files in os.walk(dir_path):  # "../paper-menuscript/resources/SemEval-2010/test_data/"):
        #     path = root.split(os.sep)
        for file in files:
            if '.xml' in file:
                tree = ElementTree.parse(os.path.join(root_dir, file))
                root = tree.getroot()
                for child in root:
                    inst_name = child.tag
                    lemma = inst_name.split('.')[0]

                    stemmed_lemma = basic_stem(lemma)

                    # pres_sent = child.text
                    target_sent = child[0].text
                    # post_sent = child[0].tail - use only

                    # if not pres_sent:
                    #     pres_sent = ''
                    # if not post_sent:
                    #     post_sent = ''

                    # the word is not marked here, so we need to find the lemma within our sentence
                    # - this does the trick for bert uncased in SE2010
                    parsed = nlp(target_sent)
                    first_occur_idx = None
                    for idx, w in enumerate(parsed):
                        token_lemma = basic_stem(
                            w.lemma_)
                        if token_lemma == stemmed_lemma or additional_mapping.get(w.lemma_.lower()) == lemma:
                            first_occur_idx = idx
                            break
                    if first_occur_idx is None:
                        print(
                            'could not find the correct lemma -probably spacy\'s lemmatizer had changed. '
                            'add the lemma from here to additional_mapping:')
                        print(file, [x.lemma_ for x in parsed], target_sent)
                        # e.g. if you see lie.v was broken and in the list of lemmas you find 'lain'
                        # - add a mapping from 'lain' -> 'lie' in additional_mapping map above
                        raise Exception('Could not pin-point lemma in SemEval sentence')

                    # pre = pres_sent + ' ' + ''.join(parsed[i].string for i in range(first_occur_idx))
                    pre = ''.join(parsed[i].string for i in range(first_occur_idx))
                    ambig = parsed[first_occur_idx].text
                    post = ''.join(
                        parsed[i].string for i in range(first_occur_idx + 1, len(parsed)))  # + ' ' + post_sent

                    pre = pre.replace(" 's ", "'s ")
                    post = post.replace(" 's ", "'s ")
                    # Generating local version of substs
                    substs['context_id'].append(inst_name)
                    substs['context'].append(' '.join([pre.strip(), ambig.strip(), post.strip()]))
                    substs['positions'].append((len(pre.strip()) + 1, len(pre.strip()) + len(ambig) + 1))
    return pd.DataFrame(substs)


def add_gold_labels(contexts: pd.DataFrame, gold_labels_path: Path):
    data_reader = SemEval2010DatasetReader()
    gold_labels = data_reader.read_gold_labels(str(gold_labels_path))
    gold_labels = pd.DataFrame(
        data={
            'context_id': list(gold_labels.keys()),
            'gold_sense_id': list(gold_labels.values()),
        }
    )
    return contexts.merge(gold_labels, on='context_id')


parser = argparse.ArgumentParser(description='Download and transform semeval10 dataset in btc-rnc format')
parser.add_argument('--use_amrami_version', type=bool,
                    help='Use same preprocessing as used in https://arxiv.org/abs/1905.12598 paper',
                    default=True)
parser.add_argument('--out_path', type=Path,
                    help='Path to result substs',
                    default=Path('en/semeval10_target_sentence.tsv'))

if __name__ == "__main__":
    args = parser.parse_args()
    with TemporaryDirectory() as temp_dir:
        download_dataset(SEMEVAL2010URL, temp_dir)
        download_dataset(SEMEVAL2010TESTURL, temp_dir)
        if args.use_amrami_version:
            contexts = se10_targetsent_amrami(
                Path(temp_dir) / 'test_data',
                out_path=args.out_path
            )
        else:
            raise NotImplementedError
        res = add_gold_labels(contexts,
                              gold_labels_path=Path(temp_dir) / 'evaluation' / 'unsup_eval' / 'keys' / 'all.key')
        res['predict_sense_id'] = np.NaN
        res['word'] = res['context_id'].apply(lambda x: x.split('.')[0])
        res['positions'] = res['positions'].apply(lambda x: f'{x[0]}-{x[1]}')
        res[['context_id','word','gold_sense_id','predict_sense_id','positions','context']].to_csv(
            args.out_path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True
        )
