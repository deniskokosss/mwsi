from lexsubgen.datasets.wsi import SemEval2010DatasetReader
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mosestokenizer import *
import os
from tqdm import tqdm
import pandas as pd

data_reader = SemEval2010DatasetReader()
data, labels, path = data_reader.read_dataset()
data['text'] = data['sentence'].apply(lambda r: ' '.join(r))
data['gold_sense_id'] = data['context_id'].apply(lambda r: labels[r])
data['word'] = data.apply(lambda r: r.sentence[r.target_id], axis=1)
detokenize = MosesDetokenizer('en')
data['text'] = data['sentence'].apply(lambda r: detokenize(r))
def get_position(r):
    start_pos = r.text.find(r.word)
    return f'{start_pos}-{start_pos + len(r.word)}'
data['positions'] = data.apply(lambda r: get_position(r), axis=1)
data_last = data.drop(columns=['group_by', 'word', 'pos_tag', 'sentence', 'target_id'])
data_last = data_last.rename(columns={'target_lemma': 'word', 'pos': 'positions', 'text': 'context'})
data_last = data_last.reindex(['context_id','word','gold_sense_id', 'predict_sense_id','positions','context'], axis=1)
if 'datasets' not in os.listdir():
    os.mkdir('datasets')
if 'semeval-2010' not in os.listdir('datasets'):
    os.mkdir('datasets/semeval-2010')
if 'en' not in os.listdir('datasets/semeval-2010'):
    os.mkdir('datasets/semeval-2010/en')
data_last.to_csv('datasets/semeval-2010/en/_train.tsv', sep='\t')
    
