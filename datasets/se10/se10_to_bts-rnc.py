from lexsubgen.datasets.wsi import SemEval2010DatasetReader
from mosestokenizer import *
import csv
import argparse


def get_position(r):
    start_pos = r.context.find(r.word)
    return f'{start_pos}-{start_pos + len(r.word)}'

parser = argparse.ArgumentParser()
parser.add_argument("--path", default='en/semeval10.tsv')
args = parser.parse_args()

detokenize = MosesDetokenizer('en')

data_reader = SemEval2010DatasetReader()
data, labels, path = data_reader.read_dataset()

#data['text'] = data['sentence'].apply(lambda r: ' '.join(r))
data['gold_sense_id'] = data['context_id'].apply(lambda r: labels[r])
data['word'] = data.apply(lambda r: r.sentence[r.target_id], axis=1)
data['context'] = data['sentence'].apply(lambda r: detokenize(r))
data['predict_sense_id'] = [None for _ in range(len(data))]
data['positions'] = data.apply(lambda r: get_position(r), axis=1)
data = data.drop(columns=['group_by', 'word', 'pos_tag', 'sentence', 'target_id'])
data = data.rename(columns={'target_lemma': 'word'})
order_data = data[['context_id', 'word', 'gold_sense_id', 'predict_sense_id', 'positions', 'context']]
order_data.to_csv(args.path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True)
