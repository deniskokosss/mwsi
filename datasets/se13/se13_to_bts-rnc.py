import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd


def get_df(file_path: Path) -> pd.DataFrame:

    # get list of instances (without root node)
    xml_tree = ET.parse(str(file_path))
    tree_root = xml_tree.getroot()
    instances = list(tree_root.iter())[1:]

    instance_dfs: List[pd.DataFrame] = list()
    for instance in instances:
        context_id, lemma, _, token, end_idx, start_idx = list(map(lambda x: x[1], instance.items()))
        instance_dfs.append(pd.DataFrame({'context_id': [context_id],
                                          'word': [lemma],
                                          'positions': [f'{start_idx}-{end_idx}'],
                                          'context': [instance.text]}))

    return pd.concat(instance_dfs)


def get_gold_labels(file_path: Path) -> pd.DataFrame:

    instances = defaultdict(list)
    word_unique_labels = defaultdict(set)

    for line in file_path.open('r').readlines():
        line = line.strip().split(' ')
        word, context_id, labels = line[0].split('.')[0], line[1], line[2:]

        labels = {label.split('/')[0]: float(label.split('/')[1]) for label in labels}
        word_unique_labels[word].update(labels.keys())

        instances['word'].append(word)
        instances['context_id'].append(context_id)
        instances['raw_labels'].append(labels)

    word_unique_labels_to_id = {word: {label: id_ for id_, label in enumerate(unique_labels)}
                                for word, unique_labels in word_unique_labels.items()}

    gold_sense_id = list()
    for instance_id, raw_instance_labels in enumerate(instances['raw_labels']):

        word = instances['word'][instance_id]
        instance_labels = [0] * len(word_unique_labels[word])

        for label, weight in raw_instance_labels.items():
            instance_labels[word_unique_labels_to_id[word][label]] = weight

        gold_sense_id.append(instance_labels)

    df = pd.DataFrame(data={'word': instances['word'],
                            'context_id': instances['context_id'],
                            'gold_sense_id': list(map(lambda label_weights: ' '.join(str(lw) for lw in label_weights), gold_sense_id))})
    return df.set_index('context_id')


if __name__ == '__main__':

    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_words', type=str, metavar='<path to directory containing files of words>', required=True)
    parser.add_argument('-l', '--path_to_labels', type=str, metavar='<path to file with all labels>', required=True)
    parser.add_argument('-s', '--save_path', type=str, metavar='<path to save>', required=True)

    args = parser.parse_args()
    words_dir = Path(args.path_to_words)
    save_path = Path(args.save_path)
    labels_file = Path(args.path_to_labels)

    # build dataset
    dataset = pd.concat([get_df(file) for file in words_dir.iterdir()]).set_index('context_id')
    dataset['predict_sense_id'] = None

    # add gold_sense_id and reorder columns
    dataset = pd.merge(dataset, get_gold_labels(labels_file), on=['word', 'context_id'])
    dataset = dataset[['word', 'gold_sense_id', 'predict_sense_id', 'positions', 'context']]

    # save dataset
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(save_path, sep='\t', index=True, quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True)
