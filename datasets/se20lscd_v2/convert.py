import os
import argparse
import pandas as pd
import csv
from collections import Counter
import numpy as np
from pathlib import Path
import fire

def agg_annotations(l, min_annots=3, min_agreed=3):
    '''
    Leaves only uses annotated at least by min_annots annotators, among which 
    at least min_agreed returned the same sense.
    '''
    if len(l) < min_annots:
        return None
    # take the most frequent answer, or None if less than min_agreed annotators agreed
    # in case of tie, among the most frequent answers most_common() returns the first appeared one 
    label, cnt = Counter(l).most_common(1)[0]
    return None if cnt < min_agreed else label

def load_sense_labels(judgments_senses_path):
    '''
    Loads sense annotations, aggregates annotations by multiple annotators for a singe use.
    '''
    df = pd.read_csv(judgments_senses_path, delimiter="\t")
    senses = pd.read_csv(judgments_senses_path.parent / 'senses.csv', delimiter="\t")
    df = pd.merge(df, senses, on='identifier_sense')
    df = df[~(df['description_sense'] == 'andere')] # 'andere' stands for 'all other senses'
    clusters = df.groupby('identifier')['identifier_sense'].apply(agg_annotations).dropna().reset_index()
    clusters = clusters.rename(columns={'identifier_sense': 'cluster'})
    return clusters

def convert_save(df, fpath):
    res = pd.DataFrame({
        'context_id': df.identifier,
        'word': df.lemma,
        'gold_sense_id': df.cluster,
        'positions': df.indexes_target_token.str.replace(":","-"),
        'context': df.context
    })
    print(len(res), fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(fpath, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True)
    

def convert(LANG, CLUSTERS_NAME=''):
    DIR_NAME = f'dwug_{LANG}'
    p = Path(DIR_NAME)

    if CLUSTERS_NAME == '':
        clusters = None
        CLUSTERS_NAME = 'unlabeled' # for nice-looking output filename
    elif CLUSTERS_NAME == 'sense':
        paths = p.glob(f'data/*/judgments_senses.csv')
        clusters = pd.concat([load_sense_labels(path) for path in paths], ignore_index=True)
    else:
        paths = p.glob(f'clusters/{CLUSTERS_NAME}/*.csv')
        clusters = pd.concat([pd.read_csv(path,delimiter="\t", quoting=csv.QUOTE_NONE) for path in paths], ignore_index=True)

    paths = p.glob('data/*/uses.csv')
    uses = pd.concat([pd.read_csv(path,delimiter="\t", quoting=csv.QUOTE_NONE) for path in paths], ignore_index=True)
    uses['grouping'] = uses.grouping.replace(1,'old').replace(2,'new')
    print('Uses loaded:',uses.grouping.value_counts().to_dict())

    if clusters is None:
        rdf = uses
        rdf['cluster'] = None
        outpath = p.parent / '../../datasets_unlabeled/se20lscd_v2'
    else:
        rdf = clusters.merge(uses, on='identifier', how='inner', validate='1:1')
        assert len(clusters)==len(rdf), clusters[~clusters.identifier.isin(rdf.identifier)]
        print(f'{len(uses)} uses loaded, {len(rdf)} have gold labels')
        print('Uses with gold labels:',rdf.grouping.value_counts().to_dict())
        outpath = p.parent

    for pdf, part in [(rdf.query('grouping=="old"'), 'old'), (rdf.query('grouping=="new"'), 'new'), (rdf, 'old+new')]:
        mask = pdf["indexes_target_token"].str.len() > 2  # old code, not sure if we need it
        assert mask.all(), pdf[~mask]
        convert_save(pdf, outpath / LANG / f'{CLUSTERS_NAME}-{part}.tsv')    
            
fire.Fire(convert)
