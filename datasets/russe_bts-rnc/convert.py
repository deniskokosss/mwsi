import csv
from pathlib import Path

import pandas as pd


def fix_end(pos: str) -> str:
    begin, end = pos.split('-')
    return f'{begin}-{int(end)+1}'


def save_df(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True)


def read_df(df_path: Path) -> pd.DataFrame:
    df = pd.read_csv(df_path, sep='\t')
    df['positions'] = df['positions'].apply(fix_end)
    return df


def process_train(source_dir: Path, output_dir: Path) -> None:
    train_df = read_df(source_dir / 'train.csv')
    save_df(train_df, output_dir / '_train.tsv')


def process_test(source_dir: Path, output_dir: Path) -> None:
    test_df = pd.read_csv(source_dir / 'test-solution.csv', sep='\t')
    test_df['positions'] = test_df['positions'].apply(fix_end)

    with open(source_dir / 'public.txt') as f:
        public_words = {line.rstrip() for line in f}

    words_mask = test_df['word'].isin(public_words)
    test_public_df = test_df.loc[words_mask]
    test_private_df = test_df.loc[~words_mask]
    assert len(test_public_df) + len(test_private_df) == len(test_df)

    save_df(test_public_df, output_dir / 'test-public.tsv')
    save_df(test_private_df, output_dir / 'test-private.tsv')


def main() -> None:
    source_dir = Path('../../russe-wsi-kit/data/main/bts-rnc')
    output_dir = Path('ru')

    process_train(source_dir, output_dir)
    process_test(source_dir, output_dir)


if __name__ == '__main__':
    main()
