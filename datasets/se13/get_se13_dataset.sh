#!/bin/bash
wget https://zenodo.org/record/5638384/files/SemEval-2013-Task-13-test-data.zip

echo Unzipping...
unzip -q *zip

zip_dir=*zip
dir=$(basename $zip_dir .zip)
words_dir=$dir'/contexts/xml-format'
labels_file=$dir'/keys/gold/all.key'

echo Converting...
cp $dir'/README.txt' 'README.txt'
python se13_to_bts-rnc.py -p $words_dir -l $labels_file -s en/semeval13.tsv

rm $zip_dir
rm -r $dir