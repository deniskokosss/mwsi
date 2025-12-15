# DWUG datasets

Datasets from https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/wugs/
Dominik Schlechtweg, Nina Tahmasebi, Simon Hengchen, Haim Dubossarsky, Barbara McGillivray. 2021. DWUG: A large Resource of Diachronic Word Usage Graphs in Four Languages

To download the original data and convert it to WSI format run:
```
bash get_se20lscd.sh
```

The datasets are stored in:
datasets_unlabeled/se20lscd_v2/ - all uses from DWUG sampled randomly from corresponding corpora, unlabeled;
datasets/se20lscd_v2 - labeled uses, cannot be assumed a random sample due to filtering during the annotation procedure.

sense-*.tsv (only for German, only 826 uses) contains uses annoted with their senses by humans, only uses identically annotated by 3 annotators are included;
presumably clean but biased data due to strong filtering. Due to small size and unambiguous examples may fit for error analysis better than for evaluation.

opt-*.tsv (~7-9K uses for German, English, Swedish) contains uses with senses inferred automatically from human pairwise annotations; presumably noisy data.

*-old.tsv, *-new.tsv, *-old+new.tsv - each subset is split into 3 files, containing uses from the old time period, or new time period, or both of them.

Reasonable comparisons may include:
1. The quality of clustering new uses alone, or with the addition of old uses without labels. Does the presence of old uses effect clustering of new uses? Notice that unlabeled uses can be loaded from datasets_unlabeled/se20lscd_v2 (randomly sampled, presumably more ambiguous and noisy uses) or datasets/se20lscd_v2 (sense-*.tsv contains much more clear examples, though not randomly sampled at all)
2. The same for old uses.
3. The quality of clustering new uses alone, or with the addition of more new uses without labels. Does augmentation help? Notice, that unlabeled subsets from datasets_unlabeled/se20lscd_v2 include uses from the labeled subsets! Also in the case of sense-*.tsv the labeled data is heavily filtered and is likely much less ambiguous and difficult than the unlabeled uses.
4. Comparing the quality of clustering for *-new.tsv, *-old+new.tsv directly can still be interesting, though seems not very sound since the test sets are different. How can we discover the possible issue when old and new examples end in different clusters even when having the same sense?


