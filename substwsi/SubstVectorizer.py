from substwsi.substs_loading import load_substs
from substwsi.max_ari import preprocess_substs, get_nf_cnt
from multilang_wsi_evaluation.interfaces import IWSIVectorizer


class SubstVectorizer(IWSIVectorizer):
    def __init__(self, topk, data_name, lemmatize=True, exclude=[], path_to_substs=None):
        self.inside_vectorizer = None
        self.path_to_substs = path_to_substs
        self.data_name = data_name
        self.dfs_substs = {}
        self.parser = {}
        self.nf_cnts = {}
        self.substs_texts = {}
        self.topk = topk
        self.lemmatize = lemmatize
        self.exclude = exclude
        for key, value in path_to_substs.items():
            self.dfs_substs[key] = load_substs(value, data_name=data_name)
            list_words_from_dataset = list(set(self.dfs_substs[key].word))
            self.parser.update({i: key for i in list_words_from_dataset})
            self.nf_cnts[key] = get_nf_cnt(self.dfs_substs[key]['substs_probs'])
            self.substs_texts[key] = self.dfs_substs[key].apply(
                lambda r: preprocess_substs(r.substs_probs[:self.topk], nf_cnt=self.nf_cnts[key],
                                            lemmatize=self.lemmatize,
                                            exclude_lemmas=self.exclude + [r.word]), axis=1).str.join(' ')

    def fit(self, all_samples):
        pass

    def predict(self, samples):
        word = samples[0].lemma
        key = self.parser[word]
        mask = (self.dfs_substs[key].word == word)
        vectors = self.inside_vectorizer.fit_transform(self.substs_texts[key][mask]).toarray()
        return vectors
