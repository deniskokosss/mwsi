import abc
import string
import typing as tp
from functools import lru_cache

import pymorphy2
import spacy
import stanza
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer


class LemmatizerBase(metaclass=abc.ABCMeta):

    def __init__(self, mode='base', lang=None):
        """
            Base class for different languages lemmatizers
            Can implement simple or context modes
        """
        assert mode in ['disable', 'word', 'context']
        self.mode = mode
        self.lang = lang

    def parse(self, word: str, **kwargs):
        if self.mode == 'disable':
            return word
        if self.mode == 'word':
            return self.get_normal_form(word)
        elif self.mode == 'context':
            context = kwargs.get('context', None)
            positions = kwargs.get('positions', None)
            assert (context and positions), "You must provide context and word positions in CONTEXT mode"

            return self.get_normal_form_in_ctx(word, context, positions)

    def remove_punc(self, word):
        return word.translate(str.maketrans('', '', string.punctuation))

    @abc.abstractmethod
    def get_normal_form(self, word: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_normal_form_in_ctx(self, word: str, context: str, word_pos: tp.Tuple[int, int]) -> tp.List[str]:
        """
            Puts word in context, lemmatize result and return only lemmatized word
        """
        raise NotImplementedError


class LemmatizerPymorphy(LemmatizerBase):
    def __init__(self, mode, lang=None):
        super(LemmatizerPymorphy, self).__init__(mode, lang)
        assert not self.lang or self.lang == 'ru', 'Pymorphy supports only russian lang'
        self.morph = pymorphy2.MorphAnalyzer()

    @lru_cache(maxsize=None)
    def get_normal_form(self, word: str) -> str:
        return self.morph.parse(word)[0].normal_form

    def get_normal_form_in_ctx(self, word: str, context: str, word_pos: tp.Tuple[int, int]) -> tp.List[str]:
        raise NotImplementedError


class LemmatizerSpacy(LemmatizerBase):
    lang_models = {
        'en': 'en_core_web_md',
        'es': 'es_core_news_md',
        'de': 'de_core_news_md',
        'sv': 'sv_core_news_md',
        'ru': 'ru_core_news_md',
        'it': 'it_core_news_md',
        'ja': 'ja_core_news_md',
    }

    def __init__(self, mode, lang):
        super(LemmatizerSpacy, self).__init__(mode, lang)
        spacy_model = LemmatizerSpacy.lang_models.get(self.lang, None)
        if spacy_model is None:
            raise NotImplementedError(f"Spacy cannot lemmatize {self.lang} lang")

        self.morph = spacy.load(spacy_model)
        self.tokenizer = self.morph.tokenizer

    @lru_cache(maxsize=None)
    def get_normal_form(self, word: str) -> str:
        doc = self.morph(word)
        return ''.join([t.lemma_ for t in doc])  # that's all about 180°

    @lru_cache(maxsize=10000)
    def get_normal_form_in_ctx(self, word: str, context: str, word_pos: tp.Tuple[int, int]) -> tp.List[str]:
        word = self.remove_punc(word)
        if word == '' or word in string.whitespace:
            return [word]
        left_span, right_span = context[:word_pos[0]], context[word_pos[1]:]
        prefix_tokens = len(self.tokenizer(left_span))
        word_tokens = len(self.tokenizer(word))

        new_ctx = left_span + word + right_span
        doc = self.morph(new_ctx)
        return [word.lemma_ for word in doc[prefix_tokens: prefix_tokens + word_tokens]]


class LemmatizerWordNet(LemmatizerBase):
    def __init__(self, mode, lang):
        super(LemmatizerWordNet, self).__init__(mode, lang)
        assert self.lang == 'en', 'WordNet lemmatizer supports only en lang'
        self.morph = WordNetLemmatizer()

    @lru_cache(maxsize=None)
    def get_normal_form(self, word: str) -> str:
        return self.morph.lemmatize(word)

    def _penn2morphy(self, penntag: str) -> str:
        """ Converts Penn Treebank tags to WordNet. """
        morphy_tag = {'NN': 'n', 'JJ': 'a',
                      'VB': 'v', 'RB': 'r'}
        try:
            return morphy_tag[penntag[:2]]
        except:
            return 'n'

    @lru_cache(maxsize=10000)
    def get_normal_form_in_ctx(self, word: str, context: str, word_pos: tp.Tuple[int, int]) -> tp.List[str]:
        word = self.remove_punc(word)
        if word == '' or word in string.whitespace:
            return [word]
        left_span, right_span = context[:word_pos[0]], context[word_pos[1]:]
        prefix_tokens = len(word_tokenize(left_span))
        word_tokens = len(word_tokenize(word))

        new_ctx = left_span + word + right_span
        lemmatized_ctx = [
            self.morph.lemmatize(word.lower(), pos=self._penn2morphy(tag)) for word, tag in
            pos_tag(word_tokenize(new_ctx))
        ]
        return lemmatized_ctx[prefix_tokens: prefix_tokens + word_tokens]


class LemmatizerNone(LemmatizerBase):
    def __init__(self, mode, lang):
        super(LemmatizerNone, self).__init__(mode, lang)

    @lru_cache(maxsize=None)
    def get_normal_form(self, word: str) -> str:
        return word

    @lru_cache(maxsize=10000)
    def get_normal_form_in_ctx(self, word: str, context: str, word_pos: tp.Tuple[int, int]) -> tp.List[str]:
        raise NotImplementedError()


class LemmatizerSnowballStemmer(LemmatizerBase):
    def __init__(self, mode, lang):
        super(LemmatizerSnowballStemmer, self).__init__(mode, lang)
        language_mapping = {
            'en': "english",
            'ru': 'russian',
            'de': 'german',
            'it': 'italian'
        }
        self.morph = SnowballStemmer(language=language_mapping[lang])

    @lru_cache(maxsize=None)
    def get_normal_form(self, word: str) -> str:
        return self.morph.stem(word)

    @lru_cache(maxsize=10000)
    def get_normal_form_in_ctx(self, word: str, context: str, word_pos: tp.Tuple[int, int]) -> tp.List[str]:
        raise NotImplementedError()

class LemmatizerStanza(LemmatizerBase):
    def __init__(self, mode, lang):
        super(LemmatizerStanza, self).__init__(mode, lang)
        stanza.download(lang)
        self.tokenizer = stanza.Pipeline(lang=lang, processors='tokenize')
        try:
            self.morph = stanza.Pipeline(lang=lang, processors='tokenize,mwt,pos,lemma')
        except:
            self.morph = stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma')

    @lru_cache(maxsize=None)
    def get_normal_form(self, word: str) -> str:
        doc = self.morph(word)
        for sent in doc.sentences:
            for tok in sent.words:
                if tok.lemma:
                    return tok.lemma
                else:
                    return word

    @lru_cache(maxsize=10000)
    def get_normal_form_in_ctx(self, word: str, context: str, word_pos: tp.Tuple[int, int]) -> tp.List[str]:
        word = self.remove_punc(word)
        if word == '' or word in string.whitespace:
            return [word]
        left_span, right_span = context[:word_pos[0]], context[word_pos[1]:]
        prefix_tokens = self.tokenizer(left_span)._num_tokens
        word_tokens = self.tokenizer(word)._num_tokens

        new_ctx = left_span + word + right_span
        doc = self.morph(new_ctx)
        lemmatized_ctx = [tok.lemma for sent in doc.sentences for tok in sent.words]
        return lemmatized_ctx[prefix_tokens: prefix_tokens + word_tokens]
