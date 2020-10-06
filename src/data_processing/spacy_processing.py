import string
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from typing import List


class SpacyProcessor(object):
    """
    This class uses the spaCy library to process a document
    """
    def __init__(self, document: str):
        self.nlp = English()
        self.nlp.max_length = 3500000
        self.document = document
        self.nlp_document = self.nlp(self.document)
        self.stop_words = STOP_WORDS
        self.punctuations = string.punctuation

    def spacy_tokenizer(self):
        mytokens = [
            word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else
            word.lower_
            for word in self.nlp_document]
        mytokens = [word for word in mytokens if
                    word not in self.stop_words and
                    word not in self.punctuations]

        return mytokens

    def is_stopword(self, word: str, with_stopwords=True,
                    with_punctuation=True,
                    with_length=True, length=2) -> bool:
        if with_stopwords and word in self.stop_words:
            return True
        if with_punctuation and word in string.punctuation:
            return True
        if with_length and length and len(word) < length:
            return True
        return False

    @staticmethod
    def is_digit(word: str) -> bool:
        if all(c.isdigit() for c in word):
            return True
        return False

    def normalized(self):
        tokens = [word.lemma_.lower().strip() for word in self.nlp_document
                  if not word.is_punct and not word.is_stop]
        return tokens

    def tokenize(self) -> List[str]:
        tokens = [token.text for token in self.nlp_document]
        return tokens
