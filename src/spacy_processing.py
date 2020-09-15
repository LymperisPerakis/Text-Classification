import string
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from typing import List


class SpacyProcessor(object):
    def __init__(self, document: str):
        """

        :param document:
        """
        self.nlp = English()
        self.document = document
        self.nlp_document = self.nlp(self.document)
        self.stop_words = STOP_WORDS
        self.punctuations = string.punctuation

    def spacy_tokenizer(self):
        # Creating our token object, which is used to create documents with
        # linguistic annotations.

        # Lemmatizing each token and converting each token into lowercase
        mytokens = [
            word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else
            word.lower_
            for word in self.nlp_document]

        # Removing stop words
        mytokens = [word for word in mytokens if
                    word not in self.stop_words and
                    word not in self.punctuations]

        # return preprocessed list of tokens
        return mytokens

    def is_stopword(self, word: str, with_stopwords=True,
                    with_punctuation=True,
                    with_length=True, length=2) -> bool:
        """

        :param word:
        :param with_stopwords:
        :param with_punctuation:
        :param with_length:
        :param length:
        :return:
        """
        if with_stopwords and word in self.stop_words:
            return True
        if with_punctuation and word in string.punctuation:
            return True
        if with_length and length and len(word) < length:
            return True
        return False

    @staticmethod
    def is_digit(word: str) -> bool:
        """

        :param word:
        :return: bool
        """
        if all(c.isdigit() for c in word):
            return True
        return False

    def lemmatize(self, word: str) -> str:
        """

        :param word: The word that we want to lemmatize
        :return: The lemmatized word
        """
        lemmatized_word = self.lemmatizer.lemmatize(word)
        return lemmatized_word

    def stemming(self, word: str) -> str:
        """

        :param word:
        :return:
        """
        stemmed_word = self.stemmer.stem(word)
        return stemmed_word

    def tokenize(self) -> List[str]:
        """

        :return:
        """
        tokens = [token.text for token in self.nlp_document]
        return tokens
