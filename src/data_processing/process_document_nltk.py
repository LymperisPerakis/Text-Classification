import re
from src.data_processing.nltk_processing import NltkProcessor
import string


class PreProcessing:
    def __init__(self, document):
        self.document = document
        self.processor = NltkProcessor(document)
        self.tokens = self.processor.tokenize()
        self.r = re.compile(r'\d.*[a-zA-Z]|[a-zA-Z].*\d')

    def is_stopword(self, word: str) -> bool:
        return True if self.processor.is_stopword(word) else False

    def is_digit(self, word: str) -> bool:
        return True if self.processor.is_digit(word) else False

    def is_number(self, word: str) -> bool:
        return True if self.r.match(word) else False

    def lemmatize_word(self, word: str) -> str:
        lemmatized_word = self.processor.lemmatize(word)
        return lemmatized_word

    def stem_word(self, word: str) -> str:
        stemmed_word = self.processor.stemming(word)
        return stemmed_word

    def is_on_vocab(self, word: str) -> bool:
        return self.processor.is_on_vocab(word)

    @staticmethod
    def has_numbers(word: str) -> bool:
        return any(char.isdigit() for char in word)

    @staticmethod
    def has_punctuation(word: str) -> bool:
        return any(char in string.punctuation for char in word)

    def is_candidate(self, word: str) -> bool:
        is_candidate = all([
            not self.is_stopword(word),
            not self.is_digit(word),
            not self.is_number(word),
            not self.has_numbers(word),
            not self.has_punctuation(word),
            # self.is_on_vocab(word)
        ])
        return is_candidate

    def process(self, word: str, normalization: str) -> str:
        word = word.lower()
        candidate = self.is_candidate(word)
        normalized_word = None
        if candidate:
            if normalization == 'stem':
                normalized_word = self.stem_word(word)
            elif normalization == 'lem':
                normalized_word = self.lemmatize_word(word)
            else:
                normalized_word = word
        return normalized_word

    def run(self, normalization: str = 'lem') -> str:
        processed_document = []
        for word in self.tokens:
            normalized_word = self.process(word, normalization)
            if normalized_word:
                processed_document.append(normalized_word)
        return ' '.join(processed_document)
