import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


class NltkProcessor(object):

    def __init__(self, document: str, language='english'):
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words(language))
        self.document = document
        self.tokenizer = RegexpTokenizer(r'[\w/]+')

    def is_stopword(self, word: str, with_stopwords=True,
                    with_punctuation=True,
                    with_length=True, length=3) -> bool:
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

    @staticmethod
    def is_on_vocab(word: str) -> bool:
        if wordnet.synsets(word):
            return True
        return False

    def lemmatize(self, word: str) -> str:
        lemmatized_word = self.lemmatizer.lemmatize(word)
        return lemmatized_word

    def stemming(self, word: str) -> str:
        stemmed_word = self.stemmer.stem(word)
        return stemmed_word

    def tokenize(self):
        tokens = self.tokenizer.tokenize(self.document)
        return tokens
