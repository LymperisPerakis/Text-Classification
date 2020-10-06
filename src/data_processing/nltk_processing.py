import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


class NltkProcessor(object):
    """A class that uses the NLTK library to process text files
    """
    def __init__(self, document: str, language='english'):
        """
        Initialization of class

        :param document: the documnet to be processed
        :param language: the language of the document
        """
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words(language))
        self.document = document
        self.tokenizer = RegexpTokenizer(r'[\w/]+')

    def is_stopword(self, word: str, with_stopwords: bool = True,
                    with_punctuation: bool = True,
                    with_length: bool = True, length: int = 3) -> bool:
        """
        Checks if the word is a stop word or not

        :param word: the word to be checked
        :param with_stopwords: whether or not to check on the list of stopwords
        :param with_punctuation: whether or not to remove punctuations
        :param with_length:
        :param length: the length of the words to remove
        :return: if the words is a stopword or not
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
        Checks if the word contains a digit

        :param word: the word to be checked
        :return: if the word contains a digit or not
        """
        if all(c.isdigit() for c in word):
            return True
        return False

    @staticmethod
    def is_on_vocab(word: str) -> bool:
        """
        Checks if the word is on the English vocabulary or not

        :param word: the word to be checked
        :return: if the word is on the English vocabulary or not
        """
        if wordnet.synsets(word):
            return True
        return False

    def lemmatize(self, word: str) -> str:
        """
        Lemmatizes the word

        :param word: the word to be lemmatized
        :return: the lemmatized word
        """
        lemmatized_word = self.lemmatizer.lemmatize(word)
        return lemmatized_word

    def stemming(self, word: str) -> str:
        """
        Stems the word

        :param word: the word to be stemmed
        :return: the stemmed word
        """
        stemmed_word = self.stemmer.stem(word)
        return stemmed_word

    def tokenize(self):
        """
        Tokenized the document

        :return: the tokens of the document
        """
        tokens = self.tokenizer.tokenize(self.document)
        return tokens
