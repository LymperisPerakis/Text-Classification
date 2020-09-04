import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


class NltkProcessor(object):
    """NltkProcessor is an object that uses nltk for making an nlp processing
    on a document

    Parameters
    ----------
    language : String, default 'english'
        The language to be used

    document : String
        The document to be processed

    Attributes
    ----------
    lemmatizer: nltk.stem.WordNetLemmatizer
        The Lemmatizer object used for lemmatization

    stop_words: set(nltk.corpus.stopwords)
        A set of a string stopwords

    document: String
        The document that is going to be processed

    tokenizer: RegexpTokenizer(r'\w+')
        The tokenizer from nltk that removes punctuations

    Methods
    ----------

    is_stopword(word: str, with_stopwords=True, with_punctuation=True,
            with_length=True, length=2) -> bool
        Check whether a word is a stopword or not

    is_digit(word: str) -> bool
        Check if a word is a number or not

    is_on_vocab(word: str) -> bool
        Check if a word is on the wordnet vocabulary or not

    lemmatize(word, with_pos=True) -> str
        Lemmatize a word using nltk lemmatizer

    """

    def __init__(self, document: str, language='english'):
        """Initialize the NltkProcessor object by downloading the needed
        libraries
            then initializing the lemmatizer and setting the stop_words

        Parameters
        ----------
        document : String
            The document that should be processed

        language : String, default 'english'
            The language to be used
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

    def is_stopword(self, word: str, with_stopwords=True,
                    with_punctuation=True,
                    with_length=True, length=2) -> bool:
        """Check whether a word is a stopword or not

        Parameters
        ----------
        word : String
            The word to be checked

        with_stopwords : boolean, default True
            Whether to use nltk stopwords or not

        with_punctuation : boolean, default True
            Whether to check if it is string punctuation or not

        with_length : boolean, default True
            Whether to consider the length of the word as a stopword
            Must be used with length parameter

        length : int, default 2
            A length under what a word is considered as a stopword
            Can only be used with 'with_length=True'

        Returns
        ----------
        is_stopword : boolean
            Whether the word is a stopword or not based on the parameters
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
        """Check whether a word is only a number or not

        Parameters
        ----------
        word : String
            The word to be checked

        Returns
        ----------
        is_digit : boolean
        """
        if all(c.isdigit() for c in word):
            return True
        return False

    @staticmethod
    def is_on_vocab(word: str) -> bool:
        """Check whether a word is on the english vocabulary or not

        Parameters
        ----------
        word : String
            The word to be checked

        Returns
        ----------
        is_on_vocab : boolean
        """
        if wordnet.synsets(word):
            return True
        return False

    def lemmatize(self, word: str) -> str:
        """Lemmatize a word using nltk lemmatizer with the option
        of using part-of-speech

        Parameters
        ----------
        word : String
            The word to be lemmatized

        Returns
        ----------
        lemmatized_word : String
            The lemmatized word
        """
        lemmatized_word = self.lemmatizer.lemmatize(word)
        return lemmatized_word

    def stemming(self, word: str) -> str:
        stemmed_word = self.stemmer.stem(word)
        return stemmed_word

    def tokenize(self):
        tokens = self.tokenizer.tokenize(self.document)
        return tokens
