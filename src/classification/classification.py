from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from typing import List, Tuple
from sklearn import tree
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


class Classifier(object):
    """ A class for splitting the data into training and test set, training a
    conventional machine learning model and validating its performance
    """

    def __init__(self, docs: List[str], label_dummy: List[int]):
        """:key"""
        self.docs = docs
        self.label_dummy = label_dummy

    def k_fold_cross_validation(self, n_splits: int = 5, shuffle: bool = True,
                                random_state: int = 0):
        """
        Creates a stratified k fold cross validation set

        :param n_splits: number of splits
        :param shuffle: if we want to shuffle the data
        :param random_state: which random state we want
        :return: the training and test sets
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                              random_state=random_state)
        X_train, y_train, X_test, y_test = [], [], [], []
        for train_index, test_index in skf.split(self.docs, self.label_dummy):
            X_train.append([self.docs[x] for x in train_index])
            y_train.append([self.label_dummy[x] for x in train_index])
            X_test.append([self.docs[x] for x in test_index])
            y_test.append([self.label_dummy[x] for x in test_index])
        return X_train, X_test, y_train, y_test

    def split_set(self, test_size: float = 0.2, random_state: int = 0) \
            -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        Splits the data into training and test set

        :param test_size: the percentage of the test set
        :param random_state: which random state we want
        :return: the training and test set
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.docs, self.label_dummy, test_size=test_size,
            random_state=random_state, stratify=self.label_dummy)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def train_model(model, X_train: List[str], y_train: List[int]):
        """
        Trains the specified model using the TFIDF method as feature extraction

        :param model: the model to be trained
        :param X_train: the training set
        :param y_train: the labels of the training set
        :return: the fitted model
        """
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', model),
        ])
        text_clf.fit(X_train, y_train)
        return text_clf

    @staticmethod
    def train_model_dim_reduction(model, X_train: List[str],
                                  y_train: List[int]):
        """
        Trains the model after applying a dimensionality reduction

        :param model: the model to be trained
        :param X_train: the training set
        :param y_train: the labels of the training set
        :return: the fitted model
        """
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            (
                'svd',
                TruncatedSVD(n_components=1000, n_iter=20, random_state=0)),
            ('clf', model),
        ])
        text_clf.fit(X_train, y_train)
        return text_clf

    def train_model_SVM(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(SGDClassifier(loss='hinge', penalty='l2',
                                                  alpha=1e-3, random_state=42,
                                                  max_iter=5, tol=None),
                                    X_train, y_train)
        return text_clf

    def train_model_SVC(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(SVC(), X_train, y_train)
        return text_clf

    def train_model_Bayes(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(MultinomialNB(), X_train, y_train)
        return text_clf

    def train_model_KNClassifier(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(KNeighborsClassifier(), X_train, y_train)
        return text_clf

    def train_model_RandomForest(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(RandomForestClassifier(), X_train, y_train)
        return text_clf

    def train_model_LDA(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(LinearDiscriminantAnalysis(),
                                    X_train, y_train)
        return text_clf

    def train_model_LogR(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(
            LogisticRegression(multi_class='multinomial'), X_train, y_train)
        return text_clf

    def train_model_AdaB(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(
            AdaBoostClassifier(), X_train, y_train)
        return text_clf

    def train_model_Rocchio(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(
            NearestCentroid(), X_train, y_train)
        return text_clf

    def train_model_DecisionTree(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(
            DecisionTreeClassifier(), X_train, y_train)
        return text_clf

    def train_model_Boosting(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(
            GradientBoostingClassifier(), X_train, y_train)
        return text_clf

    def train_model_Voting(self, X_train: List[str], y_train: List[int]):
        text_clf = self.train_model(
            VotingClassifier(), X_train, y_train)
        return text_clf

    @staticmethod
    def export_decision_tree_graph(X_train: List[str], y_train: List[int],
                                   class_names: List[str],
                                   output_file: str = 'tree.dot'):
        """
        Trains a decision tree, exports its graph and saves it

        :param X_train: the training set
        :param y_train: the labels of the training set
        :param class_names: the names of the classes
        :param output_file: the path of the output file
        :return: the graph
        """
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X_train)
        feature_names = vectorizer.get_feature_names()
        clf = DecisionTreeClassifier().fit(X, y_train)
        dot_data = tree.export_graphviz(clf, out_file=output_file,
                                        feature_names=feature_names,
                                        class_names=class_names,
                                        filled=True, rounded=True,
                                        special_characters=True)
        return dot_data, f'Decision tree graph saved in file {output_file}'

    @staticmethod
    def get_metrics(y_test, predicted, labels):
        """
        Validates the performance of the classifier using metrics

        :param y_test: the labels of the data
        :param predicted: the predicted classes of the model
        :param labels: the classes of the data
        :return: the metrics to validate the classifiers' performance
        """
        accuracy = np.mean(predicted == y_test)
        report = metrics.classification_report(y_test, predicted,
                                               target_names=labels)
        confusion_matrix = metrics.confusion_matrix(y_test, predicted)
        precision = metrics.precision_score(y_test, predicted, average='macro')
        recall = metrics.recall_score(y_test, predicted, average='macro')
        f1 = metrics.f1_score(y_test, predicted, average='weighted')
        balanced_accuracy = balanced_accuracy_score(y_test, predicted)

        return accuracy, report, confusion_matrix, precision, recall, f1, \
               balanced_accuracy

    def test_model(self, text_clf, X_test: List[str], y_test: List[int],
                   labels: list):
        """
        Tests the model and validates its performance

        :param text_clf: the model to be tested
        :param X_test: the testing set
        :param y_test: the labels of the testing set
        :param labels: the classes of the data
        :return: the metrics to validate the classifiers' performance
        """
        predicted = text_clf.predict(X_test)
        accuracy, report, confusion_matrix, precision, recall, f1, \
        balanced_accuracy = self.get_metrics(y_test, predicted, labels)
        return accuracy, report, confusion_matrix, predicted, precision, \
               recall, f1, balanced_accuracy
