from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from typing import List, Tuple
from sklearn import tree


class Classifier(object):
    """ Classifier is an object that trains a machine learning model
     to classify documents

    Parameters
    ----------

    docs : List[str]
        The documents to be used for training and testing

    label_dummy: List[str]
        The dummy labels of the documents

    Methods
    --------

    split_set(self, test_size: float = 0.2, random_state: int = 0)
        Splits the documents into training and testing set

    train_model_Bayes(self, X_train: List[str], y_train: List[int])
        Trains a Naive Bayes classifier given the training set

    train_model_SVM(self, X_train: List[str], y_train: List[int])
        Trains an SVM classifier given the training set

    train_model(self, model, X_train: List[str], y_train: List[int])
        Trains the given model with the given training set

    test_model(self, text_clf, xTest, yTest, label_dict)
        Testes the trained model given the test set and returns metrics
    """

    def __init__(self, docs: List[str], label_dummy: List[int]):
        self.docs = docs
        self.label_dummy = label_dummy

    def split_set(self, test_size: float = 0.2, random_state: int = 0) \
            -> Tuple[List[str], List[str], List[int], List[int]]:
        X_train, X_test, y_train, y_test = train_test_split(
            self.docs, self.label_dummy, test_size=test_size,
            random_state=random_state)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def train_model(model, X_train: List[str], y_train: List[int]):
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
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

    def export_decision_tree_graph(self, X_train: List[str], y_train: List[int],
                                   class_names: List[str], output_file: str = 'tree.dot'):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X_train)
        feature_names = vectorizer.get_feature_names()
        clf = DecisionTreeClassifier().fit(X, y_train)
        dot_data = tree.export_graphviz(clf, out_file=output_file,
                                        feature_names=feature_names,
                                        class_names=class_names,
                                        filled=True, rounded=True, special_characters=True)
        return dot_data, f'Decision tree graph saved in file {output_file}'

# TODO: prettify https://scikit-learn.org/stable/modules/tree.html
# TODO: sklearn.ensemble.BaggingClassifier
# TODO: check ripper from wittgenstein
    @staticmethod
    def get_metrics(y_test, predicted, labels):
        accuracy = np.mean(predicted == y_test)
        report = metrics.classification_report(y_test, predicted,
                                               target_names=labels)
        confusion_matrix = metrics.confusion_matrix(y_test, predicted)
        return accuracy, report, confusion_matrix

    def test_model(self, text_clf, X_test: List[str], y_test: List[int],
                   labels: list):
        predicted = text_clf.predict(X_test)
        accuracy, report, confusion_matrix = self.get_metrics(y_test,
                                                              predicted,
                                                              labels)
        # predicted_prob = text_clf.predict_proba(X_test)
        # auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovr")
        return accuracy, report, confusion_matrix
