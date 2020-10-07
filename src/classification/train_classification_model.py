import numpy as np
import os

import fasttext
import ktrain
from ktrain import text
import pickle
from transformers import TFAutoModelForSequenceClassification

from src.classification.classification import Classifier
from src.configs.configs import text_path_interface, conventional_models, \
    transformers_models
from src.data_loading.make_training_files_from_text import \
    MakeTrainingFilesFromText
from src.configs.ml_mapping import ml_models_mapping


class TrainClassificationModel:
    """
    A class to train various ML models
    """

    def __init__(self, model: str = 'SVM',
                 text_path: str = text_path_interface,
                 process: bool = True, lem_or_stem: str = 'lem',
                 on_vocab: bool = True, preprocessing_tool: str = 'NLTK',
                 ):
        self.text_path = text_path
        self.training_files = MakeTrainingFilesFromText(self.text_path)
        self.model = model
        self.process = process
        self.lem_or_stem = lem_or_stem
        self.on_vocab = on_vocab
        self.preprocessing_tool = preprocessing_tool
        self.documents_list = None
        self.label_list = None
        self.label_dummy = None
        self.labels = None
        self.documents = None
        self.clf = None
        self.classifier = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.trn = None
        self.val = None
        self.transformer_model = None
        self.learner = None
        self.fasttext_training_data = None
        self.fasttext_model = None
        self.accuracy, self.report, self.confusion_matrix = None, None, None
        self.transformers_output = None
        self.predicted = None
        self.precision, self.recall, self.f1 = None, None, None
        self.balanced_accuracy = None

    def load_files(self,
                   file_path: str =
                   'training_data/cleaned_docs_25_classes.pickle'):
        with open(file_path, 'rb') as f:
            self.documents_list, self.label_list, self.labels, \
            self.label_dummy, self.documents = pickle.load(
                f)
        return self.documents_list, self.label_list, self.labels, \
               self.label_dummy, self.documents

    def preprocess_files(self):
        self.documents_list, self.label_list, self.labels, self.label_dummy, \
        self.documents \
            = self.training_files.run(self.lem_or_stem, self.process,
                                      self.on_vocab)
        return self.documents_list, self.label_list, self.labels, \
               self.label_dummy, self.documents

    def save_training_files(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump([self.documents_list, self.label_list, self.labels,
                         self.label_dummy, self.documents], f)

    def k_fold_cross_validation(self, n_splits: int = 5, shuffle: bool = True,
                                random_state: int = 0):
        self.classifier = Classifier(docs=self.documents,
                                     label_dummy=self.label_dummy)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.classifier.k_fold_cross_validation(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def split_training_files(self, test_size: float = 0.2,
                             random_state: int = 0):
        self.classifier = Classifier(docs=self.documents,
                                     label_dummy=self.label_dummy)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.classifier.split_set(
                test_size=test_size,
                random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, **kwargs):
        if self.model in conventional_models:
            self.clf = self.classifier.train_model(
                ml_models_mapping[self.model], self.X_train, self.y_train)
            return self.clf
        elif self.model in transformers_models:
            t = text.Transformer(model_name=self.model, maxlen=512,
                                 class_names=self.labels)
            self.trn = t.preprocess_train(self.X_train, self.y_train)
            self.val = t.preprocess_test(self.X_test, self.y_test)
            self.transformer_model = t.get_classifier()
            if kwargs:
                self.learner = ktrain.get_learner(self.transformer_model,
                                                  train_data=self.trn,
                                                  val_data=self.val,
                                                  batch_size=kwargs[
                                                      'batch_size'])
                self.learner.fit_onecycle(kwargs['lr'], kwargs['epochs'])
            return self.learner
        elif self.model == 'fasttext':
            self.fasttext_training_data = [
                f'__label__{self.label_dummy[i]} {self.documents[i]}'
                for i in range(len(self.documents))]
            if kwargs:
                self.classifier = Classifier(docs=self.fasttext_training_data,
                                             label_dummy=self.label_dummy)
                self.X_train, self.X_test, self.y_train, self.y_test = \
                    self.classifier.split_set(test_size=kwargs['test_size'],
                                              random_state=kwargs[
                                                  'random_state'])
                if os.path.isfile(
                        'training_data/fasttext_training_data/fasttext_train'
                        '.txt'):
                    os.remove(
                        'training_data/fasttext_training_data/fasttext_train'
                        '.txt')
                if os.path.isfile(
                        'training_data/fasttext_training_data/fasttext_test'
                        '.txt'):
                    os.remove(
                        'training_data/fasttext_training_data/fasttext_test'
                        '.txt')
                with open(
                        'training_data/fasttext_training_data/fasttext_train'
                        '.txt',
                        'w', encoding='utf-8') as filehandle:
                    filehandle.writelines(
                        "%s\n" % document for document in self.X_train)
                with open(
                        'training_data/fasttext_training_data/fasttext_test'
                        '.txt',
                        'w', encoding='utf-8') as filehandle:
                    filehandle.writelines(
                        "%s\n" % document for document in self.X_test)
                self.fasttext_model = fasttext.train_supervised(
                    'training_data/fasttext_training_data/fasttext_train.txt',
                    epoch=kwargs['epoch'], lr=kwargs['lr'],
                    wordNgrams=kwargs['wordNgrams'])
                return self.fasttext_model

    def validate_model(self):
        if self.model in conventional_models:
            self.accuracy, self.report, self.confusion_matrix, \
            self.predicted, self.precision, self.recall, self.f1, \
            self.balanced_accuracy \
                = self.classifier.test_model(self.clf,
                                             self.X_test,
                                             self.y_test,
                                             self.labels)
            return self.accuracy, self.report, self.confusion_matrix, \
                   self.predicted, self.precision, self.recall, self.f1, \
                   self.balanced_accuracy
        elif self.model in transformers_models:
            pred = self.learner.predict(self.val)
            self.predicted = np.argmax(pred, axis=1)
            self.accuracy, self.report, self.confusion_matrix, \
            self.precision, self.recall, self.f1, self.balanced_accuracy = \
                self.classifier.get_metrics(self.y_test, self.predicted,
                                            self.labels)
            return self.accuracy, self.report, self.confusion_matrix, \
                   self.predicted, self.precision, \
                   self.recall, self.f1, self.balanced_accuracy
        elif self.model == 'fasttext':
            pred = self.fasttext_model.predict(
                [x.strip('__label__ ') for x in self.X_test])
            self.predicted = np.array(
                [int(p[0].split('__label__')[1]) for p in pred[0]])
            self.accuracy, self.report, self.confusion_matrix, \
            self.precision, self.recall, self.f1, self.balanced_accuracy = \
                self.classifier.get_metrics(self.y_test, self.predicted,
                                            self.labels)
            return self.accuracy, self.report, self.confusion_matrix, \
                   self.predicted, self.precision, \
                   self.recall, self.f1, self.balanced_accuracy

    def save_transformers_training_files(self):
        with open(
                f'training_data/transformers_training_data/'
                f'{self.model}.pickle',
                'wb') as f:
            pickle.dump([self.trn, self.val], f)

    def load_transformers_training_files(self):
        with open(
                f'training_data/transformers_training_data/'
                f'{self.model}.pickle',
                'wb') as f:
            pickle.dump([self.trn, self.val], f)

    def save_trained_model(self):
        if self.model in conventional_models:
            with open(f'training_data/conventional_models/{self.model}.pickle',
                      'wb') as f:
                pickle.dump(self.clf, f)
        elif self.model in transformers_models:
            self.learner.model.save_pretrained(
                f'training_data/transformers_model/{self.model}')
        elif self.model == 'fasttext':
            self.fasttext_model.save_model(
                'training_data/fasttext_model/fasttext.bin')

    def load_trained_model(self):
        if self.model in conventional_models:
            with open(f'training_data/conventional_models/{self.model}.pickle',
                      'wb') as f:
                self.clf = pickle.load(f)
        elif self.model in transformers_models:
            self.learner = \
                TFAutoModelForSequenceClassification.from_pretrained(
                    'training_data/transformers_model/{self.model}')
        elif self.model == 'fasttext':
            self.fasttext_model = fasttext.load_model(
                'training_data/fasttext_model/fasttext.bin')
