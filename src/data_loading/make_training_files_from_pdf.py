from typing import List, Tuple
import os
from src.data_processing.process_document_nltk import PreProcessing
from src.configs.configs import interface_pdf_path2, text_path_interface
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.data_conversion.pdf_to_text import PdfToText


class MakeTrainingFiles(object):
    """
    A class to make training files from a given path
    """
    def __init__(self, pdf_path: str = interface_pdf_path2,
                 text_path: str = text_path_interface):
        self.pdf_path = pdf_path
        self.text_path = text_path

    def find_documents_list(self) -> Tuple[List[str], List[str]]:
        """
        Creates a list with the documents and their labels from a folder

        :return: the list of the documents and their labels
        """
        documents_list = []
        label_list = []
        for root, dirs, files in os.walk(self.pdf_path):
            for name in files:
                root = root.replace('\\', '/')
                documents_list.append(f'{root}/{name}')
                label_list.append(root.split('/')[-1])
        return documents_list, label_list

    @staticmethod
    def make_documents(documents_list: List[str], text_path: str) -> List[str]:
        """
        Creates or reads the text from a list of pdf files

        :param documents_list: the list of the pdf files
        :param text_path: the path of the where the converted text should be
        saved
        :return: the converted text of the documents
        """
        docs = []
        for document in documents_list:
            pdf2text = PdfToText(os.path.abspath(document), text_path)
            pdf = pdf2text.run()
            docs.append(pdf)
        return docs

    @staticmethod
    def make_label_dict(label_list: List[str]) -> Tuple[list, List[int]]:
        """
        Transforms the labels into dummy

        :param label_list: the list of documents' labels
        :return: the classes and the dummy labels
        """
        le = LabelEncoder()
        df = np.asarray(label_list)
        fitted = le.fit(df)
        label_dummy = list(le.transform(df))
        labels = list(fitted.classes_)
        return labels, label_dummy

    @staticmethod
    def process_documents(docs: List[str],
                          lem_or_stem: str = 'lem',
                          on_vocab: bool = True) -> List[str]:
        """
        Processes the text documents

        :param docs: the list of documents
        :param lem_or_stem: the normalization to perform on the text
        :param on_vocab: if we want to exclude the words not on the vocabulary
        :return: the processed text
        """
        for i, document in enumerate(docs):
            preprocessor = PreProcessing(document)
            docs[i] = preprocessor.run(lem_or_stem, on_vocab)
        return docs

    def run(self, lem_or_stem: str = 'lem', process: bool = True,
            on_vocab: bool = True
            ) -> Tuple[List[str], List[str], list, List[int], List[str]]:
        """
        Processes the text files and returns them with their labels

        :param lem_or_stem: the normalization to perform on the text
        :param process: if we want to process the text or not
        :param on_vocab: if we want to exclude the words not on the vocabulary
        :return: the processed text and its labels
        """
        documents_list, label_list = self.find_documents_list()
        labels, label_dummy = self.make_label_dict(label_list)
        unprocessed = self.make_documents(documents_list, self.text_path)
        if process:
            documents = self.process_documents(unprocessed, lem_or_stem,
                                               on_vocab)
        else:
            documents = unprocessed
        return documents_list, label_list, labels, label_dummy, documents
