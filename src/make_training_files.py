from typing import List, Tuple
import os
from src.process_document import PreProcessing
from src.configs import interface_pdf_path, interface_text_path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.pdf_to_text import PdfToText


class MakeTrainingFiles(object):
    """ Pdf2text is an object that extracts text from pdfs

    Parameters
    ----------

    pdf_path: str
        The path of the pdf documents

    text_path: str
        The path of the extracted text

    Methods
    ----------

    find_documents_list(self) -> Tuple[List[str], List[str], dict]
        Makes a list of documents and labels from the given path

    make_label_dict(self, label_list: List[str]) -> Tuple[dict, List[int]]
        Makes a label dictionary and and label dummies in order to be used
        by the model

    make_documents(self, documents_list, inf):
        Returns a list of the processed documents

    process_pdf(self, documents_list, inf) -> List[str]:
        Processes the documents of the documents list

    load_documents(self, inf) -> List[str]
        Loads the preprocessed documents


    """

    def __init__(self, pdf_path: str = interface_pdf_path,
                 text_path: str = interface_text_path):
        self.pdf_path = pdf_path
        self.text_path = text_path

    def find_documents_list(self) -> Tuple[List[str], List[str]]:
        documents_list = []
        label_list = []
        for root, dirs, files in os.walk(self.pdf_path):
            for name in files:
                documents_list.append(f'{root}/{name}')
                label_list.append(root.split('/')[-1])
        return documents_list, label_list

    @staticmethod
    def make_documents(documents_list: List[str]) -> List[str]:
        docs = []
        for i, document in enumerate(documents_list):
            pdf2text = PdfToText(document)
            pdf = pdf2text.run()
            docs.append(pdf)
        return docs

    @staticmethod
    def make_label_dict(label_list: List[str]) -> Tuple[list, List[int]]:
        le = LabelEncoder()
        df = np.asarray(label_list)
        fitted = le.fit(df)
        label_dummy = list(le.transform(df))
        labels = list(fitted.classes_)
        return labels, label_dummy

    @staticmethod
    def process_documents(docs: List[str]) -> List[str]:
        for i, document in enumerate(docs):
            preprocessor = PreProcessing(document)
            docs[i] = preprocessor.run('stem')
        return docs

    def get_unprocessed(self) -> Tuple[List[str], List[str],
                                       list, List[int], List[str]]:
        documents_list, label_list = self.find_documents_list()
        labels, label_dummy = self.make_label_dict(label_list)
        documents = self.make_documents(documents_list)
        return documents_list, label_list, labels, label_dummy, documents

    def run(self) -> Tuple[List[str], List[str], list, List[int], List[str]]:
        documents_list, label_list = self.find_documents_list()
        labels, label_dummy = self.make_label_dict(label_list)
        unprocessed = self.make_documents(documents_list)
        documents = self.process_documents(unprocessed)
        return documents_list, label_list, labels, label_dummy, documents
