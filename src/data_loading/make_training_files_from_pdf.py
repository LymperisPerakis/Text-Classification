from typing import List, Tuple
import os
from src.data_processing.process_document_nltk import PreProcessing
from src.configs.configs import interface_pdf_path2, interface_text_path2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.data_conversion.pdf_to_text import PdfToText


class MakeTrainingFiles(object):
    def __init__(self, pdf_path: str = interface_pdf_path2,
                 text_path: str = interface_text_path2):
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
    def make_documents(documents_list: List[str], text_path: str) -> List[str]:
        docs = []
        for document in documents_list:
            pdf2text = PdfToText(document, text_path)
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
    def process_documents(docs: List[str],
                          lem_or_stem: str = 'lem') -> List[str]:
        for i, document in enumerate(docs):
            preprocessor = PreProcessing(document)
            docs[i] = preprocessor.run(lem_or_stem)
        return docs

    def run(self, lem_or_stem: str = 'lem', process: bool = True
            ) -> Tuple[List[str], List[str], list, List[int], List[str]]:
        documents_list, label_list = self.find_documents_list()
        labels, label_dummy = self.make_label_dict(label_list)
        unprocessed = self.make_documents(documents_list, self.text_path)
        if process:
            documents = self.process_documents(unprocessed, lem_or_stem)
        else:
            documents = unprocessed
        return documents_list, label_list, labels, label_dummy, documents
