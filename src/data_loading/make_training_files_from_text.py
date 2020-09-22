from typing import List, Tuple
import os
from src.configs.configs import text_path_interface
from src.data_loading.make_training_files_from_pdf import MakeTrainingFiles


class MakeTrainingFilesFromText(MakeTrainingFiles):

    def __init__(self, text_path: str = text_path_interface):
        self.text_path = text_path

    def find_documents_list(self) -> Tuple[List[str], List[str]]:
        documents_list = []
        label_list = []
        for root, dirs, files in os.walk(self.text_path):
            for name in files:
                documents_list.append(f'{root}/{name}')
                label_list.append(root.split('/')[-1])
        return documents_list, label_list

    @staticmethod
    def make_documents(documents_list: List[str], text_path: str) -> List[str]:
        docs = []
        for document in documents_list:
            with open(document, 'r', encoding="utf8") as f:
                pdf = f.read()
            docs.append(pdf)
        return docs
