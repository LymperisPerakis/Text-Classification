import pdftotext
import os
from pathlib import Path


class PdfToText(object):
    def __init__(self, pdf_path: str,
                 text_folder: str = './Data/Interface Ics text'):
        self.pdf_path = pdf_path
        self.pdf_category = pdf_path.split('/')[-2]
        self.text_name = self.pdf_path.split('/')[-1].split('.pdf')[
                             0] + '.text'
        self.text_folder = text_folder
        self.destination = f'{self.text_folder}/{self.pdf_category}/' \
                           f'{self.text_name}'

    def read_from_text(self) -> str:
        with open(self.destination, 'r') as f:
            pdf = f.read()
        return pdf

    @property
    def convert_to_text(self) -> str:
        with open(self.pdf_path, "rb") as f:
            pdf = pdftotext.PDF(f)
            pdf = "\n\n".join(pdf)
        return pdf

    def save_text_to_file(self, pdf):
        Path(f'{self.text_folder}/{self.pdf_category}').mkdir(parents=True,
                                                              exist_ok=True)
        with open(self.destination, 'w') as f:
            f.write(pdf)

    def get_text(self):
        if os.path.isfile(self.destination):
            pdf = self.read_from_text()
        else:
            pdf = self.convert_to_text
            self.save_text_to_file(pdf)
        return pdf

    def run(self):
        pdf = self.get_text()
        return pdf
