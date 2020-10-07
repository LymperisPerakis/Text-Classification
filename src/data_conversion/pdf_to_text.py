import os
from pathlib import Path
from src.configs.configs import text_path_interface


class PdfToText(object):
    """
    A class to convert a pdf to a text file
    """
    def __init__(self, pdf_path: str,
                 text_folder: str = text_path_interface):
        """
        Initialization of the PdfToText class

        :param pdf_path: the path of the pdf
        :param text_folder: the folder to save the exported text
        """
        self.pdf_path = pdf_path.replace('\\', '/')
        self.pdf_category = self.pdf_path.split('/')[-2]
        self.text_name = self.pdf_path.split('/')[-1].split('.pdf')[
                             0] + '.text'
        self.text_folder = text_folder.replace('\\', '/')
        self.destination = f'{self.text_folder}/{self.pdf_category}/' \
                           f'{self.text_name}'

    def read_from_text(self) -> str:
        """
        Reads the text from a file

        :return: the text of the file
        """
        with open(self.destination, 'r', encoding='utf8') as f:
            pdf = f.read()
        return pdf

    @property
    def convert_to_text(self) -> str:
        """
        Converts the pdf to a text

        :return: the text of the pdf
        """
        import pdftotext
        with open(self.pdf_path, "rb") as f:
            pdf = pdftotext.PDF(f)
            pdf = "\n\n".join(pdf)
        return pdf

    def save_text_to_file(self, pdf):
        """
        Saves the text of the pdf to the text folder

        :param pdf: the text of the pdf
        """
        Path(f'{self.text_folder}/{self.pdf_category}').mkdir(parents=True,
                                                              exist_ok=True)
        with open(self.destination, 'w') as f:
            f.write(pdf)

    def get_text(self):
        """
        Checks if the pdf is already converted to text and if not it converts
        it and returns it

        :return: the text of the pdf
        """
        if os.path.isfile(self.destination):
            pdf = self.read_from_text()
        else:
            pdf = self.convert_to_text
            self.save_text_to_file(pdf)
        return pdf

    def run(self):
        """
        Runs the class

        :return: the text of the pdf
        """
        pdf = self.get_text()
        return pdf
