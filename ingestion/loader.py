import os
from typing import List
from PyPDF2 import PdfReader


class DocumentLoader:
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path

    def load_documents(self) -> List[str]:
        documents = []

        for file_name in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file_name)

            # Load TXT files
            if file_name.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        documents.append(text)

            # Load PDF files
            elif file_name.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                text = text.strip()
                if text:
                    documents.append(text)

        return documents