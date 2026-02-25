from typing import List


class TextChunker:
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = words[start:end]
            chunks.append(" ".join(chunk))
            start += self.chunk_size - self.overlap

        return chunks

    def chunk_documents(self, documents: List[str]) -> List[str]:
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)
        return all_chunks