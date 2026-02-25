from ingestion.loader import DocumentLoader
from ingestion.chunker import TextChunker
from ingestion.embedder import Embedder
from retriever.pinecone_client import PineconeClient
import json


def run_ingestion():
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents()

    # Chunk documents
    chunker = TextChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk_documents(documents)

    # Generate embeddings
    embedder = Embedder()
    embeddings = embedder.embed_texts(chunks)

    # Upload to Pinecone
    pinecone_client = PineconeClient()
    pinecone_client.upsert_embeddings(embeddings, chunks)

    # Save chunks locally for BM25 usage
    with open("data/indexed_chunks.json", "w") as f:
        json.dump(chunks, f)

    print("Chunks saved locally.")
    print("Ingestion completed successfully!")


if __name__ == "__main__":
    run_ingestion()