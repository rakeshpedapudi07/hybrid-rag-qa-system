import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()


class PineconeClient:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX", "rag-index")

        self.pc = Pinecone(api_key=api_key)

        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )

        self.index = self.pc.Index(index_name)

    def upsert_embeddings(self, embeddings, texts):
        vectors = [
            {
                "id": str(i),
                "values": embedding.tolist(),
                "metadata": {"text": texts[i]},
            }
            for i, embedding in enumerate(embeddings)
        ]

        self.index.upsert(vectors=vectors)

    def query(self, embedding, top_k=5):
        results = self.index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
        )

        output = []
        for match in results["matches"]:
            output.append({
                "text": match["metadata"]["text"],
                "score": match["score"]
            })

        return output