import json
import numpy as np
from ingestion.embedder import Embedder
from retriever.pinecone_client import PineconeClient
from rank_bm25 import BM25Okapi


class HybridSearcher:
    def __init__(self, mode="hybrid"):
        self.mode = mode
        self.embedder = Embedder()
        self.pinecone = PineconeClient()

        with open("data/indexed_chunks.json", "r") as f:
            self.texts = json.load(f)

        tokenized_corpus = [text.split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_search(self, query, top_k=5):
        query_embedding = self.embedder.embed_texts([query])[0]
        return self.pinecone.query(query_embedding, top_k)

    def bm25_search(self, query, top_k=5):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked:
            results.append({
                "text": self.texts[idx],
                "score": scores[idx]
            })

        return results

    def search(self, query, top_k=3):

        if self.mode == "dense":
            results = self.dense_search(query, top_k)
            return [r["text"] for r in results]

        if self.mode == "bm25":
            results = self.bm25_search(query, top_k)
            return [r["text"] for r in results]

        # HYBRID WEIGHTED FUSION
        dense_results = self.dense_search(query, top_k=10)
        bm25_results = self.bm25_search(query, top_k=10)

        alpha = 0.6

        score_dict = {}

        # Normalize dense scores
        dense_scores = np.array([r["score"] for r in dense_results])
        if len(dense_scores) > 0:
            dense_scores = dense_scores / (dense_scores.max() + 1e-8)

        for i, r in enumerate(dense_results):
            score_dict[r["text"]] = alpha * dense_scores[i]

        # Normalize BM25 scores
        bm25_scores = np.array([r["score"] for r in bm25_results])
        if len(bm25_scores) > 0:
            bm25_scores = bm25_scores / (bm25_scores.max() + 1e-8)

        for i, r in enumerate(bm25_results):
            if r["text"] in score_dict:
                score_dict[r["text"]] += (1 - alpha) * bm25_scores[i]
            else:
                score_dict[r["text"]] = (1 - alpha) * bm25_scores[i]

        # Sort by final score
        ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

        return [text for text, _ in ranked[:top_k]]