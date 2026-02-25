import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever.hybrid_search import HybridSearcher
from app.services.generator import RAGGenerator

app = FastAPI(
    title="NLP-Driven Intelligent Retrieval & QA System",
    description="RAG-based QA system using Hybrid Search (Dense + BM25) with Pinecone and Transformer-based generation.",
    version="1.0.0"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize components once at startup
searcher = HybridSearcher()
generator = RAGGenerator()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")

        retrieved_chunks = searcher.search(request.query)

        if not retrieved_chunks:
            logger.warning("No relevant documents found.")
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        answer = generator.generate(request.query, retrieved_chunks)

        logger.info("Answer generated successfully.")

        return {
            "answer": answer,
            "sources": retrieved_chunks
        }

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}