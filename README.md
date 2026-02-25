# Hybrid RAG QA System

A scalable NLP-driven Retrieval-Augmented Generation (RAG) system combining Dense Retrieval (Sentence Transformers) and BM25 sparse search with Pinecone vector database and FastAPI backend.

---

## 🚀 Project Overview

This project implements a Hybrid Retrieval-Augmented Generation pipeline designed to improve factual accuracy and retrieval precision over large document corpora.

The system supports:

- Dense semantic retrieval using sentence-transformers (all-MiniLM-L6-v2)
- Sparse retrieval using BM25
- Hybrid search combining dense + sparse scores
- FastAPI-based inference API
- Evaluation framework for Precision@K comparison
- Corpus generation and ingestion pipeline

---

## 🏗 Architecture
```
┌────────────────────────────────────────────┐
│                Client Request              │
│        (HTTP / REST API via FastAPI)       │
└────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│             API Layer (FastAPI)            │
│  • Request Validation (Pydantic)           │
│  • Routing & Endpoint Handling             │
└────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│           Hybrid Retrieval Layer           │
│  • Dense Search (Sentence Transformers)    │
│  • Sparse Search (BM25)                    │
│  • Score Fusion & Ranking                  │
└────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│        Top-K Retrieved Context Chunks      │
│        (Semantic + Lexical Relevance)      │
└────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│          Generation Layer (RAG)            │
│  • Context Injection                       │
│  • Prompt Construction                     │
│  • Answer Synthesis                        │
└────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│            Final Response Payload          │
│  • Generated Answer                        │
│  • Source Documents                        │
└────────────────────────────────────────────┘
```
---

## 📊 Retrieval Performance (Precision@1)
```
| Method  | Precision@1 |
|----------|-------------|
| Dense    | 33.33%      |
| BM25     | 66.67%      |
| Hybrid   | 66.67%      |
```
Hybrid retrieval improves reliability compared to standalone dense retrieval.

---
```
## 📂 Project Structure
hybrid-rag-qa-system/
│
├── app/                          # FastAPI application layer
│   ├── main.py                   # API entry point
│   └── services/
│       └── generator.py          # RAG answer generation logic
│
├── ingestion/                    # Data ingestion pipeline
│   ├── loader.py                 # Document loading (TXT, PDF)
│   ├── chunker.py                # Semantic text chunking
│   └── embedder.py               # Sentence-transformer embeddings
│
├── retriever/                    # Retrieval layer
│   ├── pinecone_client.py        # Vector database interface
│   └── hybrid_search.py          # Dense + BM25 hybrid retrieval
│
├── evaluation/                   # Evaluation & benchmarking
│   ├── evaluate_retrieval.py     # Precision@K evaluation
│   ├── evaluate_accuracy.py      # QA accuracy testing
│   ├── metrics.py                # Metric utilities
│   ├── plot_results.py           # Retrieval comparison plots
│   └── qa_dataset.json           # Evaluation dataset
│
├── scripts/                      # Utility scripts
│   ├── ingest.py                 # Full ingestion pipeline runner
│   └── generate_corpus.py        # Large-scale corpus generator
│
├── data/                         # Local document storage
│   ├── sample.txt
│   └── noise.txt
│
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation
├── LICENSE
└── .gitignore
```
---

## ⚙️ Tech Stack

- Python 3.12
- FastAPI
- Pinecone (Vector DB)
- Sentence Transformers
- BM25
- HuggingFace Datasets
- Matplotlib
- PyPDF2

---

## 🛠 Setup Instructions

### 1️⃣ Clone repository

```bash
git clone https://github.com/rakeshpedapudi07/hybrid-rag-qa-system.git
cd hybrid-rag-qa-system
```
### 2️⃣ Create virtual environment
```
python -m venv venv
venv\Scripts\activate
```
### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```
### 📥 Generate Corpus (Optional Large Scale Test)
```
python scripts/generate_corpus.py
```
### 📌 Run Ingestion
```
python -m scripts.ingest
```
### 📈 Evaluate Retrieval
```
python -m evaluation.evaluate_retrieval
```
### Plot results:
```
python -m evaluation.plot_results
```
### 🌐 Run API Server
```
uvicorn app.main:app --reload
```
### API endpoint:

POST /query
Example request:
```
{
  "query": "What improves factual accuracy?"
}
```
### 🎯 Key Highlights

- Designed for scalable document ingestion (tested on 10K+ documents)
- Modular retriever architecture
- Evaluation metrics included
- Hybrid search implementation
- Clean project structure for production scaling

### 📌 Future Improvements

- Cross-Encoder Re-Ranking for improved top-1 precision
- Query Expansion Techniques for semantic coverage improvement
- LLM Fine-Tuning for domain-specific optimization
- Interactive UI (Streamlit / React Frontend)
- Dockerized Deployment & Cloud Hosting

### 📄 License

This project is licensed under the **MIT License**.


