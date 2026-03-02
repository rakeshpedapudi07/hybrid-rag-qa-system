import gradio as gr
from retriever.hybrid_search import HybridSearcher
from app.services.generator import RAGGenerator

# Initialize once (just like FastAPI startup)
searcher = HybridSearcher()
generator = RAGGenerator()


def rag_query(user_query: str):
    if not user_query.strip():
        return "Please enter a question.", ""

    try:
        retrieved_chunks = searcher.search(user_query)

        if not retrieved_chunks:
            return "No relevant documents found.", ""

        answer = generator.generate(user_query, retrieved_chunks)

        sources_text = "\n\n".join(retrieved_chunks)

        return answer, sources_text

    except Exception as e:
        return f"Error: {str(e)}", ""


iface = gr.Interface(
    fn=rag_query,
    inputs=gr.Textbox(lines=2, placeholder="Ask your question here..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Retrieved Sources")
    ],
    title="Hybrid RAG QA System",
    description="Dense + BM25 Hybrid Retrieval with Transformer-based Answer Generation"
)

if __name__ == "__main__":
    iface.launch()