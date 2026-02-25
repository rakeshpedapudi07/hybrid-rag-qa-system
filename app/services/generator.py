from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class RAGGenerator:
    def __init__(self):
        model_name = "google/flan-t5-base"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, query: str, context_chunks):
        context = "\n\n".join(context_chunks)

        prompt = f"""
Use ONLY the context below to answer the question clearly and completely.

Context:
{context}

Question:
{query}

Answer in one complete sentence:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()