import json
from sentence_transformers import SentenceTransformer, util
from retriever.hybrid_search import HybridSearcher
from app.services.generator import RAGGenerator

model = SentenceTransformer("all-MiniLM-L6-v2")
searcher = HybridSearcher()
generator = RAGGenerator()


def semantic_match(predicted, ground_truth, threshold=0.75):
    emb1 = model.encode(predicted, convert_to_tensor=True)
    emb2 = model.encode(ground_truth, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return score >= threshold


def evaluate():
    with open("evaluation/qa_dataset.json", "r") as f:
        dataset = json.load(f)

    correct = 0

    for item in dataset:
        question = item["question"]
        true_answer = item["answer"]

        retrieved = searcher.search(question)
        predicted = generator.generate(question, retrieved)

        if semantic_match(predicted, true_answer):
            correct += 1

        print("\nQuestion:", question)
        print("Predicted:", predicted)
        print("True:", true_answer)

    accuracy = correct / len(dataset)
    print("\nFinal Accuracy:", round(accuracy * 100, 2), "%")


if __name__ == "__main__":
    evaluate()