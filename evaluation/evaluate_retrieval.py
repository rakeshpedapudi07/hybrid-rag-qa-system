import json
from retriever.hybrid_search import HybridSearcher


def evaluate():
    with open("evaluation/qa_dataset.json", "r") as f:
        dataset = json.load(f)

    modes = ["dense", "bm25", "hybrid"]
    k = 1

    for mode in modes:
        print(f"\n===== Evaluating Mode: {mode.upper()} =====")

        searcher = HybridSearcher(mode=mode)
        correct = 0

        for item in dataset:
            question = item["question"]
            true_answer = item["answer"]

            retrieved = searcher.search(question, top_k=k)

            found = any(true_answer.lower() in chunk.lower() for chunk in retrieved)

            if found:
                correct += 1

            print("\nQuestion:", question)
            print("Found in top", k, ":", found)

        precision = correct / len(dataset)
        print(f"\n{mode.upper()} Precision@{k}: {round(precision * 100, 2)}%")


if __name__ == "__main__":
    evaluate()