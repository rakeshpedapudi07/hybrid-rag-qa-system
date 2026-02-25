from datasets import load_dataset
import os

def generate_corpus(num_docs=10000):
    print("Starting corpus generation...")

    dataset = load_dataset("ag_news", split="train")

    os.makedirs("data/wiki", exist_ok=True)

    for i in range(min(num_docs, len(dataset))):
        text = dataset[i]["text"]
        with open(f"data/wiki/doc_{i}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        if i % 1000 == 0:
            print(f"Saved {i} documents")

    print("Done generating corpus.")


if __name__ == "__main__":
    generate_corpus(10000)