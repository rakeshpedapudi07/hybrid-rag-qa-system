import matplotlib.pyplot as plt

# Manually update these after evaluation
results = {
    "Dense": 33.33,
    "BM25": 66.67,
    "Hybrid": 66.67
}

def plot_scores(scores):
    models = list(scores.keys())
    values = list(scores.values())

    plt.figure()
    plt.bar(models, values)

    plt.xlabel("Retrieval Method")
    plt.ylabel("Precision@1 (%)")
    plt.title("Retrieval Method Comparison")

    plt.ylim(0, 100)

    plt.savefig("evaluation/retrieval_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_scores(results)