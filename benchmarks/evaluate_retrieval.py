import argparse
import math
from pathlib import Path

from engine.index import build_index
from engine.ranker import bm25_search
from engine.semantic_search import (
    _load_model,
    embed_documents,
    semantic_search,
)


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "evaluation_data.txt"
)

# Relevance grades:
# 2 = highly relevant
# 1 = partially relevant
# 0 or omitted = nonrelevant for this evaluation.
#
# Document IDs are zero-based line positions in evaluation_data.txt.
# These judgments are illustrative starting labels. Review them against
# the final document file before treating results as evidence.

EVALUATION_QUERIES = [
    (
        "How does BM25 rank search results?",
        {0: 2, 1: 2, 2: 1, 7: 1, 8: 1, 9: 1},
    ),
    (
        "How can a search engine find text with similar meaning?",
        {3: 2, 4: 2, 5: 2, 6: 1, 9: 1},
    ),
    (
        "How do Python dictionaries find values using keys?",
        {10: 2, 11: 2, 12: 1},
    ),
    (
        "How can I protect myself against fraudulent emails and stolen passwords?",
        {16: 2, 17: 2, 18: 2, 19: 1},
    ),
    (
        "What are renewable ways to generate electricity from the sun and wind?",
        {20: 2, 21: 2, 22: 1, 23: 1},
    ),
    (
        "How can cities reduce excessive heat caused by buildings and pavement?",
        {24: 2, 25: 2, 26: 1, 73: 1},
    ),
    (
        "How does plastic waste affect animals and ecosystems in the ocean?",
        {27: 2, 28: 2, 29: 2, 70: 1},
    ),
    (
        "What makes sourdough bread rise and develop its flavor?",
        {30: 2, 31: 2, 32: 1},
    ),
    (
        "How can gardeners save water and maintain healthy soil?",
        {33: 1, 34: 2, 35: 2, 36: 1},
    ),
    (
        "Which foods provide fiber and protein as part of a balanced diet?",
        {37: 2, 38: 2, 39: 2},
    ),
    (
        "What affects the body's sleep cycle and internal clock?",
        {40: 2, 41: 2, 42: 2},
    ),
    (
        "How did the Romans build infrastructure to transport people and water?",
        {45: 2, 46: 2, 47: 1},
    ),
    (
        "How does compound interest affect savings, and what is inflation?",
        {51: 2, 52: 2, 53: 1, 57: 1},
    ),
    (
        "What cultural attractions and transportation options are available in Kyoto?",
        {59: 2, 60: 2, 61: 2},
    ),
    (
        "Which musical styles feature improvisation and expressive rhythms?",
        {62: 2, 63: 2, 64: 1},
    ),
    (
        "How do astronomers observe space and what happens during a lunar eclipse?",
        {71: 2, 72: 2},
    ),
    (
        "What study techniques improve long-term memory and learning?",
        {74: 1, 75: 2, 76: 2},
    ),
    (
        "How do atmospheric gases trap heat and contribute to global warming?",
        {73: 2, 24: 1, 25: 1},
    ),
]


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")


def dcg_at_k(relevances, k: int = 10) -> float:
    """Calculate discounted cumulative gain from ranked relevance grades."""
    _validate_k(k)

    if k == 0:
        return 0.0

    score = 0.0

    for rank, relevance in enumerate(relevances[:k], start=1):
        if (
            isinstance(relevance, bool)
            or not isinstance(relevance, (int, float))
            or not math.isfinite(relevance)
            or relevance < 0
        ):
            raise ValueError("relevance grades must be finite and non-negative")

        score += (2**relevance - 1) / math.log2(rank + 1)

    return score


def ndcg_at_k(
    ranked_doc_ids,
    relevance_by_doc: dict[int, int],
    k: int = 10,
) -> float:
    """Calculate normalized discounted cumulative gain for a ranked list."""
    _validate_k(k)

    if k == 0:
        return 0.0

    actual_grades = [
        relevance_by_doc.get(doc_id, 0)
        for doc_id in ranked_doc_ids[:k]
    ]

    ideal_grades = sorted(
        relevance_by_doc.values(),
        reverse=True,
    )[:k]

    ideal_dcg = dcg_at_k(ideal_grades, k)

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(actual_grades, k) / ideal_dcg


def recall_at_k(
    ranked_doc_ids,
    relevance_by_doc: dict[int, int],
    k: int = 10,
) -> float:
    """Calculate recall at k; relevance grades >= 1 count as relevant."""
    _validate_k(k)

    relevant_docs = {
        doc_id
        for doc_id, grade in relevance_by_doc.items()
        if grade >= 1
    }

    if not relevant_docs:
        return 0.0

    retrieved_docs = set(ranked_doc_ids[:k])

    return len(relevant_docs & retrieved_docs) / len(relevant_docs)


def load_documents(path: Path) -> list[str]:
    """Read one document per line, preserving line-based document IDs."""
    with path.open("r", encoding="utf-8") as file:
        documents = file.read().splitlines()

    if not documents:
        raise ValueError(f"No documents found in {path}")

    if any(not document.strip() for document in documents):
        raise ValueError(
            "Evaluation data contains a blank document line; "
            "remove blank lines to preserve document IDs."
        )

    return documents


def evaluate_method(search_fn, queries, k: int) -> dict:
    """Evaluate a retrieval function against fixed relevance judgments."""
    per_query = []

    for query, relevance in queries:
        results = search_fn(query, k)
        ranked_doc_ids = [doc_id for doc_id, _ in results]

        per_query.append(
            {
                "query": query,
                "ndcg": ndcg_at_k(ranked_doc_ids, relevance, k),
                "recall": recall_at_k(ranked_doc_ids, relevance, k),
            }
        )

    return {
        "ndcg_at_k": sum(row["ndcg"] for row in per_query)
        / len(per_query),
        "recall_at_k": sum(row["recall"] for row in per_query)
        / len(per_query),
        "per_query": per_query,
    }


def run_evaluation(data_path: Path, k: int = 10) -> dict:
    """Compare BM25 and semantic retrieval on the same labeled queries."""
    _validate_k(k)

    if k == 0:
        raise ValueError("k must be positive for evaluation")

    documents = load_documents(data_path)

    # Build the lexical index once.
    index_data = build_index(documents)

    # Load the model and embed the corpus once.
    model = _load_model()
    embeddings = embed_documents(documents, model=model)

    def run_bm25(query, limit):
        return bm25_search(index_data, query)[:limit]

    def run_semantic(query, limit):
        return semantic_search(
            documents,
            query,
            k=limit,
            embeddings=embeddings,
            model=model,
        )

    results = {
        "documents": len(documents),
        "queries": len(EVALUATION_QUERIES),
        "k": k,
        "bm25": evaluate_method(run_bm25, EVALUATION_QUERIES, k),
        "semantic": evaluate_method(
            run_semantic,
            EVALUATION_QUERIES,
            k,
        ),
    }

    return results


def print_results(results: dict) -> None:
    """Print aggregate metrics and per-query scores."""
    k = results["k"]

    print(f"Documents: {results['documents']}")
    print(f"Queries:   {results['queries']}")
    print(f"Cutoff:    {k}")

    print(f"\n{'Method':<12} {'NDCG@' + str(k):>12} {'Recall@' + str(k):>12}")
    print("-" * 38)

    for name in ("bm25", "semantic"):
        metrics = results[name]
        print(
            f"{name.upper():<12} "
            f"{metrics['ndcg_at_k']:>12.4f} "
            f"{metrics['recall_at_k']:>12.4f}"
        )

    print("\nPer-query results")

    for i, (query, _) in enumerate(EVALUATION_QUERIES):
        bm25_row = results["bm25"]["per_query"][i]
        semantic_row = results["semantic"]["per_query"][i]

        print(f"\n{query}")
        print(
            f"  BM25:     NDCG={bm25_row['ndcg']:.4f}, "
            f"Recall={bm25_row['recall']:.4f}"
        )
        print(
            f"  Semantic: NDCG={semantic_row['ndcg']:.4f}, "
            f"Recall={semantic_row['recall']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 and semantic retrieval quality."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to a one-document-per-line evaluation corpus.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Ranking cutoff for evaluation metrics.",
    )
    args = parser.parse_args()

    results = run_evaluation(args.data, args.k)
    print_results(results)


if __name__ == "__main__":
    main()