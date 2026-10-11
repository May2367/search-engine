import argparse
import math

from .config import BenchmarkConfig, CorpusConfig
from .corpus import (
    corpus_statistics,
    generate_corpus,
    load_source_words,
)
from .runner import benchmark
from engine.index import build_index
from engine.ranker import bm25_search
from engine.search import (
    and_index_search,
    brute_force_search,
    or_index_search,
    single_term_basic_index_search,
)
from engine.tokenizer import tokenizer


def print_results(name: str, results: dict):
    print(f"\n{name}")

    for metric, value in results.items():
        if metric == "runs":
            print(f"  {metric}: {value}")
        else:
            print(f"  {metric}: {value:.4f} ms")


def brute_force_and_search(documents, terms):
    """Reference AND search directly against the documents."""
    if not terms:
        return []

    query_terms = set(terms)
    results = []

    for doc_id, document in enumerate(documents):
        document_terms = set(tokenizer(document))
        if query_terms.issubset(document_terms):
            results.append(doc_id)

    return results


def brute_force_or_search(documents, terms):
    """Reference OR search directly against the documents."""
    query_terms = set(terms)
    results = []

    for doc_id, document in enumerate(documents):
        if query_terms.intersection(tokenizer(document)):
            results.append(doc_id)

    return results


def run_benchmark(
    corpus_config: CorpusConfig,
    benchmark_config: BenchmarkConfig,
    source_words: list[str],
):
    # Generate the corpus outside timed operations.
    documents = generate_corpus(corpus_config, source_words)

    if not documents:
        raise ValueError("Benchmark corpus must contain documents")

    print(f"Documents: {len(documents)}")
    print(f"Minimum tokens per document: {corpus_config.min_length}")

    lengths = [len(tokenizer(document)) for document in documents]
    print(f"Document length range: {min(lengths)}–{max(lengths)} tokens")
    print(f"Seed: {corpus_config.seed}")

    stats = corpus_statistics(
        documents,
        corpus_config.benchmark_terms,
    )

    print("\nControlled-term statistics:")
    for term, target in corpus_config.benchmark_terms.items():
        actual_df = stats["document_frequency"][term]
        actual_tf = stats["token_frequency"][term]
        expected_df = round(target.df * len(documents))

        print(
            f"  {term}: DF={actual_df} "
            f"(target {expected_df}), TF={actual_tf}"
        )

        if actual_df != expected_df:
            raise AssertionError(f"Unexpected DF for {term}")

    # Build the index once; exclude construction from search timings.
    index_results = benchmark(
        lambda: build_index(documents),
        benchmark_config.warmup_runs,
        benchmark_config.measured_runs,
    )
    index_data = build_index(documents)
    index = index_data["index"]

    # Single-term queries.
    queries = [
        *corpus_config.benchmark_terms,
        "nonexistentterm",
        "missingtoken",
    ]
    queries = [
        tokenizer(query)[0]
        for query in queries
        if tokenizer(query)
    ]

    # Multi-term queries cover common, rare, and overlapping terms.
    configured_terms = corpus_config.benchmark_terms
    and_or_queries = [
        ["commonfrequency", "mediumfrequency"],
        ["overlapalpha", "overlapbeta"],
        ["overlapalpha", "overlapbeta", "overlapgamma"],
        ["rarefrequency", "veryrarefrequency"],
        ["commonfrequency", "overlapalpha", "repeatedterm"],
        ["commonfrequency", "nonexistentterm"],
    ]

    # Verify single-term retrieval against brute force.
    for term in queries:
        expected = set(brute_force_search(documents, term))
        actual = set(single_term_basic_index_search(index, term))

        if expected != actual:
            raise AssertionError(f"Single-term results differ for {term}")

    # Verify AND and OR results against independent document scans.
    for terms in and_or_queries:
        expected_and = set(brute_force_and_search(documents, terms))
        actual_and = set(and_index_search(index, terms))

        if expected_and != actual_and:
            raise AssertionError(
                f"AND results differ for {terms}: "
                f"expected {len(expected_and)}, got {len(actual_and)}"
            )

        expected_or = set(brute_force_or_search(documents, terms))
        actual_or = set(or_index_search(index, terms))

        if expected_or != actual_or:
            raise AssertionError(
                f"OR results differ for {terms}: "
                f"expected {len(expected_or)}, got {len(actual_or)}"
            )

    # Benchmark single-term retrieval.
    def run_brute_force():
        for term in queries:
            brute_force_search(documents, term)

    def run_single_term_index():
        for term in queries:
            single_term_basic_index_search(index, term)

    brute_results = benchmark(
        run_brute_force,
        benchmark_config.warmup_runs,
        benchmark_config.measured_runs,
    )
    single_results = benchmark(
        run_single_term_index,
        benchmark_config.warmup_runs,
        benchmark_config.measured_runs,
    )

    # Benchmark Boolean AND and OR independently.
    def run_and_search():
        for terms in and_or_queries:
            and_index_search(index, terms)

    def run_or_search():
        for terms in and_or_queries:
            or_index_search(index, terms)

    and_results = benchmark(
        run_and_search,
        benchmark_config.warmup_runs,
        benchmark_config.measured_runs,
    )
    or_results = benchmark(
        run_or_search,
        benchmark_config.warmup_runs,
        benchmark_config.measured_runs,
    )

    # Benchmark ranked retrieval separately.
    ranked_queries = [
        " ".join(terms)
        for terms in and_or_queries
    ]

    for query in ranked_queries:
        results = bm25_search(index_data, query)
        scores = [score for _, score in results]
        doc_ids = [doc_id for doc_id, _ in results]

        if scores != sorted(scores, reverse=True):
            raise AssertionError(f"BM25 scores are not descending: {query}")

        if len(doc_ids) != len(set(doc_ids)):
            raise AssertionError(f"Duplicate BM25 document IDs: {query}")

        for doc_id, score in results:
            if doc_id not in index_data["corpus_stats"]["doc_lengths"]:
                raise AssertionError(f"Invalid BM25 document ID: {doc_id}")
            if not math.isfinite(score):
                raise AssertionError(f"Invalid BM25 score: {score}")

    def run_bm25_search():
        for query in ranked_queries:
            bm25_search(index_data, query)

    bm25_results = benchmark(
        run_bm25_search,
        benchmark_config.warmup_runs,
        benchmark_config.measured_runs,
    )

    # Print timing results.
    print_results("Index construction", index_results)
    print_results(
        f"Brute force single-term search ({len(queries)} queries/run)",
        brute_results,
    )
    print_results(
        f"Inverted index single-term search ({len(queries)} queries/run)",
        single_results,
    )
    print_results(
        f"Boolean AND search ({len(and_or_queries)} queries/run)",
        and_results,
    )
    print_results(
        f"Boolean OR search ({len(and_or_queries)} queries/run)",
        or_results,
    )
    print_results(
        f"BM25 ranked retrieval ({len(ranked_queries)} queries/run)",
        bm25_results,
    )

    print("\nCorrectness: passed.")

    return {
        "index_construction": index_results,
        "brute_force_batch": brute_results,
        "indexed_search_batch": single_results,
        "and_search_batch": and_results,
        "or_search_batch": or_results,
        "bm25_ranked_search_batch": bm25_results,
        "corpus_statistics": stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/background_corpus.txt",
    )
    parser.add_argument("--documents", type=int, default=1_000)
    parser.add_argument("--min-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()

    source_words = load_source_words(args.source)

    corpus_config = CorpusConfig(
        num_documents=args.documents,
        seed=args.seed,
        min_length=args.min_length,
    )
    benchmark_config = BenchmarkConfig(
        warmup_runs=args.warmups,
        measured_runs=args.runs,
    )

    run_benchmark(corpus_config, benchmark_config, source_words)


if __name__ == "__main__":
    main()
