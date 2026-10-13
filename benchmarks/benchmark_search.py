import argparse
import importlib.metadata
import math
import os
import platform
import statistics
import time

import numpy as np

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
from engine.semantic_search import (
    _load_model,
    embed_documents,
    embed_query,
    semantic_search,
)
from engine.tokenizer import tokenizer


def print_results(name: str, results: dict):
    """Print results from the existing benchmark runner."""
    print(f"\n{name}")

    for metric, value in results.items():
        if metric == "runs":
            print(f"  {metric}: {value}")
        else:
            print(f"  {metric}: {value:.4f} ms")


def measure_latency(fn, warmups, runs):
    """Measure repeated per-call latency in milliseconds."""
    for _ in range(warmups):
        fn()

    samples = []

    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)

    return {
        "runs": runs,
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def print_latency_stats(name, results):
    """Print repeated-run latency statistics."""
    print(f"  {name}")
    print(f"    Runs:   {results['runs']}")
    print(f"    Median: {results['median_ms']:.4f} ms")
    print(f"    P95:    {results['p95_ms']:.4f} ms")
    print(f"    Min:    {results['min_ms']:.4f} ms")
    print(f"    Max:    {results['max_ms']:.4f} ms")


def print_environment():
    """Print relevant hardware and software information."""
    print("\nBenchmark environment")
    print(f"  OS: {platform.platform()}")
    print(f"  Processor: {platform.processor() or 'Not reported'}")
    print(f"  Logical CPUs: {os.cpu_count()}")
    print(f"  Python: {platform.python_version()}")

    for package in ("numpy", "sentence-transformers", "torch"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "Not installed"

        print(f"  {package}: {version}")


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
            raise AssertionError(
                f"Single-term results differ for {term}"
            )

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
            raise AssertionError(
                f"BM25 scores are not descending: {query}"
            )

        if len(doc_ids) != len(set(doc_ids)):
            raise AssertionError(
                f"Duplicate BM25 document IDs: {query}"
            )

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

    # Preserve all existing lexical benchmark measurements.
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

    # ---------------------------------------------------------
    # Semantic search benchmarks
    # ---------------------------------------------------------
    print_environment()

    # Measure model initialization separately. Clearing the cache
    # ensures this measurement includes a fresh model initialization.
    _load_model.cache_clear()

    start = time.perf_counter()
    model = _load_model()
    model_loading_ms = (time.perf_counter() - start) * 1000

    print(f"\nSemantic model loading: {model_loading_ms:.2f} ms")

    # Measure corpus embedding time independently.
    start = time.perf_counter()
    semantic_embeddings = embed_documents(documents, model=model)
    embedding_ms = (time.perf_counter() - start) * 1000

    embedding_bytes = semantic_embeddings.nbytes

    print(f"Document embedding time: {embedding_ms:.2f} ms")
    print(
        f"Embedding matrix: {semantic_embeddings.shape}, "
        f"{embedding_bytes / (1024 ** 2):.2f} MiB"
    )

    # Use identical natural-language queries for both retrieval methods.
    semantic_queries = [
        "common frequency medium frequency",
        "overlap alpha overlap beta",
        "rare frequency repeated term",
        "nonexistent term",
        "common frequency overlap alpha repeated term",
    ]

    # Real controlled-term queries taken from ranked_queries. The
    # three-term overlap query is left out to keep the set at five.
    # and_or_queries / ranked_queries themselves are unchanged.
    controlled_term_queries = [
        query
        for query in ranked_queries
        if query != "overlapalpha overlapbeta overlapgamma"
    ]

    # All comparison queries: the five natural-language queries followed
    # by the five controlled-term queries.
    comparison_queries = semantic_queries + controlled_term_queries

    # BM25 result counts per comparison query (informational only).
    bm25_result_counts = {
        query: len(bm25_search(index_data, query))
        for query in comparison_queries
    }

    # Encode query vectors outside the vector-only timing.
    query_vectors = [
        embed_query(query, model=model)
        for query in comparison_queries
    ]

    def run_vector_search(query_vector, k=10):
        """Exact vector retrieval, excluding model inference."""
        scores = semantic_embeddings @ query_vector
        scores = np.clip(scores, -1.0, 1.0)

        doc_ids = np.arange(len(documents))
        indices = np.lexsort((doc_ids, -scores))[:k]

        return [
            (int(i), float(scores[i]))
            for i in indices
        ]

    # Validate semantic result structure and deterministic ranking.
    for query in comparison_queries:
        results = semantic_search(
            documents,
            query,
            k=10,
            embeddings=semantic_embeddings,
            model=model,
        )

        if len(results) > 10:
            raise AssertionError("Semantic search exceeded k=10")

        doc_ids = [doc_id for doc_id, _ in results]

        if len(doc_ids) != len(set(doc_ids)):
            raise AssertionError("Duplicate semantic document IDs")

        if any(
            doc_id < 0 or doc_id >= len(documents)
            for doc_id in doc_ids
        ):
            raise AssertionError("Invalid semantic document ID")

        if any(not math.isfinite(score) for _, score in results):
            raise AssertionError("Invalid semantic similarity score")

        repeated = semantic_search(
            documents,
            query,
            k=10,
            embeddings=semantic_embeddings,
            model=model,
        )

        if results != repeated:
            raise AssertionError("Semantic ranking is not deterministic")

    # End-to-end semantic latency includes query embedding.
    semantic_e2e_results = [
        measure_latency(
            lambda query=query: semantic_search(
                documents,
                query,
                k=10,
                embeddings=semantic_embeddings,
                model=model,
            ),
            benchmark_config.warmup_runs,
            benchmark_config.measured_runs,
        )
        for query in comparison_queries
    ]

    # Vector-only latency excludes query embedding and document encoding.
    vector_results = [
        measure_latency(
            lambda vector=vector: run_vector_search(vector),
            benchmark_config.warmup_runs,
            benchmark_config.measured_runs,
        )
        for vector in query_vectors
    ]

    # Query encoding latency, measured on its own.
    query_encoding_results = [
        measure_latency(
            lambda query=query: embed_query(query, model=model),
            benchmark_config.warmup_runs,
            benchmark_config.measured_runs,
        )
        for query in comparison_queries
    ]

    # BM25 comparison uses exactly the same queries and corpus.
    bm25_comparison_results = [
        measure_latency(
            lambda query=query: bm25_search(index_data, query),
            benchmark_config.warmup_runs,
            benchmark_config.measured_runs,
        )
        for query in comparison_queries
    ]

    print("\nSemantic query comparison (per-query latency)")

    for i, query in enumerate(comparison_queries):
        print(f"\n  Query: {query!r}")
        print(f"  BM25 results returned: {bm25_result_counts[query]}")
        print_latency_stats("BM25", bm25_comparison_results[i])
        print_latency_stats(
            "Semantic end-to-end",
            semantic_e2e_results[i],
        )
        print_latency_stats(
            "Semantic query encoding only",
            query_encoding_results[i],
        )
        print_latency_stats("Semantic vector-only", vector_results[i])

    print("\nCorrectness: passed.")

    return {
        # Existing results: preserved.
        "index_construction": index_results,
        "brute_force_batch": brute_results,
        "indexed_search_batch": single_results,
        "and_search_batch": and_results,
        "or_search_batch": or_results,
        "bm25_ranked_search_batch": bm25_results,
        "corpus_statistics": stats,

        # Additional semantic results.
        "semantic_model_loading_ms": model_loading_ms,
        "semantic_document_embedding_ms": embedding_ms,
        "semantic_embedding_matrix_bytes": embedding_bytes,
        "semantic_end_to_end": semantic_e2e_results,
        "semantic_vector_only": vector_results,
        "semantic_query_encoding": query_encoding_results,
        "bm25_comparison": bm25_comparison_results,
        "bm25_result_counts": bm25_result_counts,
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
