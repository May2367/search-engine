import argparse

from .config import BenchmarkConfig, CorpusConfig
from .corpus import (
    corpus_statistics,
    generate_corpus,
    load_source_words,
)
from .runner import benchmark
from engine.index import build_index
from engine.search import (
    brute_force_search,
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


def run_benchmark(
    corpus_config: CorpusConfig,
    benchmark_config: BenchmarkConfig,
    source_words: list[str],
):
    # Generate corpus outside timed operations.
    documents = generate_corpus(corpus_config, source_words)

    print(f"Documents: {len(documents)}")
    print(f"Minimum tokens per document: {corpus_config.min_length}")

    lengths = [len(document.split()) for document in documents]
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

    # Measure index construction separately.
    index_results = benchmark(
        lambda: build_index(documents),
        benchmark_config.warmup_runs,
        benchmark_config.measured_runs,
    )

    index = build_index(documents)["index"]

    # Normalize queries before timing either search method.
    raw_queries = [
        *corpus_config.benchmark_terms,
        "nonexistentterm",
        "missingtoken",
    ]
    
    queries = []
    for query in raw_queries:
        tokens = tokenizer(query)
        if tokens:
            queries.append(tokens[0])
        
    # Verify correctness before comparing performance.
    for term in queries:
        expected = set(brute_force_search(documents, term))
        actual = set(single_term_basic_index_search(index, term))

        if expected != actual:
            raise AssertionError(f"Search results differ for {term}")

    def run_brute_force():
        for term in queries:
            brute_force_search(documents, term)

    def run_index_search():
        for term in queries:
            single_term_basic_index_search(index, term)

    brute_results = benchmark(
        run_brute_force,
        benchmark_config.warmup_runs,
        benchmark_config.measured_runs,
    )

    indexed_results = benchmark(
        run_index_search,
        benchmark_config.warmup_runs,
        benchmark_config.measured_runs,
    )

    print_results("Index construction", index_results)
    print_results(
        f"Brute force ({len(queries)} queries per run)",
        brute_results,
    )
    print_results(
        f"Inverted index ({len(queries)} queries per run)",
        indexed_results,
    )

    print("\nCorrectness: passed.")

    return {
        "index_construction": index_results,
        "brute_force_batch": brute_results,
        "indexed_search_batch": indexed_results,
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
