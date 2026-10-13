# Search Engine

A search and retrieval engine built from scratch in Python to explore how modern search systems work. It implements an inverted index, Boolean search, BM25 ranking, and exact semantic (embedding) search, and compares them on retrieval quality and speed.

## Status

Implemented:

- Tokenization and normalization
- Inverted index with term frequencies and corpus statistics
- Single-term and Boolean AND/OR search, plus a brute-force baseline
- BM25 ranking
- Exact semantic search (cosine similarity over `all-MiniLM-L6-v2` embeddings)
- Retrieval evaluation (NDCG@10, Recall@10) and a latency benchmark suite
- Unit tests

## Architecture

```
Documents -> Tokenizer -> Inverted Index -> BM25 -> ranked (doc_id, score)
Documents -> Embedding model -> Embedding matrix -> Cosine similarity -> ranked (doc_id, score)
```

- **BM25** (`engine/ranker.py`): Lucene-style IDF, `k1=1.5`, `b=0.75`. A document's score is the sum over query terms, so multi-term queries behave like OR. Ties break by ascending doc ID.
- **Semantic search** (`engine/semantic_search.py`): documents and queries are encoded with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), L2-normalized, and scored by dot product (equal to cosine similarity). Embeddings are kept in memory.

## Setup

```
pip install -r requirements.txt
```

Developed and tested on Python 3.11.9. On Apple's system Python 3.9.6, numpy is capped at 2.0.2, and the semantic search matmul emitted `RuntimeWarning`s (divide by zero, overflow, invalid value). Input embeddings were finite and correctly normalized, and rankings and metrics were correct. The warnings disappeared after moving to Python 3.11.9 with numpy 2.4.6. I did not isolate the exact cause.

## Usage

```
pytest                                    # unit tests
python -m benchmarks.evaluate_retrieval   # BM25 vs semantic quality
python -m benchmarks.benchmark_search     # latency benchmarks
```

Benchmark options: `--documents`, `--min-length`, `--seed`, `--runs`, `--warmups`. For example:

```
python -m benchmarks.benchmark_search --documents 10000 --min-length 30 --seed 42 --runs 20 --warmups 5
```

## Results

### Retrieval quality

90 documents, 18 natural-language queries, graded relevance labels (0/1/2), cutoff k=10. Python 3.11.9, numpy 2.4.6, sentence-transformers 6.1.0.

| Method   | NDCG@10 | Recall@10 |
|----------|---------|-----------|
| BM25     | 0.7404  | 0.7870    |
| Semantic | 0.9742  | 0.9815    |

### Speed

All runs: MacBook Air (Apple Silicon, 10 logical CPUs), macOS 15.5, Python 3.11.9, numpy 2.4.6, torch 2.14.1, sentence-transformers 6.1.0. Synthetic corpus (see Limitations); every document in a run has the same length. Values are medians, except document embedding, which is a single timing. Seed 42; 20 runs and 5 warmups up to 10k documents, 5 runs and 1 warmup at 50k and 100k.

#### Scaling with corpus size (30 tokens per document)

| Documents | Index build | Brute force (10 queries) | Inverted index (10 queries) | BM25 (6 queries) | Document embedding |
|-----------|-------------|--------------------------|-----------------------------|------------------|--------------------|
| 1,000     | 19 ms       | 153 ms                   | 0.002 ms                    | 0.21 ms          | 0.7 s              |
| 10,000    | 195 ms      | 1,549 ms                 | 0.013 ms                    | 2.2 ms           | 6.0 s              |
| 50,000    | 998 ms      | 7,693 ms                 | 0.064 ms                    | 11.3 ms          | 29.8 s             |
| 100,000   | 2,114 ms    | 15,434 ms                | 0.128 ms                    | 22.8 ms          | 59.5 s             |

Semantic search, per query:

| Documents | Query encoding | Vector scan   | End-to-end | Embedding matrix |
|-----------|----------------|---------------|------------|------------------|
| 1,000     | ~4.3 ms        | 0.02 ms       | ~4.6 ms    | 1.5 MiB          |
| 10,000    | ~4.3 ms        | ~0.4-0.55 ms  | ~9.2 ms    | 14.7 MiB         |
| 50,000    | ~4.5 ms        | ~3.9 ms       | ~39 ms     | 73.2 MiB         |
| 100,000   | ~4.5 ms        | ~9.3 ms       | ~74 ms     | 146.5 MiB        |

#### Scaling with document length (10,000 documents)

| Tokens per document | Index build | Brute force (10 queries) | BM25 (6 queries) | Document embedding |
|---------------------|-------------|--------------------------|------------------|--------------------|
| 10                  | 67 ms       | 536 ms                   | 2.2 ms           | 3.5 s              |
| 30                  | 195 ms      | 1,549 ms                 | 2.2 ms           | 6.0 s              |
| 100                 | 705 ms      | 5,584 ms                 | 2.2 ms           | 14.0 s             |
| 200                 | 1,415 ms    | 11,196 ms                | 2.2 ms           | 26.9 s             |

The BM25 column is constant because those queries use controlled terms whose document frequencies do not change with length.

Three seeds (7, 42, 123) at 10,000 documents and 30 tokens agreed within 1% on index build, brute force, BM25, and document embedding.

#### Takeaways

- Index build, brute force, and document embedding grow roughly linearly with both corpus size and document length.
- Inverted-index lookups stay far below brute force (0.13 ms vs 15.4 s for 10 queries at 100,000 documents). The brute-force baseline re-tokenizes every document per query, so this gap reflects precomputation, not a tuned scan.
- BM25 latency tracks how many documents match: at 100,000 documents, ~0.2 ms for a query matching ~1,100 documents and ~6 ms for one matching ~25,000.
- Semantic query encoding costs ~4-4.5 ms regardless of corpus size. The vector scan grows from ~0.5 ms at 10,000 documents to ~9 ms at 100,000, a bit faster than linear.
- Embedding the corpus is the main up-front semantic cost: ~60 s at 100,000 documents, versus ~2 s to build the BM25 index.
- End-to-end semantic search exceeds encoding plus scan by ~0.3 ms at 1,000 documents, ~4 ms at 10,000, ~31 ms at 50,000, and ~60 ms at 100,000. This is likely per-call validation and normalization of the full embedding matrix in `semantic_search`; I have not profiled it.
- Model loading takes ~1 s. Two runs showed ~48 s because of Hugging Face request retries, which I treat as noise.

## Limitations

- **The evaluation set is small and self-labeled** (90 documents, 18 queries, written by me). Treat it as a sanity check, not strong evidence. It also favors semantic search, since most queries are paraphrases.
- **The benchmark corpus is synthetic**, built from a background word list plus controlled terms with fixed document frequencies. All documents in a run have the same length, so absolute timings are not representative of real text.
- **Semantic timings in the benchmark use synthetic token strings**, so they compare speed only, not quality. Quality is measured in `evaluate_retrieval`.
- **Speed results come from one laptop**, with only 5 runs at 50,000 and 100,000 documents.
- **The brute-force baseline re-tokenizes every document on each query**, so its gap to the inverted index shows the benefit of precomputation, not a tuned scan.

## Next Steps

1. **Investigate semantic overhead:** end-to-end semantic search is slower than query encoding plus the vector scan, and the gap grows with corpus size. I suspect per-call validation and normalization of the embedding matrix, but have not profiled it.
2. **Benchmark on real text:** repeat the speed tests on a real document collection.
3. **Strengthen the evaluation set:** add queries where BM25 should win (exact names, rare keywords) and more queries overall.
4. **Normalize queries consistently:** apply the same tokenization to queries in single-term and brute-force search.
5. **Memory measurements:** track memory usage during index construction.
6. **Simple API and web UI:** use FastAPI to let users choose BM25 or semantic search.
7. **HNSW approximate nearest-neighbor search:** implement it and compare against exact search.
8. **Hybrid lexical/semantic search:** combine BM25 and semantic scores into a single ranking.

## Development Philosophy

Simple baseline -> measure -> profile -> optimize. Each optimization is compared against a simpler reference implementation for correctness, retrieval quality, and performance.
