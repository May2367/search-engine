# Search Engine

A search and retrieval engine built from scratch in Python to explore how modern search systems work.

The project focuses on implementing the underlying algorithms rather than relying on search-engine libraries such as Lucene or Elasticsearch.

## Current Status

Implemented:

- Text tokenization and normalization
- Inverted index with term frequencies
- Single-term search
- Boolean AND/OR search
- Brute-force search baseline
- Unit tests

Currently working toward:

- TF-IDF and BM25 ranking
- Retrieval evaluation and benchmarking
- Semantic/vector search
- HNSW approximate nearest-neighbor search
- Simple web API and UI

## Architecture

Documents ↓ Tokenizer ↓ Inverted Index ↓ BM25 ↓ Search Results


The semantic-search path will eventually be:

Documents → Embeddings → Vector Index → HNSW → Search Results


The two approaches can eventually be compared on both retrieval quality and performance.

## Roadmap

**Lexical search** — Complete the core text-search pipeline with TF-IDF and BM25 ranking.

**Evaluation** — Add labeled datasets and benchmarks for retrieval quality, query latency, indexing time, and memory usage.

**Semantic search** — Add pretrained embeddings and exact vector search, followed by an implementation of HNSW for approximate nearest-neighbor search.

**Application** — Expose the engine through a simple API and web interface for experimentation and demonstration.

**Stretch goals** — Explore hybrid lexical/semantic search, performance-critical C++ components, persistent indexes, and other optimizations.

## Project Structure

search_engine/ ├── engine/ # Core search implementation ├── tests/ # Unit and integration tests ├── benchmarks/ # Performance experiments ├── data/ # Development datasets ├── requirements.txt └── README.md


## Development Philosophy

The project follows:

Simple baseline → Measure → Profile → Optimize


Each major optimization will be compared against a simpler reference implementation for correctness, retrieval quality, and performance.

The goal is not to recreate a production search engine, but to build and understand the core algorithms behind one.