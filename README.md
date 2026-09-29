# Search Engine

A search engine built from scratch in Python.

The goal of this project is to implement the core components of a
search engine without relying on search-engine libraries such as
Lucene, Elasticsearch, or Whoosh.

## Current Goals

- Tokenization and normalization
- Inverted index
- Boolean queries
- TF-IDF ranking
- BM25 ranking
- Phrase queries
- Fuzzy search
- Autocomplete
- Index persistence
- Performance benchmarking

## Project Structure

```text
engine/       Core search engine implementation
tests/        Unit and integration tests
data/         Test and development datasets
benchmarks/   Performance benchmarks
