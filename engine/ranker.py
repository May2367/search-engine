import math

from .tokenizer import tokenizer


def bm25_search(index: dict, query: str, k1: float = 1.5, b: float = 0.75):
    for name, value in (("k1", k1), ("b", b)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    if k1 < 0:
        raise ValueError("k1 must be non-negative")

    if not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1")
        
    postings = index["index"]
    stats = index["corpus_stats"]

    num_docs = stats["num_docs"]
    doc_freq = stats["doc_freq"]
    doc_lengths = stats["doc_lengths"]
    avg_doc_length = stats["avg_doc_length"]

    if num_docs == 0 or avg_doc_length == 0:
        return []

    query_terms = tokenizer(query)
    scores = {}

    for term in query_terms:
        if term not in postings:
            continue

        df = doc_freq[term]

        # BM25 inverse document frequency
        idf = math.log(
            1 + (num_docs - df + 0.5) / (df + 0.5)
        )

        for doc_id, tf in postings[term].items():
            doc_length = doc_lengths[doc_id]

            length_norm = 1 - b + b * (doc_length / avg_doc_length)

            tf_score = (
                tf * (k1 + 1) / (tf + k1 * length_norm)
            )

            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_score

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )
