from .tokenizer import tokenizer


def build_index(documents: list[str]) -> dict:
    index = {}
    doc_lengths = {}

    for doc_id, document in enumerate(documents):
        tokens = tokenizer(document)
        doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in index:
                index[token] = {}

            if doc_id not in index[token]:
                index[token][doc_id] = 0

            index[token][doc_id] += 1

    num_docs = len(documents)
    ttl_tokens = sum(doc_lengths.values())

    corpus_stats = {
        "num_docs": num_docs,
        "doc_freq": {
            term: len(postings)
            for term, postings in index.items()
        },
        "doc_lengths": doc_lengths,
        "avg_doc_length": (
            ttl_tokens / num_docs if num_docs else 0.0
        ),
    }

    return {
        "index": index,
        "corpus_stats": corpus_stats,
    }
