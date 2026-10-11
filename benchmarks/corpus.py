import random

from .config import CorpusConfig
from engine.tokenizer import tokenizer


def load_source_words(path: str) -> list[str]:
    """Load comma-separated words from the source vocabulary file."""
    with open(path, "r", encoding="utf-8") as source:
        contents = source.read()

    words = []
    seen = set()

    for item in contents.split(","):
        normalized = tokenizer(item.strip())

        for word in normalized:
            if word not in seen:
                seen.add(word)
                words.append(word)

    return words


def _allocate_controlled_terms(config, rng):
    """Assign controlled terms to documents with exact target DFs."""
    assignments = [[] for _ in range(config.num_documents)]

    for term, settings in config.benchmark_terms.items():
        count = round(settings.df * config.num_documents)
        selected = rng.sample(range(config.num_documents), count)

        for doc_id in selected:
            tf = rng.randint(settings.min_tf, settings.max_tf)
            assignments[doc_id].extend([term] * tf)

    return assignments


def generate_corpus(
    config: CorpusConfig,
    source_words: list[str],
) -> list[str]:
    """Generate unique documents with at least min_length tokens."""
    rng = random.Random(config.seed)

    benchmark_terms = set(config.benchmark_terms)

    background_words = list(dict.fromkeys(
        word
        for raw_word in source_words
        for word in tokenizer(raw_word)
        if word not in benchmark_terms
    ))

    if len(background_words) != config.background_vocabulary_size:
        raise ValueError(
            f"Expected {config.background_vocabulary_size} distinct "
            f"background words; got {len(background_words)}"
        )

    assignments = _allocate_controlled_terms(config, rng)

    documents = []
    seen_documents = set()

    for controlled_tokens in assignments:
        tokens = list(controlled_tokens)

        while len(tokens) < config.min_length:
            tokens.append(rng.choice(background_words))

        key = tuple(sorted(tokens))

        while key in seen_documents:
            tokens.append(rng.choice(background_words))
            key = tuple(sorted(tokens))

        seen_documents.add(key)
        rng.shuffle(tokens)
        documents.append(" ".join(tokens))

    return documents


def corpus_statistics(documents, benchmark_terms):
    """Return actual document frequencies and token frequencies."""
    document_frequency = {term: 0 for term in benchmark_terms}
    token_frequency = {term: 0 for term in benchmark_terms}

    for document in documents:
        tokens = tokenizer(document)
        present = set(tokens)

        for term in benchmark_terms:
            if term in present:
                document_frequency[term] += 1
            token_frequency[term] += tokens.count(term)

    return {
        "num_documents": len(documents),
        "document_frequency": document_frequency,
        "token_frequency": token_frequency,
    }