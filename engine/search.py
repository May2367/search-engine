from .tokenizer import tokenizer

def brute_force_search(documents, term):
    results = []

    for doc_id, document in enumerate(documents):
        tokens = tokenizer(document)

        if term in tokens:
            results.append(doc_id)

    return results

def single_term_basic_index_search(index, term):
    if term not in index:
        return []

    return list(index[term].keys())

def and_index_search(index, terms):
    if not terms:
        return []

    doc_sets = []

    for term in terms:
        if term not in index:
            return []

        doc_sets.append(set(index[term].keys()))

    result = doc_sets[0]

    for doc_set in doc_sets[1:]:
        result &= doc_set

    return list(result)

def or_index_search(index, terms):
    result = set()

    for term in terms:
        if term in index:
            result |= index[term].keys()

    return list(result)
