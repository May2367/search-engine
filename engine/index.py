from .tokenizer import tokenizer

def build_index(documents: list[str]):
    index = {}

    for doc_id, document in enumerate(documents):
        tokens = tokenizer(document)

        for token in tokens:
            if token in index:
                index[token].append(doc_id)
            else:
                index[token] = [doc_id]

    return index
