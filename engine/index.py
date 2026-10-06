from .tokenizer import tokenizer

def build_index(documents: list[str]):
    index = {}

    for doc_id, document in enumerate(documents):
        tokens = tokenizer(document)

        for token in tokens:
            if token not in index:
                index[token] = {}
            
            if doc_id not in index[token]:
                index[token][doc_id] = 0

            index[token][doc_id] += 1

    return index
