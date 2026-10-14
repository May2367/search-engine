from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    # Load and cache the embedding model.
    return SentenceTransformer(model_name)


def _validate_documents(documents):
    if not isinstance(documents, (list, tuple)):
        raise TypeError("documents must be a list or tuple of strings")

    if any(not isinstance(doc, str) for doc in documents):
        raise TypeError("every document must be a string")


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    # Normalize rows safely, leaving zero vectors unchanged.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms > 0,
    )


def embed_documents(documents, model=None) -> np.ndarray:
    # Encode documents into a normalized NumPy embedding matrix.
    _validate_documents(documents)

    if not documents:
        return np.empty((0, 0), dtype=np.float32)

    if model is None:
        model = _load_model()

    embeddings = np.asarray(
        model.encode(
            list(documents),
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )

    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise ValueError("document embeddings must be a 2D matrix with one row per document")

    if embeddings.shape[1] == 0:
        raise ValueError("embedding dimension must be positive")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("document embeddings must contain only finite values")

    return _normalize_rows(embeddings)


def embed_query(query: str, model=None) -> np.ndarray:
    # Encode a query into a normalized embedding vector.
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if not query.strip():
        return np.empty(0, dtype=np.float32)

    if model is None:
        model = _load_model()

    embedding = np.asarray(
        model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )

    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding[0]

    if embedding.ndim != 1 or embedding.size == 0:
        raise ValueError("query embedding must be a non-empty vector")

    if not np.all(np.isfinite(embedding)):
        raise ValueError("query embedding must contain only finite values")

    return _normalize_rows(embedding.reshape(1, -1))[0]


def semantic_search(documents, query: str, k: int = 10, embeddings: np.ndarray = None, model=None) -> list[tuple[int, float]]:
    """
    Return documents ranked by exact cosine similarity.

    Document IDs correspond to positions in the original documents list.
    Precomputed embeddings can be passed to avoid re-encoding documents.
    Ties are resolved by ascending document ID.
    """
    _validate_documents(documents)

    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be a non-negative integer")

    if k < 0:
        raise ValueError("k must be a non-negative integer")

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if k == 0 or not documents or not query.strip():
        return []

    if embeddings is None:
        embeddings = embed_documents(documents, model=model)
    else:
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")

        if embeddings.shape[0] != len(documents):
            raise ValueError("embedding rows must match the number of documents")

        if embeddings.shape[1] == 0:
            raise ValueError("embedding dimension must be positive")

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("embeddings must contain only finite values")

        embeddings = _normalize_rows(embeddings)

    query_embedding = embed_query(query, model=model)

    if query_embedding.size == 0:
        return []

    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("query and document embedding dimensions must match")

    scores = embeddings @ query_embedding

    # Guard against tiny floating-point deviations from cosine's range.
    scores = np.clip(scores, -1.0, 1.0)

    doc_ids = np.arange(len(documents))
    ranked_indices = np.lexsort((doc_ids, -scores))[:k]

    return [
        (int(i), float(scores[i]))
        for i in ranked_indices
    ]


class SemanticIndex:
    """
    Exact cosine-similarity search over a fixed set of documents.

    Validation and normalization of the document embeddings happen once,
    in the constructor, so each search only encodes the query, scores the
    documents, and ranks them. Results match semantic_search on valid
    inputs. Unlike semantic_search, malformed embeddings raise at
    construction even when a later query would have returned early.
    """

    def __init__(self, documents, embeddings: np.ndarray = None, model=None):
        _validate_documents(documents)

        # Copy the references so later changes to the caller's list
        # cannot misalign document IDs with the embeddings.
        self._documents = tuple(documents)
        self._model = model
        self._doc_ids = np.arange(len(self._documents))

        if not self._documents:
            self._embeddings = np.empty((0, 0), dtype=np.float32)
        elif embeddings is None:
            self._embeddings = embed_documents(self._documents, model=model)
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)

            if embeddings.ndim != 2:
                raise ValueError("embeddings must be a 2D matrix")

            if embeddings.shape[0] != len(self._documents):
                raise ValueError("embedding rows must match the number of documents")

            if embeddings.shape[1] == 0:
                raise ValueError("embedding dimension must be positive")

            if not np.all(np.isfinite(embeddings)):
                raise ValueError("embeddings must contain only finite values")

            self._embeddings = _normalize_rows(embeddings)

    def __len__(self) -> int:
        return len(self._documents)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return up to k (doc_id, score) pairs; ties go to the lower doc ID."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError("k must be a non-negative integer")

        if k < 0:
            raise ValueError("k must be a non-negative integer")

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if k == 0 or not self._documents or not query.strip():
            return []

        query_embedding = embed_query(query, model=self._model)

        if query_embedding.size == 0:
            return []

        if self._embeddings.shape[1] != query_embedding.shape[0]:
            raise ValueError("query and document embedding dimensions must match")

        scores = self._embeddings @ query_embedding

        # Guard against tiny floating-point deviations from cosine's range.
        scores = np.clip(scores, -1.0, 1.0)

        ranked_indices = np.lexsort((self._doc_ids, -scores))[:k]

        return [
            (int(i), float(scores[i]))
            for i in ranked_indices
        ]

