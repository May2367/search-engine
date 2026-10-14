import os

import numpy as np
import pytest

import engine.semantic_search as semantic
from engine.semantic_search import (
    SemanticIndex,
    embed_documents,
    embed_query,
    semantic_search,
)


class FakeModel:
    """Deterministic model substitute for unit tests."""

    def encode(self, texts, **kwargs):
        vectors = {
            "cat": [1.0, 0.0],
            "feline": [0.9, 0.1],
            "dog": [0.0, 1.0],
            "opposite": [-1.0, 0.0],
            "zero": [0.0, 0.0],
        }

        def vector(text):
            return vectors.get(text, [1.0, 0.0])

        if isinstance(texts, str):
            return np.array(vector(texts), dtype=np.float32)

        return np.array(
            [vector(text) for text in texts],
            dtype=np.float32,
        )


@pytest.fixture
def fake_model():
    return FakeModel()


def test_document_embeddings_are_normalized(fake_model):
    embeddings = embed_documents(
        ["cat", "feline", "dog"],
        model=fake_model,
    )

    assert embeddings.shape == (3, 2)
    assert np.linalg.norm(embeddings[0]) == pytest.approx(1.0)
    assert np.linalg.norm(embeddings[2]) == pytest.approx(1.0)


def test_query_embedding_matches_document_dimensions(fake_model):
    documents = embed_documents(["cat"], model=fake_model)
    query = embed_query("cat", model=fake_model)

    assert documents.shape[1] == query.shape[0] == 2
    assert np.dot(documents[0], query) == pytest.approx(1.0)


def test_empty_documents_return_empty_matrix():
    embeddings = embed_documents([])

    assert embeddings.shape == (0, 0)


def test_empty_query_returns_empty_vector():
    assert embed_query("   ").size == 0


@pytest.mark.parametrize("documents", [None, "cat", [1, "cat"]])
def test_invalid_documents_raise(documents):
    with pytest.raises(TypeError):
        embed_documents(documents)


@pytest.mark.parametrize("query", [None, 123, ["cat"]])
def test_invalid_queries_raise(query):
    with pytest.raises(TypeError):
        embed_query(query)


@pytest.mark.parametrize("k", [-1, 1.5, True, "2"])
def test_invalid_k_raises(k):
    with pytest.raises(ValueError):
        semantic_search(["cat"], "cat", k=k, embeddings=np.array([[1.0, 0.0]]))


def test_zero_k_returns_empty_results():
    assert semantic_search(["cat"], "cat", k=0) == []


def test_empty_corpus_returns_empty_results():
    assert semantic_search([], "cat") == []


def test_empty_query_returns_empty_results():
    assert semantic_search(["cat"], " ") == []


def test_ranks_by_cosine_similarity(monkeypatch):
    documents = ["first", "second", "third"]
    embeddings = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
    ])

    monkeypatch.setattr(
        semantic,
        "embed_query",
        lambda query, model=None: np.array([1.0, 0.0]),
    )

    results = semantic_search(
        documents,
        "query",
        embeddings=embeddings,
    )

    assert [doc_id for doc_id, _ in results] == [1, 0, 2]
    assert [score for _, score in results] == pytest.approx(
        [1.0, 0.0, -1.0]
    )


def test_ties_are_broken_by_document_id(monkeypatch):
    monkeypatch.setattr(
        semantic,
        "embed_query",
        lambda query, model=None: np.array([1.0, 0.0]),
    )

    results = semantic_search(
        ["a", "b", "c"],
        "query",
        embeddings=np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
    )

    assert [doc_id for doc_id, _ in results] == [0, 1, 2]


def test_result_limit_and_document_id_alignment(monkeypatch):
    monkeypatch.setattr(
        semantic,
        "embed_query",
        lambda query, model=None: np.array([1.0, 0.0]),
    )

    results = semantic_search(
        ["zero", "best", "second"],
        "query",
        k=2,
        embeddings=np.array([
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.6],
        ]),
    )

    assert [doc_id for doc_id, _ in results] == [1, 2]
    assert len(results) == 2
    assert results[0][1] > results[1][1]


def test_zero_vectors_are_safe(monkeypatch):
    monkeypatch.setattr(
        semantic,
        "embed_query",
        lambda query, model=None: np.array([1.0, 0.0]),
    )

    results = semantic_search(
        ["zero", "aligned"],
        "query",
        embeddings=np.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ]),
    )

    assert results == [(1, 1.0), (0, 0.0)]


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([1.0, 0.0]),
        np.empty((1, 0)),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[np.nan, 0.0]]),
        np.array([[np.inf, 0.0]]),
    ],
)
def test_malformed_document_embeddings_raise(embeddings):
    with pytest.raises(ValueError):
        semantic_search(
            ["only one document"],
            "query",
            embeddings=embeddings,
        )


def test_embedding_model_output_shape_is_validated():
    class BadModel:
        def encode(self, texts, **kwargs):
            return np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        embed_documents(["a", "b"], model=BadModel())


def test_non_finite_model_output_is_rejected():
    class BadModel:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                return np.array([np.nan, 0.0])
            return np.array([[np.nan, 0.0]])

    with pytest.raises(ValueError):
        embed_documents(["a"], model=BadModel())

    with pytest.raises(ValueError):
        embed_query("a", model=BadModel())


def test_query_embedding_must_be_one_vector():
    class BadModel:
        def encode(self, texts, **kwargs):
            return np.ones((2, 3))

    with pytest.raises(ValueError):
        embed_query("query", model=BadModel())


def test_query_and_document_dimensions_must_match(monkeypatch):
    monkeypatch.setattr(
        semantic,
        "embed_query",
        lambda query, model=None: np.array([1.0, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="dimensions must match"):
        semantic_search(
            ["a"],
            "query",
            embeddings=np.array([[1.0, 0.0]]),
        )


def test_repeated_searches_are_deterministic(fake_model):
    documents = ["cat", "feline", "dog"]
    embeddings = embed_documents(documents, model=fake_model)

    first = semantic_search(
        documents, "cat", embeddings=embeddings, model=fake_model
    )
    second = semantic_search(
        documents, "cat", embeddings=embeddings, model=fake_model
    )

    assert first == second


@pytest.mark.skipif(
    os.environ.get("RUN_SEMANTIC_INTEGRATION") != "1",
    reason="Set RUN_SEMANTIC_INTEGRATION=1 to download/load the real model",
)
def test_real_model_semantic_retrieval():
    documents = [
        "Python is a programming language used to build software.",
        "A dog is a domesticated animal that enjoys walks.",
        "Software developers write code and debug applications.",
        "Fresh vegetables are used in healthy cooking.",
    ]

    embeddings = embed_documents(documents)
    results = semantic_search(
        documents,
        "How do I write software code?",
        k=2,
        embeddings=embeddings,
    )

    assert results
    assert results[0][0] in (0, 2)
    assert all(np.isfinite(score) for _, score in results)


# ---------------------------------------------------------------------
# SemanticIndex
# ---------------------------------------------------------------------

def _mock_query_vector(monkeypatch, vector):
    monkeypatch.setattr(
        semantic,
        "embed_query",
        lambda query, model=None: np.array(vector),
    )


def test_index_ranks_by_cosine_similarity(monkeypatch):
    _mock_query_vector(monkeypatch, [1.0, 0.0])

    index = SemanticIndex(
        ["first", "second", "third"],
        embeddings=np.array([
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ]),
    )
    results = index.search("query")

    assert [doc_id for doc_id, _ in results] == [1, 0, 2]
    assert [score for _, score in results] == pytest.approx(
        [1.0, 0.0, -1.0]
    )


def test_index_ties_are_broken_by_document_id(monkeypatch):
    _mock_query_vector(monkeypatch, [1.0, 0.0])

    index = SemanticIndex(
        ["a", "b", "c"],
        embeddings=np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
    )

    assert [doc_id for doc_id, _ in index.search("query")] == [0, 1, 2]


def test_index_result_limit_and_document_id_alignment(monkeypatch):
    _mock_query_vector(monkeypatch, [1.0, 0.0])

    index = SemanticIndex(
        ["zero", "best", "second"],
        embeddings=np.array([
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.6],
        ]),
    )
    results = index.search("query", k=2)

    assert [doc_id for doc_id, _ in results] == [1, 2]
    assert results[0][1] > results[1][1]


def test_index_zero_vectors_are_safe(monkeypatch):
    _mock_query_vector(monkeypatch, [1.0, 0.0])

    index = SemanticIndex(
        ["zero", "aligned"],
        embeddings=np.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ]),
    )

    assert index.search("query") == [(1, 1.0), (0, 0.0)]


def test_index_matches_semantic_search_with_model(fake_model):
    documents = ["cat", "feline", "dog", "opposite", "zero"]

    index = SemanticIndex(documents, model=fake_model)

    for query in ["cat", "dog", "feline", "unknown"]:
        for k in [1, 3, 10]:
            assert index.search(query, k=k) == semantic_search(
                documents, query, k=k, model=fake_model
            )


def test_index_matches_semantic_search_with_precomputed_embeddings(fake_model):
    documents = ["cat", "feline", "dog"]
    embeddings = embed_documents(documents, model=fake_model)

    index = SemanticIndex(documents, embeddings=embeddings, model=fake_model)

    assert index.search("cat") == semantic_search(
        documents, "cat", embeddings=embeddings, model=fake_model
    )


def test_index_unnormalized_embeddings_are_normalized(monkeypatch):
    _mock_query_vector(monkeypatch, [1.0, 0.0])

    index = SemanticIndex(
        ["a", "b"],
        embeddings=np.array([
            [5.0, 0.0],
            [0.0, 3.0],
        ]),
    )

    assert index.search("query") == [(0, 1.0), (1, 0.0)]


def test_index_reuse_gives_identical_results(fake_model):
    index = SemanticIndex(["cat", "feline", "dog"], model=fake_model)

    first = index.search("cat")
    second = index.search("cat")
    other = index.search("dog")
    third = index.search("cat")

    assert first == second == third
    assert other != first


def test_index_len(fake_model):
    assert len(SemanticIndex(["cat", "dog"], model=fake_model)) == 2
    assert len(SemanticIndex([])) == 0


def test_index_empty_corpus_returns_empty_results():
    index = SemanticIndex([])

    assert index.search("cat") == []


def test_index_empty_corpus_ignores_embeddings():
    index = SemanticIndex([], embeddings=np.empty((0, 0)))

    assert index.search("cat") == []


def test_index_empty_query_returns_empty_results(fake_model):
    index = SemanticIndex(["cat"], model=fake_model)

    assert index.search(" ") == []


def test_index_zero_k_returns_empty_results(fake_model):
    index = SemanticIndex(["cat"], model=fake_model)

    assert index.search("cat", k=0) == []


@pytest.mark.parametrize("documents", [None, "cat", [1, "cat"]])
def test_index_invalid_documents_raise(documents):
    with pytest.raises(TypeError):
        SemanticIndex(documents)


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([1.0, 0.0]),
        np.empty((1, 0)),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[np.nan, 0.0]]),
        np.array([[np.inf, 0.0]]),
    ],
)
def test_index_malformed_embeddings_raise_at_construction(embeddings):
    with pytest.raises(ValueError):
        SemanticIndex(["only one document"], embeddings=embeddings)


@pytest.mark.parametrize("k", [-1, 1.5, True, "2"])
def test_index_invalid_k_raises(k, fake_model):
    index = SemanticIndex(["cat"], model=fake_model)

    with pytest.raises(ValueError):
        index.search("cat", k=k)


@pytest.mark.parametrize("query", [None, 123, ["cat"]])
def test_index_invalid_queries_raise(query, fake_model):
    index = SemanticIndex(["cat"], model=fake_model)

    with pytest.raises(TypeError):
        index.search(query)


def test_index_query_and_document_dimensions_must_match(monkeypatch):
    _mock_query_vector(monkeypatch, [1.0, 0.0, 0.0])

    index = SemanticIndex(["a"], embeddings=np.array([[1.0, 0.0]]))

    with pytest.raises(ValueError, match="dimensions must match"):
        index.search("query")


def test_index_is_not_affected_by_later_document_list_changes(fake_model):
    documents = ["cat", "dog"]
    index = SemanticIndex(documents, model=fake_model)

    before = index.search("cat")
    documents.append("feline")

    assert len(index) == 2
    assert index.search("cat") == before


def test_index_does_not_modify_input_embeddings(monkeypatch):
    _mock_query_vector(monkeypatch, [1.0, 0.0])

    embeddings = np.array([[5.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    original = embeddings.copy()

    SemanticIndex(["a", "b"], embeddings=embeddings).search("query")

    assert np.array_equal(embeddings, original)


def test_index_raises_at_construction_where_function_returns_early():
    malformed = np.array([[np.nan, 0.0]])

    # semantic_search returns before looking at the embeddings.
    assert semantic_search(["a"], " ", embeddings=malformed) == []

    # SemanticIndex validates up front instead.
    with pytest.raises(ValueError):
        SemanticIndex(["a"], embeddings=malformed)

