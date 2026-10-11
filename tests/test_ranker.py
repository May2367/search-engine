import math

import pytest
from rank_bm25 import BM25Okapi

from engine.index import build_index
from engine.ranker import bm25_search
from engine.tokenizer import tokenizer


DOCUMENTS = [
    "cat sat on mat",
    "cat cat sat",
    "dog ran in park",
    "cat slept peacefully",
    "",
]


class PositiveIDFBM25(BM25Okapi):
    """Reference implementation using our ranker's IDF formula."""

    def _calc_idf(self, nd):
        document_frequency = {}

        for document in self.doc_freqs:
            for term in document:
                document_frequency[term] = (
                    document_frequency.get(term, 0) + 1
                )

        self.idf = {
            term: math.log(
                1 + (self.corpus_size - df + 0.5) / (df + 0.5)
            )
            for term, df in document_frequency.items()
        }


@pytest.fixture
def index_data():
    return build_index(DOCUMENTS)


def test_scores_match_rank_bm25(index_data):
    reference = PositiveIDFBM25(
        [tokenizer(document) for document in DOCUMENTS],
        k1=1.5,
        b=0.75,
    )
    expected = reference.get_scores(tokenizer("cat sat"))
    actual = dict(bm25_search(index_data, "cat sat"))

    assert actual.keys() == {
        doc_id
        for doc_id, score in enumerate(expected)
        if score > 0
    }

    for doc_id, score in actual.items():
        assert score == pytest.approx(expected[doc_id])


def test_results_are_sorted_by_descending_score(index_data):
    results = bm25_search(index_data, "cat")
    scores = [score for _, score in results]

    assert scores == sorted(scores, reverse=True)
    assert {doc_id for doc_id, _ in results} == {0, 1, 3}


def test_ties_are_broken_by_document_id():
    data = build_index(["cat", "cat"])

    assert bm25_search(data, "cat") == [
        (0, pytest.approx(bm25_search(data, "cat")[0][1])),
        (1, pytest.approx(bm25_search(data, "cat")[1][1])),
    ]


def test_repeated_query_terms_repeat_contributions(index_data):
    single = dict(bm25_search(index_data, "cat"))
    repeated = dict(bm25_search(index_data, "cat cat"))

    for doc_id in single:
        assert repeated[doc_id] == pytest.approx(2 * single[doc_id])


def test_query_is_tokenized(index_data):
    assert bm25_search(index_data, "CAT!!!") == bm25_search(
        index_data, "cat"
    )


@pytest.mark.parametrize("query", ["", " ", "!!!", "123"])
def test_empty_or_tokenless_query(query, index_data):
    assert bm25_search(index_data, query) == []


def test_missing_term_returns_no_results(index_data):
    assert bm25_search(index_data, "nonexistent") == []


def test_missing_terms_are_ignored(index_data):
    assert bm25_search(
        index_data, "cat nonexistent"
    ) == bm25_search(index_data, "cat")


def test_empty_corpus_returns_no_results():
    assert bm25_search(build_index([]), "cat") == []


def test_corpus_with_only_empty_documents_returns_no_results():
    assert bm25_search(build_index(["", "!!!"]), "cat") == []


def test_empty_documents_are_not_returned(index_data):
    results = bm25_search(index_data, "cat")

    assert all(doc_id != 4 for doc_id, _ in results)


@pytest.mark.parametrize(
    ("k1", "b"),
    [
        (-1, 0.75),
        (1.5, -0.1),
        (1.5, 1.1),
        (float("nan"), 0.75),
        (1.5, float("inf")),
        ("invalid", 0.75),
        (True, 0.75),
        (1.5, False),
    ],
)
def test_invalid_parameters_raise_value_error(k1, b, index_data):
    with pytest.raises(ValueError):
        bm25_search(index_data, "cat", k1=k1, b=b)


@pytest.mark.parametrize("k1", [0, 1.5, 3])
@pytest.mark.parametrize("b", [0, 0.75, 1])
def test_valid_parameters(k1, b, index_data):
    results = bm25_search(index_data, "cat", k1=k1, b=b)

    assert results
    assert all(math.isfinite(score) for _, score in results)


def test_missing_index_keys_raise():
    with pytest.raises(KeyError):
        bm25_search({}, "cat")


def test_missing_corpus_statistics_raise():
    with pytest.raises(KeyError):
        bm25_search({"index": {}}, "cat")


def test_incomplete_corpus_statistics_raise():
    with pytest.raises(KeyError):
        bm25_search(
            {"index": {}, "corpus_stats": {}},
            "cat",
        )


def test_corpus_statistics():
    data = build_index(["cat cat dog", "dog", ""])
    stats = data["corpus_stats"]

    assert stats["num_docs"] == 3
    assert stats["doc_freq"] == {"cat": 1, "dog": 2}
    assert stats["doc_lengths"] == {0: 3, 1: 1, 2: 0}
    assert stats["avg_doc_length"] == pytest.approx(4 / 3)


def test_single_document_corpus():
    data = build_index(["cat cat dog"])
    results = bm25_search(data, "cat")

    assert len(results) == 1
    assert results[0][0] == 0
    assert results[0][1] > 0
