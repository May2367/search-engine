from engine.index import build_index
from engine.search import (
    brute_force_search,
    single_term_basic_index_search,
    and_index_search,
    or_index_search,
)


def test_brute_force_search():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    assert brute_force_search(documents, "python") == [0, 1]
    assert brute_force_search(documents, "java") == [0, 2]


def test_brute_force_search_missing_term():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    assert brute_force_search(documents, "javascript") == []


def test_brute_force_search_duplicate_terms():
    documents = [
        "python python python",
        "java",
    ]

    assert brute_force_search(documents, "python") == [0]


def test_single_term_index_search():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    index = build_index(documents)

    assert set(single_term_basic_index_search(index, "python")) == {0, 1}
    assert set(single_term_basic_index_search(index, "java")) == {0, 2}


def test_single_term_index_search_missing_term():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    index = build_index(documents)

    assert single_term_basic_index_search(index, "javascript") == []


def test_and_search():
    documents = [
        "python java",
        "python rust",
        "java rust",
        "python java rust",
    ]

    index = build_index(documents)

    result = and_index_search(index, ["python", "java"])

    assert set(result) == {0, 3}


def test_and_search_multiple_terms():
    documents = [
        "python java rust",
        "python java",
        "python rust",
        "java rust",
    ]

    index = build_index(documents)

    result = and_index_search(index, ["python", "java", "rust"])

    assert set(result) == {0}


def test_and_search_missing_term():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    index = build_index(documents)

    assert and_index_search(index, ["python", "javascript"]) == []


def test_and_search_empty_query():
    documents = [
        "python java",
        "python rust",
    ]

    index = build_index(documents)

    assert and_index_search(index, []) == []


def test_and_search_duplicate_terms():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    index = build_index(documents)

    result = and_index_search(index, ["python", "python"])

    assert set(result) == {0, 1}


def test_or_search():
    documents = [
        "python java",
        "python rust",
        "java",
        "rust",
    ]

    index = build_index(documents)

    result = or_index_search(index, ["python", "java"])

    assert set(result) == {0, 1, 2}


def test_or_search_multiple_terms():
    documents = [
        "python",
        "java",
        "rust",
        "go",
    ]

    index = build_index(documents)

    result = or_index_search(index, ["python", "java", "rust"])

    assert set(result) == {0, 1, 2}


def test_or_search_missing_term():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    index = build_index(documents)

    assert or_index_search(index, ["javascript"]) == []


def test_or_search_mixed_existing_and_missing_terms():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    index = build_index(documents)

    result = or_index_search(index, ["python", "javascript"])

    assert set(result) == {0, 1}


def test_or_search_empty_query():
    documents = [
        "python java",
        "python rust",
    ]

    index = build_index(documents)

    assert or_index_search(index, []) == []


def test_or_search_duplicate_terms():
    documents = [
        "python java",
        "python rust",
        "java",
    ]

    index = build_index(documents)

    result = or_index_search(index, ["python", "python"])

    assert set(result) == {0, 1}


def test_index_search_matches_brute_force():
    documents = [
        "python java",
        "python rust",
        "java",
        "rust python",
        "go",
    ]

    index = build_index(documents)

    terms = [
        "python",
        "java",
        "rust",
        "go",
        "javascript",
    ]

    for term in terms:
        brute_force_results = set(
            brute_force_search(documents, term)
        )

        index_results = set(
            single_term_basic_index_search(index, term)
        )

        assert index_results == brute_force_results
