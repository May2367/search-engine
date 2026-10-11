import math

import pytest

from benchmarks.evaluate_retrieval import (
    DEFAULT_DATA_PATH,
    dcg_at_k,
    load_documents,
    ndcg_at_k,
    recall_at_k,
)


def test_dcg_at_k_calculates_discounted_gain():
    expected = 7.0 + 3.0 / math.log2(3)

    assert dcg_at_k([3, 2], k=2) == pytest.approx(expected)


def test_dcg_at_k_returns_zero_for_empty_results():
    assert dcg_at_k([], k=10) == 0.0


def test_dcg_at_k_returns_zero_when_k_is_zero():
    assert dcg_at_k([3, 2, 1], k=0) == 0.0


def test_ndcg_is_one_for_ideal_ranking():
    relevance = {10: 2, 11: 1, 12: 0}

    score = ndcg_at_k([10, 11, 12], relevance, k=3)

    assert score == pytest.approx(1.0)


def test_ndcg_rewards_better_ranked_documents():
    relevance = {10: 2, 11: 1}

    ideal = ndcg_at_k([10, 11], relevance, k=2)
    reversed_ranking = ndcg_at_k([11, 10], relevance, k=2)

    assert ideal == pytest.approx(1.0)
    assert reversed_ranking < ideal
    assert reversed_ranking >= 0.0


def test_ndcg_handles_irrelevant_results():
    relevance = {10: 2, 11: 1}

    score = ndcg_at_k([20, 21], relevance, k=2)

    assert score == 0.0


def test_ndcg_handles_no_relevant_documents():
    assert ndcg_at_k([1, 2], {}, k=2) == 0.0


def test_ndcg_returns_zero_for_k_zero():
    assert ndcg_at_k([10, 11], {10: 2}, k=0) == 0.0


def test_recall_at_k_calculates_fraction_retrieved():
    relevance = {10: 2, 11: 1, 12: 2, 13: 0}

    score = recall_at_k([10, 13], relevance, k=2)

    assert score == pytest.approx(1 / 3)


def test_recall_at_k_is_one_when_all_relevant_docs_are_retrieved():
    relevance = {10: 2, 11: 1}

    assert recall_at_k([10, 11, 20], relevance, k=3) == 1.0


def test_recall_at_k_returns_zero_for_empty_results():
    assert recall_at_k([], {10: 2}, k=10) == 0.0


def test_recall_at_k_handles_no_relevant_documents():
    assert recall_at_k([1, 2], {}, k=2) == 0.0


@pytest.mark.parametrize("k", [-1, 1.5, True, "10"])
def test_metrics_reject_invalid_k(k):
    with pytest.raises(ValueError):
        ndcg_at_k([], {}, k=k)

    with pytest.raises(ValueError):
        recall_at_k([], {}, k=k)


def test_dcg_rejects_invalid_relevance_grades():
    with pytest.raises(ValueError):
        dcg_at_k([-1], k=1)

    with pytest.raises(ValueError):
        dcg_at_k([float("nan")], k=1)


def test_evaluation_dataset_has_90_nonempty_documents():
    documents = load_documents(DEFAULT_DATA_PATH)

    assert len(documents) == 90
    assert all(document.strip() for document in documents)