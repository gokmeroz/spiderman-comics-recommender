"""
test_evaluation.py

Tests for the evaluation metrics (Version 5).

We test:
  1. precision_at_k returns correct value on simple examples.
  2. recall_at_k returns correct value on simple examples.
  3. ndcg_at_k returns 1.0 for a perfect (ideal) ranking.
  4. ndcg_at_k returns a lower value for a worse ranking.
  5. Empty relevant set is handled safely (no division by zero).
  6. evaluate_recommender returns a DataFrame.
  7. evaluate_recommender output has the expected metric columns.
"""

import pytest

from src.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    evaluate_recommender,
    TEST_CASES,
)
from src.data_loader import load_items, load_interactions
from src.content_recommender import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.hybrid_ranker import HybridRecommender


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hybrid_recommender() -> HybridRecommender:
    items = load_items()
    interactions = load_interactions()
    content_rec = ContentBasedRecommender(items)
    collab_rec = CollaborativeFilteringRecommender(items, interactions)
    return HybridRecommender(content_rec, collab_rec)


# ─────────────────────────────────────────────────────────────────────────────
# Precision@K
# ─────────────────────────────────────────────────────────────────────────────

def test_precision_at_k_correct_value():
    """2 hits out of 5 recommendations → Precision@5 = 0.40."""
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 99}  # 99 is relevant but not recommended
    assert precision_at_k(recommended, relevant, k=5) == pytest.approx(2 / 5)


def test_precision_at_k_perfect():
    """All top K items are relevant → Precision@3 = 1.0."""
    recommended = [1, 2, 3, 4, 5]
    relevant = {1, 2, 3}
    assert precision_at_k(recommended, relevant, k=3) == pytest.approx(1.0)


def test_precision_at_k_zero():
    """No hits in top K → Precision@5 = 0.0."""
    recommended = [1, 2, 3, 4, 5]
    relevant = {10, 20}
    assert precision_at_k(recommended, relevant, k=5) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Recall@K
# ─────────────────────────────────────────────────────────────────────────────

def test_recall_at_k_correct_value():
    """2 out of 3 relevant items found in top 5 → Recall@5 = 0.667."""
    recommended = [1, 2, 3, 4, 5]
    relevant = {1, 3, 99}  # 99 not in recommendations
    assert recall_at_k(recommended, relevant, k=5) == pytest.approx(2 / 3)


def test_recall_at_k_perfect():
    """All relevant items found in top K → Recall = 1.0."""
    recommended = [1, 2, 3, 4, 5]
    relevant = {1, 2, 3}
    assert recall_at_k(recommended, relevant, k=5) == pytest.approx(1.0)


def test_recall_at_k_empty_relevant_set():
    """Empty relevant set should return 0.0 without raising an error."""
    recommended = [1, 2, 3]
    assert recall_at_k(recommended, set(), k=3) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# NDCG@K
# ─────────────────────────────────────────────────────────────────────────────

def test_ndcg_at_k_ideal_ranking():
    """
    All relevant items at positions 1, 2, 3 → NDCG@3 = 1.0.

    When DCG equals IDCG, the ranking is perfect.
    """
    recommended = [1, 2, 3, 4, 5]
    relevant = {1, 2, 3}
    assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_worse_ranking():
    """
    Relevant items pushed to positions 3, 4, 5 → NDCG@5 should be < 1.0.

    The same hits but ranked lower should produce a lower NDCG.
    """
    # Perfect ranking: relevant at 1, 2, 3
    perfect = ndcg_at_k([1, 2, 3, 4, 5], {1, 2, 3}, k=5)

    # Worse ranking: relevant items pushed to end
    worse = ndcg_at_k([4, 5, 1, 2, 3], {1, 2, 3}, k=5)

    assert worse < perfect


def test_ndcg_at_k_empty_relevant_set():
    """Empty relevant set should return 0.0 without raising an error."""
    recommended = [1, 2, 3]
    assert ndcg_at_k(recommended, set(), k=3) == pytest.approx(0.0)


def test_ndcg_at_k_no_hits():
    """No relevant items in the top K → NDCG@5 = 0.0."""
    recommended = [1, 2, 3, 4, 5]
    relevant = {10, 20, 30}
    assert ndcg_at_k(recommended, relevant, k=5) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_recommender
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluate_recommender_returns_dataframe(hybrid_recommender):
    """evaluate_recommender should return a pandas DataFrame."""
    import pandas as pd
    result = evaluate_recommender(hybrid_recommender, TEST_CASES, k=5)
    assert isinstance(result, pd.DataFrame)


def test_evaluate_recommender_has_metric_columns(hybrid_recommender):
    """Output must include the three metric columns and recommended_titles."""
    result = evaluate_recommender(hybrid_recommender, TEST_CASES, k=5)
    for col in ["name", "precision_at_5", "recall_at_5", "ndcg_at_5", "recommended_titles"]:
        assert col in result.columns, f"Missing column: {col}"


def test_evaluate_recommender_one_row_per_test_case(hybrid_recommender):
    """Output should have exactly one row per test case."""
    result = evaluate_recommender(hybrid_recommender, TEST_CASES, k=5)
    assert len(result) == len(TEST_CASES)
