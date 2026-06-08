"""
test_collaborative_filtering.py

Tests for the CollaborativeFilteringRecommender (Version 3).

We test:
  1. The user-item matrix is built correctly.
  2. A known user gets recommendations.
  3. An unknown user raises ValueError.
  4. Already-rated items are excluded from results.
  5. get_similar_users does not return the target user.
  6. Output includes collaborative_score.
  7. Output includes similar_users.
"""

import pytest
import pandas as pd

from src.data_loader import load_items, load_interactions
from src.collaborative_filtering import CollaborativeFilteringRecommender


@pytest.fixture(scope="module")
def recommender() -> CollaborativeFilteringRecommender:
    items = load_items()
    interactions = load_interactions()
    return CollaborativeFilteringRecommender(items, interactions)


def test_user_item_matrix_is_built(recommender):
    """The user-item matrix should have users as rows and items as columns."""
    matrix = recommender._user_item_matrix
    assert matrix.shape[0] > 0, "Matrix should have at least one user row."
    assert matrix.shape[1] > 0, "Matrix should have at least one item column."


def test_known_user_gets_recommendations(recommender):
    """A user who exists in interactions should get a non-empty result."""
    results = recommender.recommend_by_similar_users("dark_venom_fan", top_k=5)
    assert len(results) > 0


def test_unknown_user_raises_value_error(recommender):
    """A user not in interactions should raise a clear ValueError."""
    with pytest.raises(ValueError, match="not found"):
        recommender.recommend_by_similar_users("ghost_user_99")


def test_already_rated_items_are_excluded(recommender):
    """Items the target user already rated must not appear in recommendations."""
    interactions = load_interactions()
    user_id = "dark_venom_fan"

    rated_item_ids = set(
        interactions[interactions["user_id"] == user_id]["item_id"].tolist()
    )
    results = recommender.recommend_by_similar_users(user_id, top_k=10)
    returned_ids = set(results["item_id"].tolist())

    overlap = rated_item_ids & returned_ids
    assert len(overlap) == 0, (
        f"Already-rated item IDs {overlap} should not appear in recommendations."
    )


def test_get_similar_users_excludes_target(recommender):
    """get_similar_users must not include the target user in results."""
    user_id = "gamer"
    similar = recommender.get_similar_users(user_id, top_n=3)
    assert user_id not in similar["user_id"].tolist(), (
        f"Target user '{user_id}' should not appear in their own similar-users list."
    )


def test_output_has_collaborative_score(recommender):
    """Results must include the collaborative_score column."""
    results = recommender.recommend_by_similar_users("miles_fan", top_k=5)
    assert "collaborative_score" in results.columns


def test_output_has_similar_users(recommender):
    """Results must include the similar_users column."""
    results = recommender.recommend_by_similar_users("miles_fan", top_k=5)
    assert "similar_users" in results.columns
    # Each row's similar_users should be a non-empty list.
    assert all(len(row) > 0 for row in results["similar_users"])
