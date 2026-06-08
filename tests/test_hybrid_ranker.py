"""
test_hybrid_ranker.py

Tests for the HybridRecommender (Version 4).

We test:
  1. Query-only recommendations work.
  2. User-id recommendations work.
  3. Liked/disliked recommendations work.
  4. Output contains all required score columns.
  5. Final score is between 0 and 1.
  6. Results are sorted by final_score descending.
  7. Already-rated or excluded items do not appear.
"""

import pytest
import pandas as pd

from src.data_loader import load_items, load_interactions
from src.content_recommender import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.hybrid_ranker import HybridRecommender


@pytest.fixture(scope="module")
def recommender() -> HybridRecommender:
    items = load_items()
    interactions = load_interactions()
    content_rec = ContentBasedRecommender(items)
    collab_rec = CollaborativeFilteringRecommender(items, interactions)
    return HybridRecommender(content_rec, collab_rec)


def test_query_only_returns_results(recommender):
    """Mode 1: query-only should return a non-empty DataFrame."""
    results = recommender.recommend(query="dark venom symbiote game", top_k=5)
    assert len(results) > 0


def test_user_id_returns_results(recommender):
    """Mode 3: known user_id should return a non-empty DataFrame."""
    results = recommender.recommend(user_id="dark_venom_fan", top_k=5)
    assert len(results) > 0


def test_liked_disliked_returns_results(recommender):
    """Mode 2: liked/disliked titles should return a non-empty DataFrame."""
    results = recommender.recommend(
        liked_titles=["Spider-Man 2", "Spider-Man: Into the Spider-Verse"],
        disliked_titles=["Spider-Man: Web of Shadows"],
        top_k=5,
    )
    assert len(results) > 0


def test_output_has_all_score_columns(recommender):
    """All required score columns must be present in the output."""
    results = recommender.recommend(query="emotional peter parker sacrifice", top_k=5)
    for col in ["content_score", "collaborative_score", "popularity_score", "final_score"]:
        assert col in results.columns, f"Missing column: {col}"


def test_final_score_is_between_0_and_1(recommender):
    """final_score must be in [0, 1] for all rows."""
    results = recommender.recommend(query="miles morales multiverse", top_k=10)
    assert (results["final_score"] >= 0).all()
    assert (results["final_score"] <= 1).all()


def test_results_sorted_by_final_score_descending(recommender):
    """Rows must be ordered from highest to lowest final_score."""
    results = recommender.recommend(query="dark venom symbiote", top_k=10)
    scores = results["final_score"].tolist()
    assert scores == sorted(scores, reverse=True), (
        "Results are not sorted by final_score descending."
    )


def test_liked_titles_excluded_from_results(recommender):
    """Liked titles must not appear in the output."""
    liked = ["Spider-Man 2", "Spider-Man: Into the Spider-Verse"]
    results = recommender.recommend(liked_titles=liked, top_k=10)
    returned_titles = results["title"].tolist()
    for title in liked:
        assert title not in returned_titles


def test_rated_items_excluded_for_known_user(recommender):
    """Items already rated by a known user must not appear in recommendations."""
    interactions = load_interactions()
    user_id = "gamer"
    rated_ids = set(interactions[interactions["user_id"] == user_id]["item_id"])

    results = recommender.recommend(user_id=user_id, top_k=15)
    returned_ids = set(results["item_id"].tolist())

    overlap = rated_ids & returned_ids
    assert len(overlap) == 0, f"Already-rated items appeared: {overlap}"


def test_no_input_raises_value_error(recommender):
    """Calling recommend with no inputs should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Provide at least one"):
        recommender.recommend()
