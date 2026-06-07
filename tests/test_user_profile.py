"""
test_user_profile.py

Tests for the UserProfileRecommender (Version 2).

We test:
  1. That liked titles return recommendations.
  2. That already-liked titles are excluded from results.
  3. That disliked titles are excluded from results.
  4. That an unknown title raises a clear ValueError.
  5. That the output contains the profile_similarity_score column.
  6. That liking Miles-related items surfaces similar modern/multiverse items.
"""

import pytest

from src.data_loader import load_items
from src.content_recommender import ContentBasedRecommender
from src.user_profile import UserProfileRecommender


# Build the recommenders once and reuse across all tests.
# This avoids re-fitting TF-IDF for every single test, which is slow.
@pytest.fixture(scope="module")
def recommender() -> UserProfileRecommender:
    items = load_items()
    content_rec = ContentBasedRecommender(items)
    return UserProfileRecommender(content_rec)


def test_liked_titles_return_results(recommender):
    """Liking one or more titles should return a non-empty DataFrame."""
    results = recommender.recommend_by_user_profile(
        liked_titles=["Spider-Man 2", "Spider-Man: Into the Spider-Verse"],
        top_k=5,
    )
    assert len(results) > 0


def test_liked_titles_are_excluded(recommender):
    """Liked titles must not appear in the recommendations."""
    liked = ["Spider-Man 2", "Spider-Man: Into the Spider-Verse"]
    results = recommender.recommend_by_user_profile(liked_titles=liked, top_k=10)
    returned_titles = results["title"].tolist()
    for title in liked:
        assert title not in returned_titles, (
            f"Liked title '{title}' should not appear in recommendations."
        )


def test_disliked_titles_are_excluded(recommender):
    """Disliked titles must not appear in the recommendations."""
    liked = ["Spider-Man 2"]
    disliked = ["Spider-Man: Web of Shadows", "Venom"]
    results = recommender.recommend_by_user_profile(
        liked_titles=liked,
        disliked_titles=disliked,
        top_k=10,
    )
    returned_titles = results["title"].tolist()
    for title in disliked:
        assert title not in returned_titles, (
            f"Disliked title '{title}' should not appear in recommendations."
        )


def test_unknown_title_raises_value_error(recommender):
    """Passing a title that doesn't exist in the catalog should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        recommender.recommend_by_user_profile(
            liked_titles=["This Title Does Not Exist"]
        )


def test_output_has_profile_similarity_score(recommender):
    """Results must include the profile_similarity_score column."""
    results = recommender.recommend_by_user_profile(
        liked_titles=["Spider-Man: Into the Spider-Verse"],
        top_k=5,
    )
    assert "profile_similarity_score" in results.columns


def test_output_has_explanation(recommender):
    """Results must include the explanation column."""
    results = recommender.recommend_by_user_profile(
        liked_titles=["Spider-Man: Into the Spider-Verse"],
        top_k=5,
    )
    assert "explanation" in results.columns
    assert results["explanation"].notna().all()


def test_miles_fan_gets_similar_recommendations(recommender):
    """
    Liking Miles-related items should surface other modern/multiverse/identity items.

    We check that at least one result contains relevant keywords in its tags,
    theme, or tone — indicating the profile vector is pointing in the right direction.
    """
    results = recommender.recommend_by_user_profile(
        liked_titles=[
            "Spider-Man: Into the Spider-Verse",
            "Spider-Man: Miles Morales",
        ],
        top_k=10,
    )
    # Collect all text content from returned items to check for thematic overlap.
    # Note: output only includes the columns defined in recommend_by_user_profile.
    all_text = " ".join([
        " ".join(results["tone"].fillna("").tolist()),
        " ".join(results["villain"].fillna("").tolist()),
        " ".join(results["explanation"].fillna("").tolist()),
    ]).lower()

    relevant_keywords = ["multiverse", "identity", "modern", "emotional", "animation"]
    matched = [kw for kw in relevant_keywords if kw in all_text]
    assert len(matched) > 0, (
        f"Expected at least one of {relevant_keywords} in recommendations, "
        f"but found none. Returned titles: {results['title'].tolist()}"
    )
