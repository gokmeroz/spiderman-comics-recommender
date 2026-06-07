"""
test_content_recommender.py

Tests for src/content_recommender.py.
"""

import pandas as pd
import pytest
from src.data_loader import load_items
from src.content_recommender import ContentBasedRecommender


VENOM_QUERY = "dark venom symbiote game"
MILES_QUERY = "miles morales multiverse animation identity"


class TestContentBasedRecommender:

    def setup_method(self):
        """Load data and build the recommender once per test."""
        items = load_items()
        self.recommender = ContentBasedRecommender(items)

    def test_query_returns_results(self):
        """A query should return at least one recommendation."""
        results = self.recommender.recommend_by_query(VENOM_QUERY)
        assert len(results) > 0

    def test_top_k_is_respected(self):
        """top_k should control the number of returned results."""
        results = self.recommender.recommend_by_query(VENOM_QUERY, top_k=3)
        assert len(results) == 3

    def test_venom_query_returns_venom_related_item(self):
        """
        Querying for dark venom symbiote should surface at least one
        item that is actually about Venom or symbiotes.
        """
        results = self.recommender.recommend_by_query(VENOM_QUERY, top_k=5)
        # Check that at least one top result has Venom as villain or in tags.
        venom_related = results[
            results["villain"].str.lower().str.contains("venom", na=False) |
            results["title"].str.lower().str.contains("venom", na=False)
        ]
        assert len(venom_related) > 0, (
            f"No Venom-related items in top 5 for query '{VENOM_QUERY}'.\n"
            f"Got: {results['title'].tolist()}"
        )

    def test_output_has_similarity_score_column(self):
        """Results must include a similarity_score column."""
        results = self.recommender.recommend_by_query(VENOM_QUERY)
        assert "similarity_score" in results.columns

    def test_output_has_explanation_column(self):
        """Results must include an explanation column."""
        results = self.recommender.recommend_by_query(VENOM_QUERY)
        assert "explanation" in results.columns

    def test_output_has_required_columns(self):
        """Results must include all expected output columns."""
        results = self.recommender.recommend_by_query(VENOM_QUERY)
        expected = ["item_id", "title", "medium", "tone", "villain",
                    "similarity_score", "explanation"]
        for col in expected:
            assert col in results.columns, f"Missing column: {col}"

    def test_results_are_sorted_by_similarity_descending(self):
        """Highest similarity score should appear first."""
        results = self.recommender.recommend_by_query(VENOM_QUERY, top_k=10)
        scores = results["similarity_score"].tolist()
        assert scores == sorted(scores, reverse=True), (
            "Results are not sorted by similarity_score descending."
        )

    def test_similarity_scores_are_between_0_and_1(self):
        """Cosine similarity values must be in [0, 1]."""
        results = self.recommender.recommend_by_query(VENOM_QUERY, top_k=10)
        assert results["similarity_score"].between(0, 1).all()

    def test_explanations_are_non_empty_strings(self):
        """Every explanation should be a non-empty string."""
        results = self.recommender.recommend_by_query(MILES_QUERY)
        for explanation in results["explanation"]:
            assert isinstance(explanation, str) and len(explanation) > 0
