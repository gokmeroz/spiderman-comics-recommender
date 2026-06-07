"""
test_data_loader.py

Tests for src/data_loader.py.

These tests verify that our raw data files load correctly
and contain the structure the rest of the project depends on.

In ML, testing data loading is important because:
- Wrong data in means wrong predictions out.
- Catching schema problems early saves debugging time later.
"""

import pytest
import pandas as pd

from src.data_loader import load_items, load_interactions


class TestLoadItems:
    """Tests for the load_items() function."""

    def test_items_loads_successfully(self):
        """items.csv should load without errors."""
        items = load_items()
        # The result should be a DataFrame (not None, not a list, etc.)
        assert isinstance(items, pd.DataFrame)

    def test_items_has_required_columns(self):
        """items.csv must contain all required columns."""
        items = load_items()
        required = [
            "item_id",
            "title",
            "medium",
            "era",
            "tone",
            "villain",
            "theme",
            "tags",
            "popularity",
        ]
        for column in required:
            assert column in items.columns, f"Missing column: {column}"

    def test_items_has_at_least_20_rows(self):
        """
        The dataset needs at least 20 items to have enough variety
        for meaningful recommendations.
        """
        items = load_items()
        assert len(items) >= 20, f"Expected at least 20 items, got {len(items)}"

    def test_items_has_no_null_item_ids(self):
        """Every item must have an ID — this is our primary key."""
        items = load_items()
        assert items["item_id"].notnull().all(), "Some item_ids are null"

    def test_items_has_no_null_titles(self):
        """Every item must have a title — it's used in explanations and output."""
        items = load_items()
        assert items["title"].notnull().all(), "Some titles are null"


class TestLoadInteractions:
    """Tests for the load_interactions() function."""

    def test_interactions_loads_successfully(self):
        """interactions.csv should load without errors."""
        interactions = load_interactions()
        assert isinstance(interactions, pd.DataFrame)

    def test_interactions_has_required_columns(self):
        """interactions.csv must contain all required columns."""
        interactions = load_interactions()
        required = ["user_id", "item_id", "rating"]
        for column in required:
            assert column in interactions.columns, f"Missing column: {column}"

    def test_interactions_has_at_least_8_unique_users(self):
        """
        We need at least 8 different synthetic users to simulate
        meaningful collaborative filtering behavior.
        """
        interactions = load_interactions()
        unique_users = interactions["user_id"].nunique()
        assert unique_users >= 8, (
            f"Expected at least 8 unique users, got {unique_users}"
        )

    def test_interactions_ratings_are_in_valid_range(self):
        """
        Ratings must be between 1 and 5.
        Values outside this range suggest a data entry problem.
        """
        interactions = load_interactions()
        assert interactions["rating"].between(1, 5).all(), (
            "Some ratings are outside the valid range of 1–5"
        )

    def test_interactions_has_no_null_values(self):
        """Every row must have a user_id, item_id, and rating."""
        interactions = load_interactions()
        assert interactions.notnull().all().all(), (
            "interactions.csv contains null values"
        )
