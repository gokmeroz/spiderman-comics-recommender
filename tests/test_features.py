"""
test_features.py

Tests for src/features.py.
"""

import pandas as pd
from src.data_loader import load_items
from src.features import build_item_text


class TestBuildItemText:

    def test_item_text_column_is_created(self):
        """build_item_text() must add an item_text column."""
        items = load_items()
        result = build_item_text(items)
        assert "item_text" in result.columns

    def test_item_text_contains_title(self):
        """item_text should include the item's title."""
        items = load_items()
        result = build_item_text(items)
        # Check a few rows — the title should appear in the combined text.
        for _, row in result.iterrows():
            assert row["title"].split()[0].lower() in row["item_text"].lower(), (
                f"Title '{row['title']}' not found in item_text: {row['item_text']}"
            )

    def test_item_text_contains_tone(self):
        """item_text should include the tone field."""
        items = load_items()
        result = build_item_text(items)
        first_row = result.iloc[0]
        assert first_row["tone"].lower() in first_row["item_text"].lower()

    def test_item_text_contains_villain(self):
        """item_text should include the villain field."""
        items = load_items()
        result = build_item_text(items)
        # Find the row for "Kraven's Last Hunt" and check the villain is present.
        kraven_row = result[result["title"] == "Kraven's Last Hunt"].iloc[0]
        assert "kraven" in kraven_row["item_text"].lower()

    def test_item_text_contains_tags(self):
        """item_text should include content from the tags field."""
        items = load_items()
        result = build_item_text(items)
        # "Spider-Man: Into the Spider-Verse" has tags containing "multiverse".
        verse_row = result[result["title"] == "Spider-Man: Into the Spider-Verse"].iloc[0]
        assert "multiverse" in verse_row["item_text"].lower()

    def test_original_dataframe_is_not_modified(self):
        """build_item_text() must not modify the original DataFrame."""
        items = load_items()
        original_columns = list(items.columns)
        build_item_text(items)
        # The original should still have the same columns — no item_text added.
        assert list(items.columns) == original_columns

    def test_no_nan_strings_in_item_text(self):
        """item_text should not contain the literal string 'nan'."""
        items = load_items()
        result = build_item_text(items)
        for _, row in result.iterrows():
            assert "nan" not in row["item_text"].lower(), (
                f"Found 'nan' in item_text for '{row['title']}': {row['item_text']}"
            )
