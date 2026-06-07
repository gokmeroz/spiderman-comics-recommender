"""
data_loader.py

Responsible for loading raw CSV data into pandas DataFrames.

In ML projects, having a dedicated data loader is important because:
- It gives you one central place to validate your data.
- It prevents silent bugs caused by missing or renamed columns.
- It makes the rest of the code assume the data is already clean.
"""

from pathlib import Path

import pandas as pd


# Path() gives us cross-platform file paths (works on Mac, Windows, Linux).
# __file__ refers to this file (data_loader.py).
# .resolve() converts it to an absolute path (no relative path ambiguity).
# .parents[1] goes two levels up: src/ -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw data lives here. We never modify files in data/raw/ directly.
# Raw data is treated like a source of truth — read only.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# These are the columns we expect to find in items.csv.
# If any are missing, something went wrong with the file.
REQUIRED_ITEM_COLUMNS = [
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

# These are the columns we expect to find in interactions.csv.
REQUIRED_INTERACTION_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
]


def load_items() -> pd.DataFrame:
    """
    Load the items dataset from data/raw/items.csv.

    Each row in this file is one Spider-Man media item:
    a movie, game, comic, animated film, or show.

    Returns:
        A pandas DataFrame with one row per item.

    Raises:
        FileNotFoundError: If items.csv does not exist.
        ValueError: If required columns are missing.
    """
    items_path = RAW_DATA_DIR / "items.csv"

    # Read the CSV file into a DataFrame.
    # A DataFrame is like a spreadsheet in memory — rows and columns.
    items = pd.read_csv(items_path)

    # Validate that the file has all the columns we expect.
    # In ML, missing columns cause silent errors that are hard to debug later.
    _validate_columns(items, REQUIRED_ITEM_COLUMNS, "items.csv")

    return items


def load_interactions() -> pd.DataFrame:
    """
    Load the interactions dataset from data/raw/interactions.csv.

    Each row represents one user's rating of one item.
    This is the behavioral data that collaborative filtering will use.

    Rating scale:
        1 = strongly dislike
        2 = dislike
        3 = neutral
        4 = like
        5 = love

    Returns:
        A pandas DataFrame with one row per user-item rating.

    Raises:
        FileNotFoundError: If interactions.csv does not exist.
        ValueError: If required columns are missing.
    """
    interactions_path = RAW_DATA_DIR / "interactions.csv"

    interactions = pd.read_csv(interactions_path)

    _validate_columns(interactions, REQUIRED_INTERACTION_COLUMNS, "interactions.csv")

    return interactions


def _validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    filename: str,
) -> None:
    """
    Check that a DataFrame contains all expected columns.

    We do this before any ML processing begins because:
    - A missing column might not cause an error right away.
    - But it will silently produce wrong results or crash much later.
    - Catching it early makes debugging much easier.

    Args:
        df: The loaded DataFrame to check.
        required_columns: The list of column names we need.
        filename: The source file name (used in the error message).

    Raises:
        ValueError: If any required column is missing.
    """
    # set() lets us find columns that are in required_columns but not in the DataFrame.
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"File '{filename}' is missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns.tolist())}"
        )
