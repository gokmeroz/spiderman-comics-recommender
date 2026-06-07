"""
features.py

Converts raw item metadata into a single text field called item_text.

Why do we need this?
  ML models cannot work directly with a table of strings like "dark" or "Venom".
  They need a numeric representation. Before we can do that, we first need to
  combine all the relevant metadata into one string per item.

  This combined string is called an "engineered feature" — a new piece of
  information we constructed from the raw columns.
"""

import pandas as pd


def build_item_text(items: pd.DataFrame) -> pd.DataFrame:
    """
    Add an item_text column to the items DataFrame.

    item_text is a single string that combines the most useful metadata
    columns for each item. It gives the TF-IDF vectorizer more signal
    to work with than any single column alone.

    For example, item_text for "Spider-Man 2" might look like:
        "Spider-Man 2 movie raimi emotional Doctor Octopus responsibility
         peter parker sacrifice mentor science tragedy responsibility train scene"

    Args:
        items: The raw items DataFrame from load_items().

    Returns:
        A new DataFrame with an added item_text column.
        The original DataFrame is not modified.
    """
    # Work on a copy so we never accidentally mutate the original DataFrame.
    # Mutating input data is a common source of hard-to-find bugs in ML pipelines.
    df = items.copy()

    # These are the columns whose text content is most useful for matching.
    # We skip item_id (just a number) and popularity (a score, not text).
    text_columns = ["title", "medium", "era", "tone", "villain", "theme", "tags"]

    # Fill missing values with an empty string before combining.
    # If we left NaN values in place, joining them would produce the literal
    # string "nan" in item_text, which would add noise to the vectorizer.
    for col in text_columns:
        df[col] = df[col].fillna("")

    # Combine all text columns into one string separated by spaces.
    # str.cat() concatenates strings across columns.
    # sep=" " puts a space between each field.
    df["item_text"] = df[text_columns].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1
    )

    # Normalize whitespace: collapse multiple spaces into one and strip edges.
    # This handles cases where a field was empty and left a double space.
    df["item_text"] = df["item_text"].str.replace(r"\s+", " ", regex=True).str.strip()

    return df
