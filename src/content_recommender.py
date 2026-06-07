"""
content_recommender.py

Implements a content-based recommender using TF-IDF and cosine similarity.

What is content-based recommendation?
  We look at the *content* (metadata) of items — title, tone, villain, tags, etc.
  We convert that text into numbers (vectors), then find items whose vectors
  are closest to the user's query vector.

  This is called "content-based" because we never use other users' behavior.
  We only use the item's own properties to decide how relevant it is.

Why is this useful?
  It solves the cold-start problem: even a brand-new user with no history
  can get recommendations by describing what they want in natural language.

What is TF-IDF?
  TF-IDF stands for Term Frequency–Inverse Document Frequency.
  - TF (Term Frequency): how often a word appears in one item's text.
  - IDF (Inverse Document Frequency): how rare that word is across all items.
  A high TF-IDF score means the word is important for THIS item but not
  too common everywhere — so it's actually distinctive and useful.

What is cosine similarity?
  A way to measure how similar two vectors are, based on the angle between them.
  Two vectors pointing in the same direction → similarity of 1 (identical).
  Two vectors pointing in opposite directions → similarity of 0 (unrelated).
  It does not care about the length of the vectors, only their direction.
  This matters because one item might have more text than another.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.features import build_item_text
from src.explain import explain_content_match


class ContentBasedRecommender:
    """
    Recommends Spider-Man media based on a text query.

    How it works:
      1. At init time: convert all item metadata into TF-IDF vectors.
      2. At query time: convert the query into a TF-IDF vector using the
         SAME vectorizer (same vocabulary, same weights).
      3. Compute cosine similarity between the query vector and every item vector.
      4. Return the top K most similar items.
    """

    def __init__(self, items: pd.DataFrame) -> None:
        """
        Build the TF-IDF item matrix at initialization time.

        We do this once and reuse it for every query. Building the matrix
        is the expensive step; querying it is fast.

        Args:
            items: The items DataFrame from load_items().
        """
        # Build the item_text feature: combine metadata columns into one string.
        self._items = build_item_text(items)

        # TfidfVectorizer converts text into TF-IDF vectors.
        # stop_words="english" removes common English words like "the", "is", "a"
        # that carry no useful meaning for similarity matching.
        self._vectorizer = TfidfVectorizer(stop_words="english")

        # Fit the vectorizer on all item texts.
        # "Fitting" means: learn the vocabulary and compute IDF weights.
        # The result is a matrix where:
        #   - each row = one item
        #   - each column = one word in the vocabulary
        #   - each value = the TF-IDF score of that word for that item
        self._item_matrix = self._vectorizer.fit_transform(self._items["item_text"])

    def recommend_by_query(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Return the top K items most similar to a text query.

        Args:
            query: A free-text description of what the user wants.
                   Example: "dark venom symbiote game"
            top_k: How many recommendations to return.

        Returns:
            A DataFrame with columns:
            item_id, title, medium, tone, villain, similarity_score, explanation
        """
        # Transform the query into a TF-IDF vector.
        # IMPORTANT: we use transform(), not fit_transform().
        # fit_transform() would learn a NEW vocabulary from just the query.
        # transform() reuses the vocabulary learned from all items.
        # If the vectorizer doesn't know a query word, it silently ignores it.
        query_vector = self._vectorizer.transform([query])

        # Compute cosine similarity between the query vector and every item vector.
        # cosine_similarity returns a 2D array: shape (1, num_items).
        # We flatten it to a 1D array with .flatten().
        similarity_scores = cosine_similarity(query_vector, self._item_matrix).flatten()

        # Get the indices that would sort similarity_scores from highest to lowest.
        # [::-1] reverses the order so highest similarity comes first.
        ranked_indices = np.argsort(similarity_scores)[::-1]

        # Take only the top K indices.
        top_indices = ranked_indices[:top_k]

        # Build the results DataFrame from the top items.
        results = self._items.iloc[top_indices].copy()
        results["similarity_score"] = similarity_scores[top_indices]

        # Round scores for clean display.
        results["similarity_score"] = results["similarity_score"].round(4)

        # Generate a human-readable explanation for each recommendation.
        results["explanation"] = results.apply(
            lambda row: explain_content_match(query, row), axis=1
        )

        # Return only the columns that are useful to the caller.
        output_columns = [
            "item_id",
            "title",
            "medium",
            "tone",
            "villain",
            "similarity_score",
            "explanation",
        ]
        return results[output_columns].reset_index(drop=True)
