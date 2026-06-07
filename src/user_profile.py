"""
user_profile.py  —  Version 2: Personalized recommendation via user profile vectors

─────────────────────────────────────────────────────────────────────────────
THE PROBLEM WITH VERSION 1
─────────────────────────────────────────────────────────────────────────────
In Version 1, the user typed a query like "dark venom symbiote game".
That worked, but it had two problems:

  1. The user had to know what words to type.
  2. The query was the same every time, even if the user's taste is complex.

Version 2 solves this by learning the user's taste FROM their history,
not from their words.

─────────────────────────────────────────────────────────────────────────────
THE CORE IDEA: REPRESENT THE USER AS A VECTOR
─────────────────────────────────────────────────────────────────────────────
Every item in our catalog already has a TF-IDF vector — a list of numbers
that encodes its content (tone, villain, tags, etc.).

The key insight of Version 2:

  The user's TASTE can be encoded as a vector in the same space.

How? We average the TF-IDF vectors of items the user liked.

  liked_vector = mean of [vector("Spider-Man 2"), vector("Spider-Verse")]

This average vector points toward the region of the vector space that
the user tends to enjoy — emotional movies, Peter Parker, sacrifice, etc.

─────────────────────────────────────────────────────────────────────────────
WHAT DOES "AVERAGE THE VECTORS" ACTUALLY MEAN?
─────────────────────────────────────────────────────────────────────────────
Think of TF-IDF vectors as lists of word weights. Each position represents
one word in the vocabulary:

  ["dark", "emotional", "symbiote", "multiverse", "teen", ...]

  Spider-Man 2 vector:     [0.0,  0.4,  0.0,  0.0,  0.1, ...]
  Spider-Verse vector:     [0.0,  0.5,  0.0,  0.6,  0.3, ...]
                           ─────────────────────────────────────
  Average (liked_vector):  [0.0,  0.45, 0.0,  0.3,  0.2, ...]

The result is a new vector that carries the combined "fingerprint" of what
the user likes. Words shared by many liked items get high weights. Words
that appear in only one item get diluted.

─────────────────────────────────────────────────────────────────────────────
HOW DISLIKED ITEMS WORK: VECTOR SUBTRACTION
─────────────────────────────────────────────────────────────────────────────
If the user dislikes certain items, we compute their average vector and
SUBTRACT it from the liked vector:

  user_vector = liked_vector - disliked_vector

Subtraction in vector space means: "move away from the direction of things
I don't like." If the user dislikes "Web of Shadows" (dark, symbiote, venom),
the word "symbiote" gets a lower weight in the final user vector — so dark
symbiote items score worse in the ranking.

Concretely:

  liked_titles   = ["Spider-Man 2", "Spider-Man: Into the Spider-Verse"]
  disliked_titles = ["Spider-Man: Web of Shadows"]

  user_vector = mean(liked vectors) - mean(disliked vectors)

─────────────────────────────────────────────────────────────────────────────
HOW WE USE THE USER VECTOR TO RECOMMEND
─────────────────────────────────────────────────────────────────────────────
Once we have user_vector, we compute cosine similarity between it and
every item's TF-IDF vector. Items whose direction is closest to the user's
taste vector score highest. We then exclude items the user already rated
and return the top K remaining items.

─────────────────────────────────────────────────────────────────────────────
WHY IS THIS STILL CONTENT-BASED?
─────────────────────────────────────────────────────────────────────────────
Because the user vector is built entirely from item METADATA (tags, tone,
villain, etc.) — the same TF-IDF vectors we use to describe items.

We are NOT using other users' behavior yet. That comes in Version 3
(collaborative filtering), where we ask: "what did similar users like?"

─────────────────────────────────────────────────────────────────────────────
WHY IS THIS MORE PERSONALIZED THAN VERSION 1?
─────────────────────────────────────────────────────────────────────────────
Version 1: User writes a query. The query IS the taste signal.
Version 2: User provides examples. The math DERIVES the taste signal.

This is more powerful because:
  - The user doesn't need to know what words describe their own taste.
  - Liked items contribute many features simultaneously (tone AND villain
    AND era AND tags), not just the words the user happened to type.
  - Negative examples actively steer recommendations away from bad regions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.content_recommender import ContentBasedRecommender
from src.explain import explain_profile_match


class UserProfileRecommender:
    """
    Recommends Spider-Man media based on a user's liked and disliked items.

    This class reuses the already-fitted ContentBasedRecommender — it borrows
    the vectorizer and item matrix without re-fitting anything. This is efficient
    and ensures the user profile lives in the same vector space as the items.
    """

    def __init__(self, content_recommender: ContentBasedRecommender) -> None:
        """
        Store a reference to the fitted content recommender.

        We don't copy or re-fit anything here. We just hold a pointer so we
        can access _items, _vectorizer, and _item_matrix later.

        Args:
            content_recommender: A fully initialized ContentBasedRecommender.
        """
        # Store the fitted recommender so we can access its internal state.
        self._recommender = content_recommender

    def recommend_by_user_profile(
        self,
        liked_titles: list[str],
        disliked_titles: list[str] | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Return top K recommendations based on liked and disliked items.

        Steps:
          1. Look up item vectors for liked titles.
          2. Average them into one "liked direction" vector.
          3. If disliked titles exist, compute their average vector and subtract it.
          4. Rank all items by cosine similarity to this user profile vector.
          5. Exclude liked and disliked titles from the results.
          6. Return the top K.

        Args:
            liked_titles: Titles the user has expressed a positive preference for.
            disliked_titles: Titles the user has expressed a negative preference for.
            top_k: How many recommendations to return.

        Returns:
            A DataFrame with columns:
            item_id, title, medium, tone, villain, profile_similarity_score, explanation
        """
        if disliked_titles is None:
            disliked_titles = []

        # --- Step 1: Build the liked vector ---
        # Find the positional indices of liked titles in the item matrix.
        liked_indices = self._get_item_indices_by_title(liked_titles)

        # Slice the item matrix to get only the liked items' vectors.
        # _item_matrix is a sparse matrix; .toarray() converts to dense numpy array.
        # Shape after toarray(): (num_liked_items, num_vocabulary_words)
        liked_vectors = self._recommender._item_matrix[liked_indices].toarray()

        # Average across rows to get one vector representing overall liked taste.
        # mean(axis=0) collapses (num_liked_items, vocab_size) → (vocab_size,)
        liked_vector = liked_vectors.mean(axis=0)

        # Start the user vector as the liked direction.
        user_vector = liked_vector.copy()

        # --- Step 2: Subtract the disliked vector (if any) ---
        if disliked_titles:
            disliked_indices = self._get_item_indices_by_title(disliked_titles)
            disliked_vectors = self._recommender._item_matrix[disliked_indices].toarray()
            disliked_vector = disliked_vectors.mean(axis=0)

            # Subtracting the disliked vector steers the user profile away
            # from items that share features with the disliked ones.
            user_vector = user_vector - disliked_vector

        # --- Step 3: Compute similarity to all items ---
        # cosine_similarity expects a 2D array, so reshape (vocab_size,) → (1, vocab_size).
        user_vector_2d = user_vector.reshape(1, -1)

        # Compute similarity between the user vector and every item vector.
        # Result shape: (1, num_items) → flatten to (num_items,)
        scores = cosine_similarity(user_vector_2d, self._recommender._item_matrix).flatten()

        # --- Step 4: Rank and filter ---
        # Get all item indices sorted by score descending.
        ranked_indices = np.argsort(scores)[::-1]

        # Build a set of titles to exclude (items the user has already rated).
        # We never recommend something the user already knows about.
        excluded_titles = set(t.lower() for t in liked_titles + disliked_titles)

        items = self._recommender._items

        # Walk through ranked items and collect the top K that are not excluded.
        results = []
        for idx in ranked_indices:
            title = items.iloc[idx]["title"]
            if title.lower() in excluded_titles:
                continue
            row = items.iloc[idx].copy()
            row["profile_similarity_score"] = round(float(scores[idx]), 4)
            row["explanation"] = explain_profile_match(row)
            results.append(row)
            if len(results) == top_k:
                break

        output_columns = [
            "item_id",
            "title",
            "medium",
            "tone",
            "villain",
            "profile_similarity_score",
            "explanation",
        ]
        return pd.DataFrame(results)[output_columns].reset_index(drop=True)

    def _get_item_indices_by_title(self, titles: list[str]) -> list[int]:
        """
        Return the positional row indices (in the item matrix) for a list of titles.

        Matching is case-insensitive. If a title is not found, a clear error
        is raised listing all available titles.

        Args:
            titles: A list of item titles to look up.

        Returns:
            A list of integer row indices into _item_matrix.

        Raises:
            ValueError: If any title is not found in the item catalog.
        """
        items = self._recommender._items
        # Lowercase all stored titles once for efficient repeated lookups.
        titles_lower = items["title"].str.lower()
        available_titles = items["title"].tolist()

        indices = []
        for title in titles:
            # np.where returns a tuple; [0] gives the array of matching positions.
            matching_positions = np.where(titles_lower == title.lower())[0]

            if len(matching_positions) == 0:
                raise ValueError(
                    f"Title '{title}' not found in the item catalog.\n"
                    f"Available titles:\n" +
                    "\n".join(f"  - {t}" for t in available_titles)
                )

            # Take the first match (titles should be unique in our dataset).
            indices.append(int(matching_positions[0]))

        return indices
