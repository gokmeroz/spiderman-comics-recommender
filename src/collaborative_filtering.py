"""
collaborative_filtering.py

Implements a user-user collaborative filtering recommender.

─────────────────────────────────────────────────────────────────────────────
WHAT IS COLLABORATIVE FILTERING?
─────────────────────────────────────────────────────────────────────────────
Versions 1 and 2 looked at item CONTENT (tags, tone, villain, etc.) to decide
what to recommend. Collaborative filtering takes a completely different approach:

  "Find users who behave like you. Recommend what they liked."

We never look at item metadata at all. We only look at the RATINGS MATRIX —
who rated what, and how highly. If two users gave similar ratings to the same
set of items, we consider them "similar" and use each other's ratings to fill
in gaps.

Example:
  dark_venom_fan rated Web of Shadows: 5, Ultimate Spider-Man: 5, Venom: 5
  gamer          rated Web of Shadows: 4, Ultimate Spider-Man: 4

  → dark_venom_fan and gamer have similar patterns.
  → dark_venom_fan also rated Kraven's Last Hunt: 4.
  → Recommend Kraven's Last Hunt to gamer, who hasn't seen it yet.

─────────────────────────────────────────────────────────────────────────────
THE USER-ITEM MATRIX
─────────────────────────────────────────────────────────────────────────────
The core data structure is a 2D matrix:

         item_1  item_2  item_3  ...
  user_A    5       0       3
  user_B    0       4       0
  user_C    2       4       5
  PS: 0 means "not rated". Ratings are between 1 and 5. This is a sparse matrix — most users have rated only a few items.

  - Rows = users
  - Columns = items
  - Values = ratings (1–5)
  - 0 means the user hasn't rated that item (NOT a rating of zero)

Missing values are filled with 0 because:
  - We need a numeric matrix to compute similarity
  - 0 is a neutral placeholder that doesn't push similarity up or down
  - In practice, sparse matrix techniques handle this more elegantly,
    but 0-filling is the clearest way to learn the concept

The similarity between two users is computed by treating their rows as vectors and
computing cosine similarity. Users with similar rating patterns will have a high cosine similarity score.

E.g., if user_A and user_B both rated the same items similarly, 
their vectors will point in a similar direction, 
resulting in a high cosine similarity score close to 1. 
If they have very different rating patterns, the cosine similarity will be closer to 0.

─────────────────────────────────────────────────────────────────────────────
USER-USER SIMILARITY
─────────────────────────────────────────────────────────────────────────────
We treat each user's row as a vector and compute cosine similarity between
all pairs of users. Two users who rated the same items similarly will have
a high cosine similarity score.

Note: this is the SAME cosine similarity we used in Versions 1 and 2,
but now we're comparing USER vectors instead of ITEM vectors.

─────────────────────────────────────────────────────────────────────────────
HOW RECOMMENDATIONS ARE GENERATED
─────────────────────────────────────────────────────────────────────────────
For a target user:
  1. Find the top N most similar users.
  2. For every item NOT yet rated by the target user:
     - Compute a score = sum(similar_user_similarity * similar_user_rating)
     - This weights high-similarity users more heavily.
  3. Sort items by score descending.
  4. Return the top K.

─────────────────────────────────────────────────────────────────────────────
KEY DIFFERENCE FROM VERSIONS 1 AND 2
─────────────────────────────────────────────────────────────────────────────
V1 and V2: item metadata drives recommendations (content-based).
V3:        user behavior drives recommendations (collaborative).

V3 can discover non-obvious connections — e.g., users who like dark games
also tend to enjoy certain dark comics — without ever reading the item tags.

─────────────────────────────────────────────────────────────────────────────
THE COLD-START PROBLEM
─────────────────────────────────────────────────────────────────────────────
Collaborative filtering breaks for brand-new users who have no ratings yet.
There's no row for them in the user-item matrix, so we can't compute similarity.
This is called the cold-start problem. It's why hybrid systems (Version 4)
fall back to content-based signals when behavior data is unavailable.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.explain import explain_collaborative_match


class CollaborativeFilteringRecommender:
    """
    Recommends Spider-Man media based on similar users' rating patterns.

    At init time, builds a user-item matrix and computes a full user-user
    similarity matrix. Both are reused for every recommendation request.
    """

    def __init__(self, items: pd.DataFrame, interactions: pd.DataFrame) -> None:
        """
        Build the user-item matrix and precompute user-user similarities.

        Args:
            items: The items DataFrame from load_items().
            interactions: The interactions DataFrame from load_interactions().
        """
        # Store items so we can look up titles, medium, tone, villain later.
        self._items = items

        # Store interactions so we can look up what each user has already rated.
        self._interactions = interactions

        # Build the user-item matrix using pivot_table.
        # pivot_table reshapes the long-format interactions table into a wide matrix.
        #   - index="user_id"  → each row is one user
        #   - columns="item_id" → each column is one item
        #   - values="rating"   → the cell value is the rating
        #   - fill_value=0      → missing ratings become 0 (not rated ≠ rated 0)
        self._user_item_matrix = interactions.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )

        # Compute cosine similarity between every pair of users.
        # Input shape:  (num_users, num_items)
        # Output shape: (num_users, num_users)
        # We do this once at init so recommendations are fast later.
        raw_similarity = cosine_similarity(self._user_item_matrix.values)

        # Wrap the similarity matrix in a DataFrame so we can index by user_id.
        # This makes lookups like self._similarity_df.loc["gamer"] very readable.
        self._similarity_df = pd.DataFrame(
            raw_similarity,
            index=self._user_item_matrix.index,
            columns=self._user_item_matrix.index,
        )

    def recommend_by_similar_users(self, user_id: str, top_k: int = 5) -> pd.DataFrame:
        """
        Return top K items liked by users similar to the target user.

        Steps:
          1. Validate that user_id exists.
          2. Find the top similar users (excluding the target user).
          3. For each item not yet rated by the target user, compute a
             similarity-weighted score from the similar users' ratings.
          4. Sort by score and return top K with item metadata.

        Args:
            user_id: The ID of the target user (must exist in interactions).
            top_k: How many recommendations to return.

        Returns:
            A DataFrame with columns:
            item_id, title, medium, tone, villain,
            collaborative_score, similar_users, explanation

        Raises:
            ValueError: If user_id is not in the interactions dataset.
        """
        self._validate_user(user_id)

        # --- Step 1: Find the most similar users ---
        # Get top 3 similar users to keep the scoring tractable and explainable.
        similar_users_df = self.get_similar_users(user_id, top_n=3)
        similar_user_ids = similar_users_df["user_id"].tolist()
        similar_user_similarities = similar_users_df.set_index("user_id")["similarity"]

        # --- Step 2: Identify items the target user has already rated ---
        already_rated_ids = set(
            self._interactions[self._interactions["user_id"] == user_id]["item_id"]
        )

        # --- Step 3: Score candidate items ---
        # For each item in the catalog, compute a weighted score.
        # Score = sum over similar users of (their similarity * their rating).
        # Items rated 5 by a very similar user score highest.
        candidate_scores: dict[int, float] = {}

        for item_id in self._user_item_matrix.columns:
            if item_id in already_rated_ids:
                continue  # skip items the user already knows about

            score = 0.0
            for sim_user_id in similar_user_ids:
                similarity = similar_user_similarities[sim_user_id]
                rating = self._user_item_matrix.loc[sim_user_id, item_id]
                # Only contribute if the similar user actually rated this item.
                # A rating of 0 means "not rated", not "hated it".
                if rating > 0:
                    score += similarity * rating

            if score > 0:
                candidate_scores[item_id] = score

        if not candidate_scores:
            return pd.DataFrame(columns=[
                "item_id", "title", "medium", "tone", "villain",
                "collaborative_score", "similar_users", "explanation",
            ])

        # --- Step 4: Build results DataFrame ---
        # Sort candidates by score descending and take top K.
        sorted_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_k]

        rows = []
        for item_id, score in top_items:
            # Look up full item metadata by item_id.
            item_row = self._items[self._items["item_id"] == item_id].iloc[0]
            row = item_row.to_dict()
            row["collaborative_score"] = round(score, 4)
            row["similar_users"] = similar_user_ids
            row["explanation"] = explain_collaborative_match(item_row, similar_user_ids)
            rows.append(row)

        output_columns = [
            "item_id",
            "title",
            "medium",
            "tone",
            "villain",
            "collaborative_score",
            "similar_users",
            "explanation",
        ]
        return pd.DataFrame(rows)[output_columns].reset_index(drop=True)

    def get_similar_users(self, user_id: str, top_n: int = 3) -> pd.DataFrame:
        """
        Return the top N users most similar to the target user.

        The target user is always excluded from their own similar-users list —
        a user is trivially 100% similar to themselves, which is useless.

        Args:
            user_id: The target user ID.
            top_n: How many similar users to return.

        Returns:
            A DataFrame with columns: user_id, similarity.
            Sorted by similarity descending.

        Raises:
            ValueError: If user_id is not in the interactions dataset.
        """
        self._validate_user(user_id)

        # Get the target user's row from the similarity matrix.
        # This gives us a Series: {other_user_id: similarity_score}
        similarities = self._similarity_df.loc[user_id].copy()

        # Drop the target user — comparing someone to themselves is always 1.0
        # and tells us nothing useful.
        similarities = similarities.drop(user_id)

        # Sort descending and take top N.
        top_similar = similarities.nlargest(top_n)

        return pd.DataFrame({
            "user_id": top_similar.index,
            "similarity": top_similar.values,
        }).reset_index(drop=True)

    def _validate_user(self, user_id: str) -> None:
        """
        Raise a clear ValueError if user_id is not in the interaction dataset.

        Collaborative filtering cannot work for users it has never seen.
        This is the cold-start problem — we fail loudly rather than silently.

        Args:
            user_id: The user ID to check.

        Raises:
            ValueError: If user_id is unknown.
        """
        known_users = self._user_item_matrix.index.tolist()
        if user_id not in known_users:
            raise ValueError(
                f"User '{user_id}' not found in the interactions dataset.\n"
                f"Available users:\n" +
                "\n".join(f"  - {u}" for u in known_users)
            )
