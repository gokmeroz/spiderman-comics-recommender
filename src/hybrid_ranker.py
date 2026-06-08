"""
hybrid_ranker.py

Implements the hybrid recommender — the final, most interview-ready version.

─────────────────────────────────────────────────────────────────────────────
WHY A HYBRID SYSTEM?
─────────────────────────────────────────────────────────────────────────────
Versions 1–3 each solve a different problem but have their own blind spots:

  Content-based (V1/V2):
    ✓ Works for brand-new users with no history (cold start)
    ✗ Can over-specialize — only recommends things similar to what you typed

  Collaborative filtering (V3):
    ✓ Discovers non-obvious recommendations from user behavior
    ✗ Breaks for new users with no ratings (cold-start problem)

  Popularity:
    ✓ Always available — no user data or query needed
    ✗ Same recommendation for everyone — not personalized at all

A hybrid system combines all three signals so that:
  - When we have a query or liked titles → lean on content signal
  - When we have a known user → also add collaborative signal
  - Always add a small popularity boost as a safe fallback

─────────────────────────────────────────────────────────────────────────────
THE SCORING FORMULA
─────────────────────────────────────────────────────────────────────────────
final_score = 0.60 * content_score
            + 0.30 * collaborative_score
            + 0.10 * popularity_score

Weights must sum to 1.0. These specific values are a design choice, not a
mathematical truth. In a real product, you'd tune them on held-out data or
run A/B tests to see which weighting drives better engagement.

─────────────────────────────────────────────────────────────────────────────
WHY NORMALIZE BEFORE COMBINING?
─────────────────────────────────────────────────────────────────────────────
Each signal lives on a different scale:
  - content_score:        cosine similarity, roughly 0.0 – 0.4
  - collaborative_score:  weighted sum of ratings, roughly 0.0 – 5.0
  - popularity_score:     integer 0 – 100

If we combined them raw, collaborative_score would dominate just because
its numbers are larger — not because it's more important. Normalizing
each signal to 0–1 first puts them on equal footing before we weight them.

─────────────────────────────────────────────────────────────────────────────
THREE USAGE MODES
─────────────────────────────────────────────────────────────────────────────
Mode 1 — Query-only cold start:
  query="dark venom symbiote game"
  → content score from query, popularity fallback, collaborative = 0

Mode 2 — Liked/disliked profile:
  liked_titles=[...], disliked_titles=[...]
  → content score from profile vector, popularity fallback, collaborative = 0

Mode 3 — Known synthetic user (optionally with query):
  user_id="dark_venom_fan", query="dark venom"  (query is optional)
  → content score if query given, collaborative score from similar users,
    popularity fallback
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.content_recommender import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.user_profile import UserProfileRecommender
from src.explain import explain_hybrid_match


class HybridRecommender:
    """
    Combines content, collaborative, and popularity signals into one ranked list.

    This is the most production-realistic recommender in the project.
    It handles cold-start users (query or liked titles only) and known users
    (with behavioral data) through the same unified interface.
    """

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        collaborative_recommender: CollaborativeFilteringRecommender,
    ) -> None:
        """
        Store references to the already-fitted sub-recommenders.

        We also create a UserProfileRecommender internally — it reuses the
        fitted content recommender so there's no extra computation.

        Args:
            content_recommender: Fitted ContentBasedRecommender.
            collaborative_recommender: Fitted CollaborativeFilteringRecommender.
        """
        self._content_rec = content_recommender
        self._collab_rec = collaborative_recommender

        # UserProfileRecommender wraps the content recommender and gives us
        # the profile-vector logic (liked/disliked title lookup) for free.
        self._profile_rec = UserProfileRecommender(content_recommender)

    def recommend(
        self,
        query: str | None = None,
        user_id: str | None = None,
        liked_titles: list[str] | None = None,
        disliked_titles: list[str] | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Return top K recommendations using a weighted combination of signals.

        At least one of query, user_id, or liked_titles must be provided.

        Args:
            query: Free-text taste description (e.g. "dark venom symbiote").
            user_id: A known synthetic user ID from interactions.csv.
            liked_titles: Titles the user has positively rated.
            disliked_titles: Titles the user has negatively rated.
            top_k: Number of recommendations to return.

        Returns:
            A DataFrame with columns:
            item_id, title, medium, tone, villain,
            content_score, collaborative_score, popularity_score,
            final_score, explanation
        """
        if query is None and user_id is None and not liked_titles:
            raise ValueError(
                "Provide at least one of: query, user_id, or liked_titles."
            )

        # --- Step 1: Build a score frame with one row per item ---
        # This is the working table we'll fill in with each signal's scores.
        score_frame = self._build_empty_score_frame()

        # --- Step 2: Content score ---
        if query is not None:
            # Transform the query into a TF-IDF vector and score all items.
            query_vector = self._content_rec._vectorizer.transform([query])
            raw_content = cosine_similarity(
                query_vector, self._content_rec._item_matrix
            ).flatten()
            score_frame["content_score"] = raw_content

        elif liked_titles:
            # Build a user profile vector from liked/disliked titles and score all items.
            raw_content = self._compute_profile_scores(
                liked_titles, disliked_titles or []
            )
            score_frame["content_score"] = raw_content

        # --- Step 3: Collaborative score ---
        if user_id is not None:
            # Score all items based on similar users' weighted ratings.
            raw_collab = self._compute_collaborative_scores(user_id)
            for item_id, collab_score in raw_collab.items():
                mask = score_frame["item_id"] == item_id
                score_frame.loc[mask, "collaborative_score"] = collab_score

        # --- Step 4: Normalize all scores to 0–1 ---
        # Without normalization, a collaborative score of 4.5 would dwarf
        # a content score of 0.3, even if content is the stronger signal.
        score_frame["content_score"] = self._normalize_score(
            score_frame["content_score"]
        )
        score_frame["collaborative_score"] = self._normalize_score(
            score_frame["collaborative_score"]
        )
        # popularity_score was already normalized to 0–1 in _build_empty_score_frame.

        # --- Step 5: Compute the final weighted score ---
        score_frame["final_score"] = (
            0.60 * score_frame["content_score"]
            + 0.30 * score_frame["collaborative_score"]
            + 0.10 * score_frame["popularity_score"]
        ).round(4)

        # --- Step 6: Exclude already-seen items ---
        score_frame = self._exclude_seen_items(
            score_frame, user_id, liked_titles, disliked_titles
        )

        # --- Step 7: Sort by final score and take top K ---
        score_frame = score_frame.sort_values("final_score", ascending=False).head(top_k)

        # --- Step 8: Generate explanations ---
        score_frame = score_frame.copy()
        score_frame["explanation"] = score_frame.apply(
            lambda row: explain_hybrid_match(
                row["content_score"],
                row["collaborative_score"],
                row["popularity_score"],
            ),
            axis=1,
        )

        output_columns = [
            "item_id", "title", "medium", "tone", "villain",
            "content_score", "collaborative_score", "popularity_score",
            "final_score", "explanation",
        ]
        return score_frame[output_columns].reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_empty_score_frame(self) -> pd.DataFrame:
        """
        Create a working DataFrame with one row per item and all scores set to 0.

        Popularity is pre-normalized here since it's always available and
        doesn't depend on the query or user.
        """
        items = self._content_rec._items
        frame = items[["item_id", "title", "medium", "tone", "villain", "popularity"]].copy()
        frame = frame.reset_index(drop=True)
        frame["content_score"] = 0.0
        frame["collaborative_score"] = 0.0
        # Divide by 100 to bring popularity from the 0–100 scale into 0–1.
        frame["popularity_score"] = frame["popularity"] / 100.0
        return frame

    def _normalize_score(self, values: pd.Series) -> pd.Series:
        """
        Min-max normalize a Series of scores to the range [0, 1].

        If all values are identical (including all zeros), returns all zeros
        to avoid division by zero.

        Args:
            values: A Series of raw scores.

        Returns:
            A Series of scores scaled to [0, 1].
        """
        min_val = values.min()
        max_val = values.max()
        if max_val == min_val:
            # All scores are the same — normalization is meaningless, return zeros.
            return pd.Series(0.0, index=values.index)
        return (values - min_val) / (max_val - min_val)

    def _compute_profile_scores(
        self,
        liked_titles: list[str],
        disliked_titles: list[str],
    ) -> np.ndarray:
        """
        Build a user profile vector from liked/disliked titles and score all items.

        Reuses the UserProfileRecommender's title-lookup logic, then computes
        cosine similarity against the full item matrix.

        Returns:
            A numpy array of shape (num_items,) with one score per item.
        """
        liked_indices = self._profile_rec._get_item_indices_by_title(liked_titles)
        liked_vectors = self._content_rec._item_matrix[liked_indices].toarray()
        user_vector = liked_vectors.mean(axis=0)

        if disliked_titles:
            disliked_indices = self._profile_rec._get_item_indices_by_title(disliked_titles)
            disliked_vectors = self._content_rec._item_matrix[disliked_indices].toarray()
            user_vector -= disliked_vectors.mean(axis=0)

        return cosine_similarity(
            user_vector.reshape(1, -1), self._content_rec._item_matrix
        ).flatten()

    def _compute_collaborative_scores(self, user_id: str) -> dict[int, float]:
        """
        Compute a similarity-weighted score for every item based on similar users.

        Same logic as CollaborativeFilteringRecommender, but returns scores for
        ALL items (not just top K) so the hybrid ranker can combine them.

        Returns:
            A dict mapping item_id → collaborative score.
        """
        similar_users_df = self._collab_rec.get_similar_users(user_id, top_n=3)
        similar_user_ids = similar_users_df["user_id"].tolist()
        similarities = similar_users_df.set_index("user_id")["similarity"]

        user_item_matrix = self._collab_rec._user_item_matrix
        scores: dict[int, float] = {}

        for item_id in user_item_matrix.columns:
            score = 0.0
            for sim_user in similar_user_ids:
                sim = similarities[sim_user]
                rating = user_item_matrix.loc[sim_user, item_id]
                if rating > 0:
                    score += sim * rating
            scores[item_id] = score

        return scores

    def _exclude_seen_items(
        self,
        score_frame: pd.DataFrame,
        user_id: str | None,
        liked_titles: list[str] | None,
        disliked_titles: list[str] | None,
    ) -> pd.DataFrame:
        """
        Remove items the user has already seen from the score frame.

        We exclude:
          - liked_titles and disliked_titles (explicitly rated by the user)
          - All items rated by user_id in interactions (if user_id is given)

        Args:
            score_frame: The working score DataFrame.
            user_id: Optional known synthetic user.
            liked_titles: Titles to exclude (if provided).
            disliked_titles: Titles to exclude (if provided).

        Returns:
            A filtered score DataFrame.
        """
        excluded_titles: set[str] = set()

        if liked_titles:
            excluded_titles.update(t.lower() for t in liked_titles)
        if disliked_titles:
            excluded_titles.update(t.lower() for t in disliked_titles)

        if user_id is not None:
            # Look up all item IDs the user has rated in interactions.
            rated_ids = set(
                self._collab_rec._interactions[
                    self._collab_rec._interactions["user_id"] == user_id
                ]["item_id"]
            )
            # Convert item IDs to lowercase titles for consistent exclusion.
            items = self._content_rec._items
            rated_titles = items[items["item_id"].isin(rated_ids)]["title"].str.lower()
            excluded_titles.update(rated_titles)

        if excluded_titles:
            mask = ~score_frame["title"].str.lower().isin(excluded_titles)
            return score_frame[mask]

        return score_frame
