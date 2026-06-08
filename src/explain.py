"""
explain.py

Generates simple, human-readable explanations for why an item was recommended.

Why explanations matter:
  A recommendation without a reason feels random and untrustworthy.
  Users are more likely to follow a recommendation if they understand it.
  In interviews, explaining *why* a system recommended something shows
  you think about the full user experience, not just the math.

  This version uses simple keyword overlap — no LLMs, no APIs.
"""

import pandas as pd


def explain_content_match(query: str, item_row: pd.Series) -> str:
    """
    Generate a short explanation of why an item matches a query.

    The explanation is based on which words from the query also appear
    in the item's metadata fields (title, tone, villain, theme, tags).

    This is a deterministic function — same inputs always produce the
    same output. That makes it easy to test and debug.

    Args:
        query: The text query the user entered (e.g. "dark venom symbiote").
        item_row: One row from the items DataFrame.

    Returns:
        A short explanation string like:
        "Recommended because it matches your interest in: dark, venom, symbiote."
    Example:
        If the query is "dark venom symbiote" and the item has tone="dark", villain="Venom", theme="symbiote", 
        then the explanation would be:
        "Recommended because it matches your interest in: dark, venom, symbiote."
    """
    # Normalize the query to lowercase for case-insensitive matching.
    query_words = set(query.lower().split())

    # Collect text from all relevant metadata fields for this item.
    # We check each field separately so we can report which ones matched.
    metadata_text = " ".join([
        str(item_row.get("tone", "")),
        str(item_row.get("villain", "")),
        str(item_row.get("theme", "")),
        str(item_row.get("tags", "")),
        str(item_row.get("medium", "")),
        str(item_row.get("era", "")),
    ]).lower()

    # Find which query words actually appear in the item's metadata.
    matched_words = [word for word in query_words if word in metadata_text]

    if matched_words:
        # Sort for consistent ordering (deterministic output).
        matched_words.sort()
        joined = ", ".join(matched_words)
        return f"Recommended because it matches your interest in: {joined}."
    else:
        # Fallback when no specific keywords match — the TF-IDF still found
        # a similarity, but it may be based on subtle word overlap.
        return f"Recommended based on overall similarity to your query."


def explain_profile_match(item_row: pd.Series) -> str:
    """
    Generate a short explanation for a user-profile-based recommendation.

    Instead of matching a query, we now know the item was surfaced because its
    vector is close to the user's averaged preference vector. We report the most
    descriptive traits of the item so the user understands why it appeared.

    Args:
        item_row: One row from the items DataFrame.

    Returns:
        A short explanation string.
    """
    tone = str(item_row.get("tone", "")).strip()
    villain = str(item_row.get("villain", "")).strip()
    theme = str(item_row.get("theme", "")).strip()
    medium = str(item_row.get("medium", "")).strip()

    traits = []

    if tone and tone.lower() not in ("nan", ""):
        traits.append(f"{tone} tone")
    if villain and villain.lower() not in ("nan", "", "multiple"):
        traits.append(f"{villain}")
    if theme and theme.lower() not in ("nan", ""):
        traits.append(f"{theme} theme")
    if medium and medium.lower() not in ("nan", ""):
        traits.append(f"{medium}")

    if traits:
        joined = ", ".join(traits[:3])
        return f"Matches your taste profile: {joined}."
    return "Recommended based on similarity to your liked items."


def explain_collaborative_match(item_row: pd.Series, similar_users: list[str]) -> str:
    """
    Generate a short explanation for a collaborative filtering recommendation.

    Unlike content-based explanations, we don't explain WHAT the item is about.
    We explain WHO liked it — because that's how collaborative filtering works.
    The item was surfaced because people with similar taste to the user enjoyed it.

    Args:
        item_row: One row from the items DataFrame.
        similar_users: List of user IDs whose ratings contributed to this recommendation.

    Returns:
        A short explanation string.
    """
    if similar_users:
        users_str = ", ".join(similar_users[:3])
        return f"Liked by users with similar taste to you ({users_str})."
    return "Recommended based on similar users' ratings."


def explain_hybrid_match(
    content_score: float,
    collaborative_score: float,
    popularity_score: float,
) -> str:
    """
    Generate a short explanation for a hybrid recommendation.

    Reports which signals were strongest so the user understands
    why this item ranked highly in the combined score.

    Args:
        content_score: Normalized content/profile similarity (0–1).
        collaborative_score: Normalized collaborative score (0–1).
        popularity_score: Normalized popularity (0–1).

    Returns:
        A short explanation string naming the dominant signals.
    """
    signals = []

    if content_score > 0.3:
        signals.append("strong content match")
    elif content_score > 0.05:
        signals.append("content match")

    if collaborative_score > 0.3:
        signals.append("highly rated by similar users")
    elif collaborative_score > 0.05:
        signals.append("liked by similar users")

    if popularity_score > 0.85:
        signals.append("very popular")
    elif popularity_score > 0.7 and not signals:
        # Only mention popularity as the lead signal when nothing else is strong.
        signals.append("popular title")

    if signals:
        return "Recommended based on: " + ", ".join(signals) + "."
    return "Recommended by hybrid ranking."
