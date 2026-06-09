"""
evaluation.py

Implements offline ranking evaluation metrics for the recommender system.

─────────────────────────────────────────────────────────────────────────────
WHY NOT JUST USE ACCURACY?
─────────────────────────────────────────────────────────────────────────────
A recommender doesn't predict a single label — it produces an ORDERED LIST.
Accuracy would treat "recommended in position 1" and "recommended in position
100" as equally good. That's wrong. We need metrics that care about:
  1. How many relevant items appeared in the top K?   → Precision@K, Recall@K
  2. Did the relevant items appear near the TOP?       → NDCG@K

─────────────────────────────────────────────────────────────────────────────
PRECISION@K, HOW MANY OF THE TOP K WERE RELEVANT?
─────────────────────────────────────────────────────────────────────────────
"Of the top K recommendations, what fraction were relevant?"

  Precision@K = |relevant ∩ top_K| / K

Example:
  top_5 = [Venom, Web of Shadows, PS5, Kraven, Spider-Man 3]
  relevant = {Web of Shadows, PS5, Venom}
  → 3 hits out of 5 → Precision@5 = 0.60

─────────────────────────────────────────────────────────────────────────────
RECALL@K, DID WE FIND ALL THE RELEVANT ITEMS IN THE TOP K?
─────────────────────────────────────────────────────────────────────────────
"Of all relevant items in the catalog, what fraction did we find in top K?"

  Recall@K = |relevant ∩ top_K| / |relevant|

Example (same as above):
  → 3 hits out of 3 relevant → Recall@5 = 1.0

Note: Precision and Recall trade off against each other. A recommender that
returns ALL items gets Recall = 1.0 but terrible Precision.

─────────────────────────────────────────────────────────────────────────────────────────────
NDCG@K (Normalized Discounted Cumulative Gain), DID WE RANK THE RELEVANT ITEMS NEAR THE TOP?
─────────────────────────────────────────────────────────────────────────────────────────────
"Did we rank the relevant items near the TOP of the list?"

Precision@K and Recall@K don't care about ORDER within the top K. NDCG does.

  DCG  = sum( relevance[i] / log2(i + 2) )   for i = 0 to K-1  (0-indexed)
  IDCG = DCG of the ideal ranking (all relevant items first)
  NDCG = DCG / IDCG

The log2 discount means: a relevant item at position 1 contributes 1.0,
at position 2 contributes 0.63, at position 3 contributes 0.5, and so on.
Relevant items buried near position K contribute almost nothing.

NDCG = 1.0 → perfect ranking (all relevant items at the very top).
NDCG = 0.0 → no relevant items found at all.

─────────────────────────────────────────────────────────────────────────────
OFFLINE EVALUATION LIMITATIONS
─────────────────────────────────────────────────────────────────────────────
Offline evaluation is imperfect. It can only measure what we THINK is
relevant — but relevance in our test cases is manually defined, not measured
from real user behavior. A real production system would also run:
  - Online A/B tests (real users, real clicks)
  - User surveys
  - Long-term retention metrics

Offline evaluation is a starting point, not a final verdict.
"""

import math

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Built-in test cases
# ─────────────────────────────────────────────────────────────────────────────
# These are pre-defined queries with hand-labeled relevant items.
# They let us quickly evaluate recommender quality without writing test code.
TEST_CASES = [
    {
        "name": "dark venom test",
        "query": "dark venom symbiote psychological",
        "relevant_titles": [
            "Spider-Man: Web of Shadows",
            "Ultimate Spider-Man",
            "Spider-Man 2 PS5",
            "Venom",
            "Kraven's Last Hunt",
        ],
    },
    {
        "name": "miles multiverse test",
        "query": "miles morales multiverse animation identity",
        "relevant_titles": [
            "Spider-Man: Into the Spider-Verse",
            "Spider-Man: Across the Spider-Verse",
            "Spider-Man: Miles Morales",
            "Spider-Gwen",
        ],
    },
    {
        "name": "emotional peter test",
        "query": "emotional peter parker responsibility sacrifice",
        "relevant_titles": [
            "Spider-Man 2",
            "Spider-Man PS4",
            "Spider-Man: Life Story",
            "Spider-Man Blue",
        ],
    },
]


def precision_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    """
    Compute Precision@K.

    Of the top K recommendations, what fraction were relevant?

    Args:
        recommended_ids: Ordered list of recommended item IDs (best first).
        relevant_ids: Set of item IDs considered ground-truth relevant.
        k: Evaluation cutoff — only the first K items are considered.

    Returns:
        A float in [0, 1].
    """
    if k <= 0:
        return 0.0

    # Take only the first K items.
    top_k = recommended_ids[:k]

    # Count how many of the top K items are relevant.
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)

    return hits / k


def recall_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Of all relevant items in the catalog, what fraction appeared in the top K?

    Args:
        recommended_ids: Ordered list of recommended item IDs (best first).
        relevant_ids: Set of item IDs considered ground-truth relevant.
        k: Evaluation cutoff — only the first K items are considered.

    Returns:
        A float in [0, 1].
    """
    # If there are no relevant items, recall is undefined — return 0.
    if not relevant_ids:
        return 0.0

    top_k = recommended_ids[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)

    # Divide by total number of relevant items, not K.
    # This measures coverage: did we find them all?
    return hits / len(relevant_ids)


def ndcg_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).

    Measures whether relevant items are ranked near the top.

    Args:
        recommended_ids: Ordered list of recommended item IDs (best first).
        relevant_ids: Set of item IDs considered ground-truth relevant.
        k: Evaluation cutoff.

    Returns:
        A float in [0, 1]. 1.0 = perfect ranking, 0.0 = no relevant items found.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = recommended_ids[:k]

    # Compute DCG: each relevant item at position i contributes 1 / log2(i+2).
    # We use i+2 because:
    #   - i is 0-indexed
    #   - standard DCG formula uses log2(position + 1) with 1-indexed positions
    #   - so position 1 (i=0) → log2(0+2) = log2(2) = 1.0 (maximum contribution)
    dcg = 0.0
    for i, item_id in enumerate(top_k):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)

    # Compute IDCG: the best possible DCG.
    # Imagine all relevant items placed at positions 1, 2, 3, ...
    # We only go up to min(|relevant|, K) because we can't place more than K items.
    n_relevant_in_top_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_top_k))

    if idcg == 0.0:
        return 0.0

    # NDCG normalizes DCG by the ideal, so it's always in [0, 1].
    return dcg / idcg


def evaluate_recommender(
    recommender,
    test_cases: list[dict],
    k: int = 5,
) -> pd.DataFrame:
    """
    Run a batch of test cases and compute Precision@K, Recall@K, NDCG@K for each.

    Each test case dict must have:
        name (str):             A label for the test.
        query (str):            The query passed to the recommender.
        relevant_titles (list): Titles considered ground-truth relevant.

    Args:
        recommender: A HybridRecommender instance.
        test_cases: List of test case dicts (see TEST_CASES above for format).
        k: Evaluation cutoff.

    Returns:
        A DataFrame with columns:
        name, precision_at_k, recall_at_k, ndcg_at_k, recommended_titles
    """
    # Access the item catalog from the hybrid recommender's content sub-recommender.
    items = recommender._content_rec._items

    rows = []
    for tc in test_cases:
        # Run the recommender for this test case's query.
        recs = recommender.recommend(query=tc["query"], top_k=k)
        recommended_ids = recs["item_id"].tolist()
        recommended_titles = recs["title"].tolist()

        # Convert relevant titles to item IDs.
        # We look up each title by exact match in the items catalog.
        # Unrecognized titles are silently skipped (they won't affect scoring).
        relevant_ids: set[int] = set()
        for title in tc["relevant_titles"]:
            match = items[items["title"] == title]
            if not match.empty:
                relevant_ids.add(int(match.iloc[0]["item_id"]))

        # Compute all three metrics.
        precision = precision_at_k(recommended_ids, relevant_ids, k)
        recall = recall_at_k(recommended_ids, relevant_ids, k)
        ndcg = ndcg_at_k(recommended_ids, relevant_ids, k)

        rows.append({
            "name": tc["name"],
            f"precision_at_{k}": round(precision, 4),
            f"recall_at_{k}": round(recall, 4),
            f"ndcg_at_{k}": round(ndcg, 4),
            "recommended_titles": recommended_titles,
        })

    return pd.DataFrame(rows)
