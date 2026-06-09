"""
demo.py

Command-line interface for the Spider-Sense Recommender.

Run with:
    python app/demo.py

This demo shows all five recommender modes in a simple interactive menu.
Each mode uses a different part of the pipeline we built across Versions 1–5.
"""

import sys
from pathlib import Path

# Add the project root to sys.path so imports like "from src.xxx" work
# regardless of which directory the user runs this from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_items, load_interactions
from src.content_recommender import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.user_profile import UserProfileRecommender
from src.hybrid_ranker import HybridRecommender
from src.evaluation import evaluate_recommender, TEST_CASES


# ─────────────────────────────────────────────────────────────────────────────
# Setup — load data and build all recommenders once at startup
# ─────────────────────────────────────────────────────────────────────────────

def build_recommenders():
    """Load data and initialize all recommenders."""
    print("Loading data and building recommenders...")
    items = load_items()
    interactions = load_interactions()
    content_rec = ContentBasedRecommender(items)
    collab_rec = CollaborativeFilteringRecommender(items, interactions)
    profile_rec = UserProfileRecommender(content_rec)
    hybrid_rec = HybridRecommender(content_rec, collab_rec)
    print("Ready.\n")
    return items, interactions, content_rec, collab_rec, profile_rec, hybrid_rec


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_separator():
    print("-" * 60)


def print_results(results, score_col: str):
    """Print a recommendations DataFrame in a readable format."""
    print_separator()
    for i, row in results.iterrows():
        print(f"{i + 1}. {row['title']} ({row['medium']}, {row['tone']})")
        print(f"   Score: {row[score_col]:.4f}")
        print(f"   Why:   {row['explanation']}")
    print_separator()


def print_hybrid_results(results):
    """Print hybrid results with all three score components."""
    print_separator()
    for i, row in results.iterrows():
        print(f"{i + 1}. {row['title']} ({row['medium']}, {row['tone']})")
        print(f"   Final score:   {row['final_score']:.4f}")
        print(f"   Content:       {row['content_score']:.4f}  |  "
              f"Collab: {row['collaborative_score']:.4f}  |  "
              f"Popularity: {row['popularity_score']:.4f}")
        print(f"   Why:   {row['explanation']}")
    print_separator()


# ─────────────────────────────────────────────────────────────────────────────
# Mode handlers
# ─────────────────────────────────────────────────────────────────────────────

def mode_query(content_rec):
    """
    Mode 1: Query-based recommendation (Version 1 — ContentBasedRecommender).

    The user types a free-text description of their taste.
    TF-IDF + cosine similarity finds the most similar items.
    Works for brand-new users with no history.
    """
    print("\nMode 1: Query-based recommendation")
    print("Describe your Spider-Verse taste in a few words.")
    print("Example: dark venom symbiote game")
    query = input("\nEnter your query: ").strip()
    if not query:
        print("No query entered.")
        return
    results = content_rec.recommend_by_query(query, top_k=5)
    print(f"\nTop 5 recommendations for: \"{query}\"")
    print_results(results, "similarity_score")


def mode_profile(profile_rec):
    """
    Mode 2: Liked/disliked profile recommendation (Version 2 — UserProfileRecommender).

    The user provides titles they liked and disliked.
    The system builds a profile vector and finds similar unseen items.
    """
    print("\nMode 2: Liked/disliked profile recommendation")
    print("Enter titles separated by commas.")

    liked_input = input("Liked titles: ").strip()
    if not liked_input:
        print("No liked titles entered.")
        return

    liked_titles = [t.strip() for t in liked_input.split(",") if t.strip()]

    disliked_input = input("Disliked titles (or leave blank): ").strip()
    disliked_titles = [t.strip() for t in disliked_input.split(",") if t.strip()]

    try:
        results = profile_rec.recommend_by_user_profile(
            liked_titles=liked_titles,
            disliked_titles=disliked_titles or None,
            top_k=5,
        )
        print(f"\nTop 5 recommendations based on your profile:")
        print_results(results, "profile_similarity_score")
    except ValueError as e:
        print(f"\nError: {e}")


def mode_synthetic_user(collab_rec):
    """
    Mode 3: Synthetic user recommendation (Version 3 — CollaborativeFilteringRecommender).

    The user picks a synthetic persona. The system finds similar users
    and recommends items they liked that the target user hasn't seen.
    """
    print("\nMode 3: Synthetic user recommendation")
    print("Available users:")
    known_users = collab_rec._user_item_matrix.index.tolist()
    for user in known_users:
        print(f"  - {user}")

    user_id = input("\nEnter user ID: ").strip()

    try:
        results = collab_rec.recommend_by_similar_users(user_id, top_k=5)
        similar = collab_rec.get_similar_users(user_id, top_n=3)
        print(f"\nMost similar users to {user_id}:")
        for _, row in similar.iterrows():
            print(f"  {row['user_id']} (similarity: {row['similarity']:.4f})")
        print(f"\nTop 5 recommendations for {user_id}:")
        print_results(results, "collaborative_score")
    except ValueError as e:
        print(f"\nError: {e}")


def mode_hybrid(hybrid_rec):
    """
    Mode 4: Hybrid recommendation (Version 4 — HybridRecommender).

    Combines content, collaborative, and popularity signals.
    Supports any combination of query, user_id, liked, and disliked titles.
    """
    print("\nMode 4: Hybrid recommendation")
    print("All inputs are optional — provide at least one.\n")

    query = input("Query (or leave blank): ").strip() or None

    known_users = hybrid_rec._collab_rec._user_item_matrix.index.tolist()
    print(f"Known users: {', '.join(known_users)}")
    user_id = input("User ID (or leave blank): ").strip() or None

    liked_input = input("Liked titles, comma-separated (or leave blank): ").strip()
    liked_titles = [t.strip() for t in liked_input.split(",") if t.strip()] or None

    disliked_input = input("Disliked titles, comma-separated (or leave blank): ").strip()
    disliked_titles = [t.strip() for t in disliked_input.split(",") if t.strip()] or None

    try:
        results = hybrid_rec.recommend(
            query=query,
            user_id=user_id,
            liked_titles=liked_titles,
            disliked_titles=disliked_titles,
            top_k=5,
        )
        print("\nTop 5 hybrid recommendations:")
        print_hybrid_results(results)
    except ValueError as e:
        print(f"\nError: {e}")


def mode_evaluation(hybrid_rec):
    """
    Mode 5: Evaluation demo (Version 5 — evaluate_recommender).

    Runs the three built-in test cases and prints Precision@5,
    Recall@5, and NDCG@5 for each.
    """
    print("\nMode 5: Evaluation demo")
    print("Running built-in test cases against the hybrid recommender...\n")

    results = evaluate_recommender(hybrid_rec, TEST_CASES, k=5)

    for _, row in results.iterrows():
        print(f"Test: {row['name']}")
        print(f"  Precision@5: {row['precision_at_5']}")
        print(f"  Recall@5:    {row['recall_at_5']}")
        print(f"  NDCG@5:      {row['ndcg_at_5']}")
        print(f"  Recommended: {row['recommended_titles']}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Spider-Sense Recommender")
    print("  A hybrid recommender system for Spider-Man media")
    print("=" * 60)
    print()

    items, interactions, content_rec, collab_rec, profile_rec, hybrid_rec = (
        build_recommenders()
    )

    while True:
        print("Choose a mode:")
        print("  1. Query-based recommendation")
        print("  2. Liked/disliked profile recommendation")
        print("  3. Synthetic user recommendation")
        print("  4. Hybrid recommendation")
        print("  5. Run evaluation demo")
        print("  q. Quit")

        choice = input("\nEnter your choice: ").strip().lower()
        print()

        if choice == "1":
            mode_query(content_rec)
        elif choice == "2":
            mode_profile(profile_rec)
        elif choice == "3":
            mode_synthetic_user(collab_rec)
        elif choice == "4":
            mode_hybrid(hybrid_rec)
        elif choice == "5":
            mode_evaluation(hybrid_rec)
        elif choice == "q":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Enter 1–5 or q.")

        print()


if __name__ == "__main__":
    main()
