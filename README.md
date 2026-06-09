# Spiderman-Comics Recommender

A hybrid recommendation system for Spider-Man movies, games, comics, animated films, and shows.

## Purpose

This project was built as a one-day AI/ML case study to learn recommender-system fundamentals for New Grad AI Engineering interviews. It starts with a simple content-based baseline and evolves into a full hybrid recommender with evaluation metrics.

## Version roadmap

| Version | What it builds |
|---------|---------------|
| 0 | Project setup and dataset creation |
| 1 | Content-based recommender using TF-IDF and cosine similarity |
| 2 | User profile recommender using liked/disliked items |
| 3 | Collaborative filtering simulation using synthetic user ratings |
| 4 | Hybrid recommender combining content, collaborative, and popularity signals |
| 5 | Evaluation using Precision@K, Recall@K, and NDCG@K |
| 6 | CLI demo, README polish, and interview explanation |

## How to install

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run tests

```bash
pytest
```

## Version 1: Content-based recommender

The user enters a text query describing their taste. The system converts both the query and all item metadata into TF-IDF vectors, then ranks items by cosine similarity to the query vector. This works for brand-new users with no history — it only needs the item catalog.

## Version 2: User profile recommender

Instead of typing a query, the user provides liked and disliked titles. The system builds a user profile vector by averaging the TF-IDF vectors of liked items, then subtracting the average of disliked item vectors. Items are ranked by cosine similarity to this profile vector. Already-rated items are excluded from results.

## Version 3: Collaborative filtering

The system now uses synthetic user ratings from `interactions.csv`. It builds a user-item matrix, computes cosine similarity between users, and recommends items liked by the most similar users. This can surface non-obvious recommendations without ever reading item tags — but it breaks for brand-new users with no ratings (the cold-start problem).

## Version 4: Hybrid recommender

Combines all three signals into one ranked list using a weighted formula: `final_score = 0.60 * content_score + 0.30 * collaborative_score + 0.10 * popularity_score`. All scores are normalized to 0–1 before combining. Supports three modes: query-only cold start, liked/disliked profile, and known synthetic user.

## Version 5: Evaluation

Adds three offline ranking metrics — Precision@K, Recall@K, and NDCG@K. Precision@K measures how many of the top K results were relevant. Recall@K measures how many relevant items were recovered. NDCG@K measures whether relevant items were ranked near the top. Includes built-in test cases for evaluating the hybrid recommender.
