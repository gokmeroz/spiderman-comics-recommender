# Spider-Man Comics Recommender

A hybrid recommendation system for Spider-Man movies, games, comics, animated films, and shows.

## Why this project exists

This project was built as a one-day AI/ML case study to learn recommender-system fundamentals for New Grad AI Engineering interviews. It starts with a simple content-based TF-IDF baseline and evolves version by version into a full hybrid recommender with offline evaluation metrics.

## What it teaches

- Content-based filtering
- TF-IDF vectorization
- Cosine similarity
- User profile vectors
- Collaborative filtering
- Hybrid ranking
- Score normalization
- Popularity fallback
- Precision@K
- Recall@K
- NDCG@K
- Recommendation explanations

## Project architecture

```
items.csv
  ↓
feature engineering (item_text)
  ↓
TF-IDF item vectors
  ↓
content-based recommender       ←── query or liked/disliked titles
  ↓
user profile personalization

interactions.csv
  ↓
user-item matrix
  ↓
collaborative filtering         ←── known synthetic user

content score + collaborative score + popularity score
  ↓
hybrid ranker (weighted sum)
  ↓
top K recommendations
  ↓
explanations
  ↓
evaluation (Precision@K, Recall@K, NDCG@K)
```

## Version roadmap

| Version | What it builds |
|---------|----------------|
| 0 | Project setup and dataset creation |
| 1 | Content-based recommender using TF-IDF and cosine similarity |
| 2 | User profile recommender using liked/disliked items |
| 3 | Collaborative filtering simulation using synthetic user ratings |
| 4 | Hybrid recommender combining content, collaborative, and popularity signals |
| 5 | Evaluation using Precision@K, Recall@K, and NDCG@K |
| 6 | CLI demo, README polish, and interview explanation |

## How to run

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
python app/demo.py
```

## Version 1: Content-based recommender

The user enters a free-text query like `dark venom symbiote game`. The system converts all item metadata into TF-IDF vectors and computes cosine similarity between the query vector and every item vector. Returns the top K most similar items. Works for brand-new users with zero history — no ratings required.

## Version 2: User profile recommender

Instead of a query, the user provides liked and disliked titles. The system averages the TF-IDF vectors of liked items into a single user profile vector, then subtracts the average of disliked item vectors to steer away from unwanted content. Items are ranked by cosine similarity to this vector. Already-seen items are excluded.

## Version 3: Collaborative filtering

Uses synthetic user ratings from `interactions.csv`. Builds a user-item matrix (rows = users, columns = items, values = ratings 1–5, 0 = unrated), computes cosine similarity between users, and recommends items liked by the most similar users. Can surface non-obvious connections without reading item tags — but breaks for new users with no ratings (cold-start problem).

## Version 4: Hybrid recommender

Combines all three signals:

```
final_score = 0.60 * content_score
            + 0.30 * collaborative_score
            + 0.10 * popularity_score
```

All scores are normalized to 0–1 before combining so no single signal dominates by scale. Supports three modes: query-only cold start, liked/disliked profile, and known synthetic user with optional query.

## Version 5: Evaluation

Three offline ranking metrics:

- **Precision@K** — of the top K recommendations, what fraction were relevant?
- **Recall@K** — of all relevant items in the catalog, what fraction did we find?
- **NDCG@K** — did we rank relevant items near the top? (order-sensitive)

Evaluation results on the built-in test cases:

| Test | Precision@5 | Recall@5 | NDCG@5 |
|------|-------------|----------|--------|
| dark venom | 0.80 | 0.80 | 0.87 |
| miles multiverse | 0.80 | 1.00 | 0.98 |
| emotional peter | 0.60 | 0.75 | 0.80 |

## Example output

```
$ python app/demo.py

Choose a mode:
  1. Query-based recommendation
  ...

Enter your choice: 1
Enter your query: dark venom symbiote

1. Venom (comic, dark)
   Score: 1.0000
   Why:   Recommended because it matches your interest in: dark, venom.

2. Spider-Man: Web of Shadows (game, dark)
   Score: 0.8938
   Why:   Recommended because it matches your interest in: dark, symbiote, venom.
```

## Interview explanation

Spider-Sense Recommender is a hybrid recommendation system for Spider-Man media. I started with a content-based TF-IDF baseline, then added user profile vectors, synthetic collaborative filtering, and a hybrid ranker that combines content similarity, collaborative behavior, and popularity. I evaluated the recommender using Precision@K, Recall@K, and NDCG@K, and added human-readable explanations for why each recommendation was returned.

## What I would improve next

- Replace TF-IDF with sentence embeddings for richer semantic matching
- Use real user feedback instead of synthetic ratings
- Add item-to-item recommendation ("if you liked X, try Y")
- Tune hybrid weights using held-out validation data
- Add A/B testing infrastructure for online evaluation
- Add a small web UI with Streamlit
- Fetch real metadata from public APIs
