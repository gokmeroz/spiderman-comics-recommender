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
