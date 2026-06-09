# CLAUDE.md

# Spider-Sense Recommender — Claude Code Project Instructions

You are building a beginner-friendly but interview-ready AI/ML project called **Spider-Sense Recommender**.

This is a hybrid recommendation system for Spider-Man movies, games, comics, animated films, and shows. The project exists to teach the fundamentals of recommender systems for New Grad AI Engineering interviews.

The user wants to learn the project deeply, not just receive working code.

Therefore, your job is to:

1. Build this project **one version at a time**.
2. Write **all code yourself**.
3. Add **clear educational comments** explaining what each important line or block does.
4. Avoid overly clever abstractions.
5. Stop after each version.
6. Ask the user concept-check questions after each version.
7. Do **not continue to the next version** until the user answers and shows they understand the current version.

The project should evolve through these versions:

- Version 0: Project setup and dataset creation
- Version 1: Content-based recommender using TF-IDF and cosine similarity
- Version 2: User profile recommender using liked/disliked items
- Version 3: Collaborative filtering simulation using synthetic user ratings
- Version 4: Hybrid recommender combining content, collaborative, and popularity signals
- Version 5: Evaluation using Precision@K, Recall@K, and NDCG@K
- Version 6: CLI demo, README polish, and interview explanation

---

# Absolute workflow rule

You must work in strict version order.

Do not implement multiple versions in the same step.

After finishing each version:

1. Summarize exactly what you built.
2. List files created or modified.
3. Explain the AI/ML concepts covered.
4. Show how to run/test the version.
5. Ask the user quiz questions.
6. Wait for the user before continuing.

Your stopping format after each version must be:

```text
VERSION X COMPLETE.

What was built:
...

Files changed:
...

How to run:
...

Concepts covered:
...

Before we continue to Version Y, answer these questions:
1. ...
2. ...
3. ...
```

Do not continue until the user answers.

If the user answers incorrectly or vaguely, explain the concept again and ask a simpler follow-up question.

---

# Project purpose

The final project should demonstrate that the user understands the full recommender-system pipeline:

```text
Raw item data
  ↓
Feature engineering
  ↓
Vectorization
  ↓
Similarity search
  ↓
Personalization
  ↓
Collaborative filtering
  ↓
Hybrid ranking
  ↓
Top-K recommendations
  ↓
Explanations
  ↓
Evaluation
```

This should be suitable to discuss in a New Grad AI Engineering interview.

The final interview story should be:

```text
SpiderMan Comics Recommender is a hybrid recommendation system for Spider-Man media. I started with a content-based TF-IDF baseline, then added user profile vectors, synthetic collaborative filtering, and a hybrid ranker that combines content similarity, collaborative behavior, and popularity. I evaluated the recommender using Precision@K, Recall@K, and NDCG@K, and added human-readable explanations for why each recommendation was returned.
```

---

# Tech stack

Use:

- Python 3.10+
- pandas
- numpy
- scikit-learn
- pytest
- rich, optional, only if useful for CLI formatting

Do not use:

- Deep learning
- TensorFlow
- PyTorch
- External APIs
- Web scraping
- Heavy frameworks
- Databases
- LangChain
- LlamaIndex
- Streamlit unless the user explicitly asks later

This is a one-day educational ML project. Keep it simple, clear, and explainable.

---

# Required final folder structure

Build toward this structure:

```text
spider-sense-recommender/
│
├── CLAUDE.md
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── raw/
│   │   ├── items.csv
│   │   └── interactions.csv
│   │
│   └── processed/
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── content_recommender.py
│   ├── user_profile.py
│   ├── collaborative_filtering.py
│   ├── hybrid_ranker.py
│   ├── evaluation.py
│   └── explain.py
│
├── app/
│   └── demo.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_content_recommender.py
│   ├── test_user_profile.py
│   ├── test_collaborative_filtering.py
│   ├── test_hybrid_ranker.py
│   └── test_evaluation.py
│
└── experiments/
    └── .gitkeep
```

Only create files needed for the current version. Do not create empty future implementation files unless needed for imports or structure.

---

# Coding style rules

Write code as if teaching a beginner who knows software engineering but is new to AI/ML.

Use:

- Type hints
- Docstrings
- Small functions
- Clear names
- Explicit intermediate variables
- Educational comments
- Simple math
- Deterministic behavior where possible

Avoid:

- Magic one-liners
- Over-abstracted classes
- Global mutable state
- Hidden side effects
- Silent failures
- Unexplained ML terminology
- Too many dependencies

Every important ML operation must have comments explaining:

1. What the operation does.
2. Why it is needed.
3. What concept it teaches.

Example style:

```python
# TF-IDF turns text into numeric vectors.
# A high TF-IDF value means a word is important for one item
# but not too common across every item.
vectorizer = TfidfVectorizer(stop_words="english")
```

---

# Data design

The project uses two CSV files.

## `data/raw/items.csv`

This file stores Spider-Man media items.

Required columns:

```text
item_id
title
medium
era
tone
villain
theme
tags
popularity
```

Column meanings:

- `item_id`: unique integer ID
- `title`: media title
- `medium`: movie, game, comic, animated_film, animated_show, live_action_show
- `era`: raimi, webb, mcu, modern, classic, 2000s, future, etc.
- `tone`: emotional, dark, funny, hopeful, reflective, cinematic, etc.
- `villain`: main villain or antagonist
- `theme`: core story theme
- `tags`: free-text metadata used for recommendation
- `popularity`: synthetic 0–100 score used later as a fallback signal

Use this dataset or a very close equivalent:

```csv
item_id,title,medium,era,tone,villain,theme,tags,popularity
1,Spider-Man: Into the Spider-Verse,animated_film,modern,emotional,Kingpin,multiverse,"animation miles morales multiverse family identity coming-of-age leap of faith",96
2,Spider-Man: Across the Spider-Verse,animated_film,modern,emotional,Spot,multiverse,"miles morales gwen stacy multiverse canon events identity animation",94
3,Spider-Man 2,movie,raimi,emotional,Doctor Octopus,responsibility,"peter parker sacrifice mentor science tragedy responsibility train scene",93
4,Spider-Man,movie,raimi,classic,Green Goblin,origin,"peter parker origin uncle ben responsibility goblin mary jane",90
5,Spider-Man 3,movie,raimi,chaotic,Venom,ego,"black suit venom sandman forgiveness revenge emo peter",72
6,The Amazing Spider-Man,movie,webb,romantic,Lizard,origin,"gwen stacy peter parker science lonely skateboard origin",76
7,The Amazing Spider-Man 2,movie,webb,tragic,Electro,loss,"gwen stacy electro green goblin tragedy romance sacrifice",70
8,Spider-Man: Homecoming,movie,mcu,funny,Vulture,coming-of-age,"teen peter parker high school iron man vulture friendly neighborhood",86
9,Spider-Man: Far From Home,movie,mcu,adventure,Mysterio,illusion,"mysterio europe grief tony stark illusion responsibility",82
10,Spider-Man: No Way Home,movie,mcu,nostalgic,Green Goblin,multiverse,"multiverse tobey andrew nostalgia sacrifice villains redemption",95
11,Spider-Man PS4,game,modern,cinematic,Mister Negative,responsibility,"open world peter parker aunt may mature story swinging combat",91
12,Spider-Man: Miles Morales,game,modern,hopeful,Rhino,identity,"miles morales harlem family music electricity identity winter",88
13,Spider-Man 2 PS5,game,modern,dark,Venom,symbiote,"venom kraven black suit peter miles symbiote open world",90
14,Spider-Man: Web of Shadows,game,2000s,dark,Venom,symbiote,"black suit venom infection choices dark action symbiote invasion",80
15,Ultimate Spider-Man,game,2000s,comic-like,Venom,dual-perspective,"venom eddie brock comic style peter parker playable villain",78
16,Spider-Man: Shattered Dimensions,game,2000s,adventure,Mysterio,multiverse,"noir 2099 ultimate amazing dimensions multiple spider-men",77
17,Spider-Man 2099,comic,future,dark,Alchemax,dystopia,"miguel ohara cyberpunk future genetics corporation antihero",82
18,Kraven's Last Hunt,comic,classic,dark,Kraven,identity,"psychological dark burial obsession survival kraven peter parker",89
19,Ultimate Spider-Man Comic,comic,2000s,modern,Green Goblin,origin,"brian michael bendis teen peter parker modern origin high school",84
20,Spider-Man: Life Story,comic,modern,reflective,Multiple,aging,"decades aging peter parker responsibility life choices emotional",87
21,Superior Spider-Man,comic,modern,complex,Doctor Octopus,identity,"doctor octopus becomes spider-man antihero morality intelligence",83
22,Spider-Man Blue,comic,classic,romantic,Green Goblin,grief,"gwen stacy mary jane memory love loss emotional reflective",81
23,Venom,comic,modern,dark,Venom,symbiote,"eddie brock symbiote antihero dark alien bond",79
24,The Spectacular Spider-Man,animated_show,2000s,balanced,Green Goblin,teen-life,"school peter parker villains romance action serialized animation",85
25,Spider-Man The Animated Series,animated_show,classic,adventure,Multiple,heroism,"90s cartoon venom carnage kingpin madame web classic villains",80
26,Spider-Man Unlimited,animated_show,classic,weird,High Evolutionary,alternate-world,"counter earth futuristic strange alternate dimension",61
27,Spider-Man Noir,comic,classic,dark,Goblin,detective,"noir detective depression 1930s gritty dark vigilante",76
28,Spider-Gwen,comic,modern,stylish,Kingpin,identity,"gwen stacy alternate universe music drums identity punk",82
29,Marvel's Midnight Suns Spider-Man,game,modern,tactical,Venom,teamwork,"tactical rpg magic team heroes venom combat cards",69
30,Spider-Man Friend or Foe,game,2000s,light,Multiple,team-up,"co-op villains allies light funny arcade action",65
```

## `data/raw/interactions.csv`

This file stores synthetic ratings from fake user personas.

Required columns:

```text
user_id
item_id
rating
```

Rating scale:

```text
1 = strongly dislike
2 = dislike
3 = neutral
4 = like
5 = love
```

Use this dataset or a very close equivalent:

```csv
user_id,item_id,rating
dark_venom_fan,13,5
dark_venom_fan,14,5
dark_venom_fan,15,5
dark_venom_fan,18,4
dark_venom_fan,23,5
dark_venom_fan,8,2
dark_venom_fan,30,2
miles_fan,1,5
miles_fan,2,5
miles_fan,12,5
miles_fan,28,4
miles_fan,17,3
miles_fan,5,2
raimi_fan,3,5
raimi_fan,4,5
raimi_fan,5,4
raimi_fan,10,5
raimi_fan,22,4
raimi_fan,26,1
comic_purist,18,5
comic_purist,20,5
comic_purist,21,5
comic_purist,22,5
comic_purist,27,4
comic_purist,8,2
gamer,11,5
gamer,12,5
gamer,13,5
gamer,14,4
gamer,15,4
gamer,29,4
gamer,22,2
multiverse_fan,1,5
multiverse_fan,2,5
multiverse_fan,10,5
multiverse_fan,16,5
multiverse_fan,28,4
multiverse_fan,26,3
emotional_peter_fan,3,5
emotional_peter_fan,10,5
emotional_peter_fan,11,5
emotional_peter_fan,20,5
emotional_peter_fan,22,5
emotional_peter_fan,30,1
future_cyberpunk_fan,17,5
future_cyberpunk_fan,27,4
future_cyberpunk_fan,16,4
future_cyberpunk_fan,26,4
future_cyberpunk_fan,14,3
future_cyberpunk_fan,8,2
animation_fan,1,5
animation_fan,2,5
animation_fan,24,5
animation_fan,25,4
animation_fan,28,4
animation_fan,29,2
light_fun_fan,8,5
light_fun_fan,9,4
light_fun_fan,24,4
light_fun_fan,30,5
light_fun_fan,14,1
light_fun_fan,18,1
```

---

# Version 0 — Project setup and dataset creation

## Goal

Create the basic project structure, dependency files, datasets, and data-loading utilities.

Do not implement recommendation logic yet.

## Files to create

```text
requirements.txt
pyproject.toml
README.md
data/raw/items.csv
data/raw/interactions.csv
data/processed/.gitkeep
src/__init__.py
src/data_loader.py
tests/test_data_loader.py
experiments/.gitkeep
```

## `requirements.txt`

Include:

```text
pandas
numpy
scikit-learn
pytest
```

## `pyproject.toml`

Create a minimal pytest configuration.

Example:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

## `README.md`

Create a starter README with:

- Project name
- One-paragraph purpose
- Version roadmap
- How to install dependencies
- How to run tests

Keep it short for now. Expand it in Version 6.

## `src/data_loader.py`

Implement:

```python
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_items() -> pd.DataFrame:
    ...


def load_interactions() -> pd.DataFrame:
    ...
```

Requirements:

- Use `Path`, not hardcoded string paths.
- Validate required columns.
- Raise clear `ValueError` if columns are missing.
- Add comments explaining why data validation matters in ML projects.

## Tests

Create tests that verify:

1. `items.csv` loads successfully.
2. `interactions.csv` loads successfully.
3. Required columns exist.
4. `items.csv` has at least 20 rows.
5. `interactions.csv` has at least 8 unique users.

## Version 0 explanation requirements

After completing Version 0, explain:

- What a dataset is in ML
- Why raw data should be kept separate from processed data
- Why validating columns matters
- What `items.csv` represents
- What `interactions.csv` represents

## Version 0 quiz questions

Ask the user these questions:

1. What is the difference between `items.csv` and `interactions.csv`?
2. Why do we keep `data/raw/` separate from `data/processed/`?
3. Why should we validate required columns before training or recommending?
4. In recommender-system language, what does an “item” mean in this project?

Stop after asking these questions.

---

# Version 1 — Content-based recommender

## Goal

Implement the first real recommender.

This version recommends Spider-Man media based on a text query like:

```text
dark venom symbiote game
```

It should return items whose metadata is textually similar to the query.

## AI/ML concepts covered

- Feature engineering
- Text representation
- TF-IDF
- Vectors
- Cosine similarity
- Content-based recommendation
- Cold start

## Files to create or modify

```text
src/features.py
src/content_recommender.py
src/explain.py
tests/test_features.py
tests/test_content_recommender.py
README.md
```

## `src/features.py`

Implement:

```python
def build_item_text(items: pd.DataFrame) -> pd.DataFrame:
    ...
```

This function should:

1. Copy the input DataFrame.
2. Combine these columns into a new `item_text` column:
   - title
   - medium
   - era
   - tone
   - villain
   - theme
   - tags

3. Fill missing values with empty strings.
4. Normalize whitespace.
5. Return the modified DataFrame.

Add educational comments explaining:

- Raw columns are human-readable metadata.
- ML models need numeric or structured representation.
- `item_text` is an engineered feature.
- We combine metadata so TF-IDF has more signal.

## `src/explain.py`

Implement a simple explanation helper:

```python
def explain_content_match(query: str, item_row: pd.Series) -> str:
    ...
```

The explanation should be simple and deterministic.

Example:

```text
Recommended because it matches your interest in dark tone, Venom, and symbiote stories.
```

Do not use LLMs or APIs.

Use basic keyword overlap between the query and the item metadata.

## `src/content_recommender.py`

Implement:

```python
class ContentBasedRecommender:
    def __init__(self, items: pd.DataFrame) -> None:
        ...

    def recommend_by_query(self, query: str, top_k: int = 5) -> pd.DataFrame:
        ...
```

Implementation requirements:

1. Use `build_item_text`.
2. Use `TfidfVectorizer(stop_words="english")`.
3. Fit the vectorizer on item text.
4. Transform the user query using the same vectorizer.
5. Use `cosine_similarity` between query vector and item vectors.
6. Sort descending by similarity.
7. Return top K rows.
8. Include columns:
   - item_id
   - title
   - medium
   - tone
   - villain
   - similarity_score
   - explanation

Important educational comments:

- Explain what TF-IDF does.
- Explain why the same vectorizer must be used for items and query.
- Explain what cosine similarity measures.
- Explain why this solves cold start for a brand-new user.
- Explain why content-based systems can over-specialize.

## Tests

Create tests that verify:

1. `build_item_text` creates the `item_text` column.
2. `item_text` contains title/tone/villain/tags information.
3. `recommend_by_query("dark venom symbiote game")` returns at least one result.
4. Top results include at least one Venom/symbiote-related item.
5. Returned DataFrame has `similarity_score`.
6. Returned DataFrame has `explanation`.

## README update

Add a Version 1 section:

```text
Version 1: Content-based recommender
```

Explain:

- User enters a text query.
- The system converts item metadata and the query into TF-IDF vectors.
- It uses cosine similarity to rank items.
- This is useful for cold-start users.

## Version 1 quiz questions

Ask the user:

1. What problem does TF-IDF solve in this project?
2. Why do we combine title, medium, tone, villain, theme, and tags into `item_text`?
3. What does cosine similarity compare?
4. Why is this called content-based recommendation?
5. Why can V1 work even for a brand-new user with no ratings?
6. What is one weakness of a pure content-based recommender?

Stop after asking these questions.

---

# Version 2 — User profile recommender

## Goal

Upgrade from query-only recommendations to personalized recommendations from liked and disliked items.

Instead of only this:

```text
dark venom symbiote game
```

The user can provide:

```python
liked_titles = ["Spider-Man 2", "Spider-Man: Into the Spider-Verse"]
disliked_titles = ["Spider-Man: Web of Shadows"]
```

The system should build a user preference vector and recommend unseen items.

## AI/ML concepts covered

- User profile vector
- Personalization
- Averaging item vectors
- Positive and negative preferences
- Excluding already-seen items
- Preference modeling

## Files to create or modify

```text
src/user_profile.py
tests/test_user_profile.py
README.md
```

## `src/user_profile.py`

Implement:

```python
class UserProfileRecommender:
    def __init__(self, content_recommender: ContentBasedRecommender) -> None:
        ...

    def recommend_by_user_profile(
        self,
        liked_titles: list[str],
        disliked_titles: list[str] | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        ...
```

Implementation requirements:

1. Reuse the already-fitted content recommender.
2. Find item vectors for liked titles.
3. Average liked item vectors.
4. If disliked titles exist:
   - Find disliked item vectors.
   - Average disliked vectors.
   - Subtract disliked vector from liked vector.

5. Compare user vector to all item vectors using cosine similarity.
6. Exclude already liked/disliked titles.
7. Return top K recommendations.
8. Include:
   - item_id
   - title
   - medium
   - tone
   - villain
   - profile_similarity_score
   - explanation

Important educational comments:

- Explain that the user is now represented as a vector.
- Explain that liked items pull the user vector toward similar items.
- Explain that disliked items push the user vector away from unwanted regions.
- Explain why already-rated items are excluded.
- Explain that this is still content-based because it uses item metadata vectors.

## Handling title lookup

Implement a helper method:

```python
def _get_item_indices_by_title(self, titles: list[str]) -> list[int]:
    ...
```

Rules:

- Matching can be case-insensitive exact title matching.
- If a title is unknown, raise a clear `ValueError`.
- The error should list available titles or suggest checking spelling.

## Tests

Create tests that verify:

1. Liked-title recommendations return results.
2. Already-liked titles are excluded.
3. Disliked titles are excluded.
4. Unknown title raises `ValueError`.
5. Output includes `profile_similarity_score`.
6. If user likes Miles-related items, recommendations should include at least one similar modern/multiverse/identity-related item.

## README update

Add a Version 2 section:

```text
Version 2: User profile recommender
```

Explain:

- The user rates or selects examples.
- Liked item vectors are averaged.
- Disliked item vectors are subtracted.
- The final vector represents the user’s taste.
- Recommendations are ranked by similarity to the user vector.

## Version 2 quiz questions

Ask the user:

1. What is a user profile vector?
2. Why do we average the vectors of liked items?
3. What does subtracting disliked item vectors do conceptually?
4. Why should already-liked or already-disliked items be excluded from recommendations?
5. Is Version 2 still content-based? Why or why not?
6. How is Version 2 more personalized than Version 1?

Stop after asking these questions.

---

# Version 3 — Collaborative filtering simulation

## Goal

Add a synthetic collaborative filtering recommender.

This version should recommend items based on similar users, not just item metadata.

Example:

```text
If dark_venom_fan liked Web of Shadows, Ultimate Spider-Man, and Venom,
then recommend items liked by users with similar rating patterns.
```

## AI/ML concepts covered

- User-item interaction matrix
- Ratings
- Sparse data
- User-user similarity
- Collaborative filtering
- Behavioral recommendation
- Similar-users logic

## Files to create or modify

```text
src/collaborative_filtering.py
tests/test_collaborative_filtering.py
README.md
```

## `src/collaborative_filtering.py`

Implement:

```python
class CollaborativeFilteringRecommender:
    def __init__(self, items: pd.DataFrame, interactions: pd.DataFrame) -> None:
        ...

    def recommend_by_similar_users(self, user_id: str, top_k: int = 5) -> pd.DataFrame:
        ...
```

Implementation requirements:

1. Build a user-item matrix:
   - Rows = users
   - Columns = item IDs
   - Values = ratings
   - Missing ratings = 0

2. Find the target user's rating vector.
3. Compute cosine similarity between target user and all other users.
4. Exclude the target user from similar users.
5. Select top similar users.
6. Compute candidate item scores from similar users' ratings weighted by similarity.
7. Exclude items already rated by the target user.
8. Return top K items.
9. Include:
   - item_id
   - title
   - medium
   - tone
   - villain
   - collaborative_score
   - similar_users
   - explanation

Important educational comments:

- Explain what a user-item matrix is.
- Explain why missing values are filled with 0.
- Explain that collaborative filtering uses behavior, not item text.
- Explain why sparse data is a real-world issue.
- Explain how similar users generate candidate recommendations.
- Explain the difference between V1/V2 and V3.

## Similar users

Add a helper method:

```python
def get_similar_users(self, user_id: str, top_n: int = 3) -> pd.DataFrame:
    ...
```

Return:

```text
user_id
similarity
```

## Unknown user handling

If `user_id` is not in the interaction dataset, raise a clear `ValueError` listing available user IDs.

## Tests

Create tests that verify:

1. User-item matrix is created.
2. Known user gets recommendations.
3. Unknown user raises `ValueError`.
4. Already-rated items are excluded.
5. `get_similar_users` does not return the target user.
6. Output includes `collaborative_score`.
7. Output includes `similar_users`.

## README update

Add a Version 3 section:

```text
Version 3: Collaborative filtering
```

Explain:

- The system now uses synthetic user ratings.
- It finds users with similar rating patterns.
- It recommends items liked by those similar users.
- This can discover recommendations not obvious from tags alone.
- The weakness is that it needs interaction data.

## Version 3 quiz questions

Ask the user:

1. What is a user-item matrix?
2. What do rows and columns represent in our interaction matrix?
3. How is collaborative filtering different from content-based filtering?
4. Why do we exclude items the user has already rated?
5. What does cosine similarity compare in Version 3?
6. What is the cold-start problem for collaborative filtering?

Stop after asking these questions.

---

# Version 4 — Hybrid recommender

## Goal

Combine content-based signals, collaborative filtering signals, and popularity into one final ranking system.

This is the most interview-realistic recommender version.

## AI/ML concepts covered

- Hybrid recommendation
- Score normalization
- Weighted ranking
- Candidate merging
- Popularity fallback
- Ranking tradeoffs
- Production-style recommender thinking

## Files to create or modify

```text
src/hybrid_ranker.py
tests/test_hybrid_ranker.py
README.md
```

## Hybrid scoring formula

Use:

```python
final_score = (
    0.60 * content_score +
    0.30 * collaborative_score +
    0.10 * popularity_score
)
```

Normalize all score components to 0–1 before combining.

## `src/hybrid_ranker.py`

Implement:

```python
class HybridRecommender:
    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        collaborative_recommender: CollaborativeFilteringRecommender,
    ) -> None:
        ...

    def recommend(
        self,
        query: str | None = None,
        user_id: str | None = None,
        liked_titles: list[str] | None = None,
        disliked_titles: list[str] | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        ...
```

The hybrid recommender should support three usage modes:

## Mode 1: Query-only cold start

Input:

```python
query="dark venom symbiote game"
```

Use:

- content score from query
- popularity score
- collaborative score = 0

## Mode 2: Liked/disliked profile

Input:

```python
liked_titles=["Spider-Man 2", "Spider-Man: Into the Spider-Verse"]
disliked_titles=["Spider-Man: Web of Shadows"]
```

Use:

- content/profile score
- popularity score
- collaborative score = 0 unless user_id is also given

## Mode 3: Known synthetic user

Input:

```python
user_id="dark_venom_fan"
query="dark venom symbiote"
```

Use:

- content score from query if query exists
- collaborative score from similar users
- popularity score

## Output columns

Return:

```text
item_id
title
medium
tone
villain
content_score
collaborative_score
popularity_score
final_score
explanation
```

## Implementation requirements

1. Create one row per item.
2. Compute content score if query or liked titles exist.
3. Compute collaborative score if `user_id` exists.
4. Normalize popularity from 0–100 to 0–1.
5. Normalize content and collaborative scores safely.
6. Exclude already liked/disliked items if provided.
7. Exclude already-rated items for a known synthetic user.
8. Sort by `final_score` descending.
9. Return top K.
10. Generate explanations that mention strongest signals.

## Helper functions

Implement private helper methods:

```python
def _normalize_score(self, values: pd.Series) -> pd.Series:
    ...

def _build_empty_score_frame(self) -> pd.DataFrame:
    ...

def _merge_scores(self, ...) -> pd.DataFrame:
    ...
```

Keep them simple and readable.

## Educational comments

Explain:

- Why hybrid systems are useful.
- Why content-based works for cold start.
- Why collaborative filtering works when behavior data exists.
- Why popularity is a fallback, not the main intelligence.
- Why score normalization is necessary before combining scores.
- Why weights are product decisions, not universal truths.

## Tests

Create tests that verify:

1. Query-only hybrid recommendations work.
2. User-id hybrid recommendations work.
3. Liked/disliked hybrid recommendations work.
4. Output contains all score columns.
5. Final score is between 0 and 1.
6. Recommendations are sorted by final score descending.
7. Already-rated or excluded items do not appear.

## README update

Add a Version 4 section:

```text
Version 4: Hybrid recommender
```

Explain:

- The final ranker combines multiple signals.
- Content score captures metadata similarity.
- Collaborative score captures similar-user behavior.
- Popularity score gives a safe fallback.
- Weighted ranking is a product/design choice.

## Version 4 quiz questions

Ask the user:

1. Why do we need to normalize scores before combining them?
2. What does the content score represent?
3. What does the collaborative score represent?
4. Why is popularity only 10% of the final score?
5. What does a hybrid recommender solve that pure content-based or pure collaborative filtering cannot solve alone?
6. If this was a real product, how would you choose or tune the weights?

Stop after asking these questions.

---

# Version 5 — Evaluation

## Goal

Add ranking evaluation metrics.

This version should help the user answer:

```text
Are the recommendations actually good?
```

## AI/ML concepts covered

- Top-K evaluation
- Relevance
- Precision@K
- Recall@K
- DCG
- IDCG
- NDCG@K
- Offline evaluation
- Holdout-style testing

## Files to create or modify

```text
src/evaluation.py
tests/test_evaluation.py
README.md
```

## `src/evaluation.py`

Implement:

```python
def precision_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    ...


def recall_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    ...


def ndcg_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    ...


def evaluate_recommender(
    recommender,
    test_cases: list[dict],
    k: int = 5,
) -> pd.DataFrame:
    ...
```

## Metric explanations to encode in comments

Precision@K:

```text
Of the top K recommendations, what fraction were relevant?
```

Recall@K:

```text
Of all relevant items, what fraction did we recover in the top K?
```

NDCG@K:

```text
Did we rank relevant items near the top?
```

## DCG and IDCG

Implement NDCG manually.

Use binary relevance:

- relevant item = 1
- irrelevant item = 0

Formula idea:

```python
dcg = sum(relevance_at_position / log2(position + 1))
ndcg = dcg / ideal_dcg
```

Explain in comments:

- DCG rewards relevant items higher in the ranking.
- IDCG is the best possible DCG.
- NDCG compares your ranking to the ideal ranking.
- NDCG is between 0 and 1.

## Test cases

Create default test cases in code or README examples like:

```python
test_cases = [
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
```

## Tests

Create tests that verify:

1. `precision_at_k` returns correct value on simple examples.
2. `recall_at_k` returns correct value on simple examples.
3. `ndcg_at_k` returns 1.0 for ideal ranking.
4. `ndcg_at_k` returns lower value for worse ranking.
5. Empty relevant set is handled safely.
6. Evaluation returns a DataFrame.
7. Evaluation DataFrame has metric columns.

## README update

Add a Version 5 section:

```text
Version 5: Evaluation
```

Explain:

- Recommendations are ranked lists.
- Accuracy is not the right metric.
- Precision@K checks quality of the top K.
- Recall@K checks coverage of relevant items.
- NDCG@K checks ranking order.
- Offline evaluation is imperfect but useful.

## Version 5 quiz questions

Ask the user:

1. Why is normal classification accuracy not the best metric for a recommender?
2. What does Precision@5 mean?
3. What does Recall@5 mean?
4. Why does NDCG care about item order?
5. What is the difference between relevant items and recommended items?
6. Why is offline evaluation not enough for a real production recommender?

Stop after asking these questions.

---

# Version 6 — CLI demo, README polish, and interview explanation

## Goal

Create a polished demo and final project documentation.

This version should make the project easy to run, understand, and explain in interviews.

## Files to create or modify

```text
app/demo.py
README.md
tests/test_cli_smoke.py
```

Only create `tests/test_cli_smoke.py` if it is practical and does not make the project unnecessarily complicated.

## `app/demo.py`

Create a simple command-line interface.

When run:

```bash
python app/demo.py
```

Show:

```text
Welcome to Spider-Sense Recommender 🕷️

Choose mode:
1. Query-based recommendation
2. Liked/disliked profile recommendation
3. Synthetic user recommendation
4. Hybrid recommendation
5. Run evaluation demo
```

## CLI mode 1: Query-based

Prompt:

```text
Enter your Spider-Verse taste:
```

Example input:

```text
dark venom symbiote game
```

Output:

```text
Top recommendations:

1. Spider-Man: Web of Shadows
Score: 0.91
Why: Matches your interest in dark tone, Venom, and symbiote stories.
```

## CLI mode 2: Liked/disliked profile

Prompt:

```text
Enter liked titles separated by commas:
Enter disliked titles separated by commas, or leave blank:
```

Use `UserProfileRecommender`.

## CLI mode 3: Synthetic user

Show available users:

```text
dark_venom_fan
miles_fan
raimi_fan
comic_purist
gamer
multiverse_fan
emotional_peter_fan
future_cyberpunk_fan
animation_fan
light_fun_fan
```

Prompt for user ID.

Use collaborative filtering.

## CLI mode 4: Hybrid

Allow:

- Optional query
- Optional synthetic user ID
- Optional liked titles
- Optional disliked titles

Use `HybridRecommender`.

## CLI mode 5: Evaluation demo

Run built-in test cases and print:

```text
name
precision_at_5
recall_at_5
ndcg_at_5
recommended_titles
```

## CLI implementation rules

- Keep the CLI simple.
- Use plain Python `input()` and `print()`.
- Do not require external UI libraries.
- Handle invalid input gracefully.
- Add comments explaining which recommender each mode uses.

## README final version

The final README must include:

```md
# Spider-Sense Recommender

A hybrid recommendation system for Spider-Man movies, games, comics, animated films, and shows.

## Why this project exists

This project was built as a one-day AI/ML case study to learn recommender-system fundamentals for New Grad AI Engineering interviews.

## What it teaches

- Content-based filtering
- TF-IDF vectorization
- Cosine similarity
- User profile vectors
- Collaborative filtering
- Hybrid ranking
- Popularity fallback
- Precision@K
- Recall@K
- NDCG@K
- Recommendation explanations

## Project architecture

items.csv
↓
feature engineering
↓
TF-IDF item vectors
↓
content-based recommender
↓
user profile personalization

interactions.csv
↓
user-item matrix
↓
collaborative filtering

content score + collaborative score + popularity score
↓
hybrid ranker
↓
top K recommendations
↓
explanations
↓
evaluation

## Versions

### Version 1: Content-based recommender

Explain V1.

### Version 2: User profile recommender

Explain V2.

### Version 3: Collaborative filtering

Explain V3.

### Version 4: Hybrid recommender

Explain V4.

### Version 5: Evaluation

Explain V5.

## How to run

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python app/demo.py

## Windows

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
python app/demo.py

## Example output

Include example query and recommendations.

## Interview explanation

Spider-Sense Recommender is a hybrid recommendation system that starts with a content-based TF-IDF baseline, then adds user profile vectors, synthetic collaborative filtering, and a hybrid ranking formula. The goal was to understand the full recommender-system pipeline: data design, feature engineering, vectorization, similarity search, personalization, ranking, evaluation, and explanation.

## What I would improve next

- Replace TF-IDF with sentence embeddings
- Use real user feedback
- Add item-to-item recommendation pages
- Tune hybrid weights using validation data
- Add a small web UI
- Add real metadata from public APIs
- Add A/B testing for online evaluation
```

## Final Version 6 quiz questions

Ask the user:

1. How would you explain this project in a 60-second interview answer?
2. What are the five main stages of the recommender pipeline?
3. What is the difference between content score, collaborative score, and popularity score?
4. Why did we start with a simple baseline before adding hybrid ranking?
5. What would you improve first if this became a real product?
6. Which part of the project do you feel weakest explaining?

Stop after asking these questions.

---

# Final acceptance criteria

The project is complete only when:

1. The project structure exists.
2. `items.csv` has at least 20 Spider-Man media items.
3. `interactions.csv` has at least 8 synthetic users.
4. Version 1 query-based recommendations work.
5. Version 2 liked/disliked recommendations work.
6. Version 3 collaborative recommendations work.
7. Version 4 hybrid recommendations work.
8. Version 5 evaluation metrics work.
9. CLI demo runs.
10. Tests pass.
11. README explains the project clearly.
12. Each major ML concept is explained in code comments and README prose.
13. The user has been quizzed after every version.
14. You did not proceed between versions without user confirmation.

---

# How to begin

Start with Version 0 only.

Do not implement Version 1 yet.

Your first task:

1. Create the folder structure needed for Version 0.
2. Create dependency/config files.
3. Create `items.csv`.
4. Create `interactions.csv`.
5. Create `src/data_loader.py`.
6. Create `tests/test_data_loader.py`.
7. Run the tests.
8. Stop and quiz the user.

Remember: one version at a time.
