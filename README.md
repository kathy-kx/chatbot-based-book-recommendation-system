# Chatbot-Based Book Recommendation System

## Project Overview

A Conversational Recommender System (CRS) that provides personalized book recommendations through natural language interaction. The system uses **TF-IDF cosine similarity** for content-based recommendations and maintains **user preference memory** across conversation turns.

**Key Technologies:**
- **FastAPI** — REST API serving the recommendation engine
- **SpaCy** — Named Entity Recognition for preference extraction
- **N8N** — Chatbot orchestration (self-hosted via Docker, or N8N Cloud)
- **Scikit-learn** — TF-IDF vectorization and cosine similarity
- **Groq API** — Optional LLM response generation (Llama 3.3 70B); system runs fully without it

---

## Architecture

```
User
  │
  ▼
N8N Chat Interface
  │  POST /chat
  ▼
FastAPI /chat
  ├── spacy_ner.py       ← preference extraction + session memory merge
  ├── detect_intent()    ← route to one of 6 handlers
  ├── cosine_similarity.py  ← similar books + general search
  ├── genre pool + TF-IDF ranking  ← genre/preference queries
  ├── author / popularity / beginner filters  
  └── Groq LLM (optional) ← natural language response generation
  │
  ▼
JSON response → N8N → User
```

**Session memory** is maintained server-side in an in-memory dict keyed by `session_id`. Preferences (liked/disliked genres, authors, reading level) accumulate across turns and are used for filtering and boosting in every subsequent query.

---

## Project Structure

```
├── fastapi_app.py               # FastAPI server - intent detection, session memory, all CQ handlers
├── cosine_similarity.py         # Cosine similarity recommender (loaded by fastapi_app.py)
├── spacy_ner.py                 # SpaCy NER — preference extraction and session memory merge
│
├── books_with_clusters.csv      # 1,985 books with genre labels and cluster assignments
├── tfidf_matrix.npz             # Pre-computed TF-IDF sparse matrix (1985 × 5000)
├── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
├── Books.csv                    # Goodreads ratings data (for popularity ranking)
│
├── Classification.ipynb         # Genre classification models (BoW/TF-IDF/LDA/Word2Vec + LR/RF/SVM)
├── workflow.json                # N8N exported workflow
│
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── User_Guide.md                # Detailed setup and usage instructions
│
├── test_cases.md                # Competency questions (designed in assignment 4, for reference) and test cases
└── test_fastapi.py              # Unit and integration tests for Chatbot-Based Book Recommendation System
```

> **Jupyter Notebooks:** `Classification.ipynb` and `clustering/Clustering.ipynb` can be run in [Google Colab](https://colab.research.google.com/). Upload the notebook and the required CSV/NPZ/csv data files, then run all cells. No local Python environment needed.

---

## Two Ways to Run

### Option 1 — Cloud (N8N Cloud + Render)

A live demo is deployed and accessible without any local setup:

- **Chat interface:** hosted on N8N Cloud. [Try Our Live chatbot](https://kathy-kx.app.n8n.cloud/workflow/ps1bfjRpaJ3OWZUg/621d3b?projectId=ivnuR5X22yOOs7ij&uiContext=workflow_list)
- **Backend API:** deployed on [Render](https://chatbot-based-book-recommendation-system.onrender.com/chat) 

> **Note:** The Render free tier spins down after inactivity. The **first request after a period of idle may take 30–60 seconds** while the server cold-starts. Subsequent requests are fast.

The system is fully configured. All required environment variables (including LLM API keys) are securely managed on the backend, so users can immediately interact with the chatbot and receive generated responses without any setup.


### Option 2 — Local (Docker N8N + local FastAPI)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Start FastAPI
uvicorn fastapi_app:app --reload --port 8000 --host 0.0.0.0

# 3. Start N8N via Docker
docker run -d --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n n8nio/n8n

# 4. Open http://localhost:5678, import workflow.json, start chatting
```

See **[User_Guide.md](./User_Guide.md)** for detailed steps.


---

## Key Features

### 1. Content-Based Recommendation
- TF-IDF vectors represent book descriptions (ngram 1–2, 5,000 features)
- Cosine similarity scores semantic closeness between descriptions
- Genre seed queries rank results within genre pools for relevance

### 2. Intent Detection — 6 Query Types
Routing based on the Competency Questions designed in Assignment 4:

| Intent | Example |
|--------|---------|
| Genre recommendation (CQ1) | "Show me sci-fi books" |
| Similar books (CQ2) | "Books similar to Dune" |
| Author search (CQ3) | "Books by Stephen King" |
| Beginner-friendly (CQ4) | "Easy reads for beginners" |
| Popular / highly rated (CQ5) | "What are the best rated books?" |
| Preference-based (CQ6) | "I like history, what should I read?" |

### 3. User Memory & Personalisation
- SpaCy NER extracts genres, authors, and reading level from natural language
- Per-mention sentiment window handles mixed sentences ("I like sci-fi but not romance")
- Session memory persists preferences across turns (no database or API key required)
- Disliked genres are hard-excluded; liked genres receive a ×2 score boost

### 4. RAG with Graceful Fallback
- With `GROQ_API_KEY`: retrieved books are passed to Llama 3.3 70B for natural language response
- Without key: system returns a clean formatted list — fully functional either way

---

## Unit Testing

The test suite (`test_fastapi.py`) covers all three layers of the system with **70 tests**.

### Test structure

| Section | Scope | Tests |
|---------|-------|-------|
| `TestExtractGenreMentions` | `spacy_ner` — genre alias mapping, sentiment window | 8 |
| `TestExtractReadingLevel` | `spacy_ner` — reading level keyword detection | 5 |
| `TestParseUserInput` | `spacy_ner` — full preference parsing | 4 |
| `TestMergeWithSessionMemory` | `spacy_ner` — cross-turn memory accumulation and conflict resolution | 4 |
| `TestDetectIntent` | `fastapi_app` — CQ1–CQ6 routing + priority ordering | 11 |
| `TestApplyPreferenceFiltering` | `fastapi_app` — genre exclusion and score boosting | 4 |
| `TestRecommendByAuthor` | `fastapi_app` — fuzzy author lookup | 4 |
| `TestRecommendPopular` | `fastapi_app` — popularity ranking and genre filter | 4 |
| `TestRecommendBeginner` | `fastapi_app` — description-length proxy for difficulty | 3 |
| `TestAPIEndpoints` | FastAPI integration — all endpoints, session memory, response schema | 23 |

One test (`test_chat_cq3_short`) documents the single remaining limitation: the bare `[Author Name] books?` pattern (e.g., "Frank Herbert books?") triggers no signal keyword and falls back to `general_search`. This is the one unresolved edge case from the 30-query benchmark (29/30 = 96.7%).

### How to run

```bash
# Activate the virtual environment, then:
source .venv/bin/activate
pip install pytest httpx          # one-time, if not already installed
pytest test_fastapi.py -v
```

Expected output: **70 passed** in ~20 s.
