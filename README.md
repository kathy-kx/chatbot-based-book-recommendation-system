# Chatbot-Based Book Recommendation System

## Project Overview

A Conversational Recommender System (CRS) that provides personalized book recommendations through natural language interaction. The system uses **TF-IDF cosine similarity** for content-based recommendations and maintains **user memory** across conversation turns.

**Key Technologies:**
- **FastAPI** — REST API serving recommendation engine
- **SpaCy** — Named Entity Recognition for preference extraction
- **N8N** — Chatbot orchestration with Docker (self-hosted)
- **Scikit-learn** — TF-IDF vectorization and cosine similarity
- **data** — Books with clusters dataset
- **classification** — Genre classification models

---

## Architecture

```
User → N8N Chatbot → FastAPI /chat endpoint → Recommendation Engine
                ↑                                    ↓
                └── Session Memory ←───────────────┘
```

**Data Flow:**
1. User sends message (e.g., "I like sci-fi books")
2. N8N forwards to FastAPI `/chat`
3. FastAPI parses preferences via SpaCy NER
4. FastAPI queries cosine similarity engine
5. Preferences stored in session memory
6. Results returned through N8N to user

---

## Project Structure

```
├── books_with_clusters.csv          # book dataset with genre/cluster labels
├── tfidf_matrix.npz                 # TF-IDF sparse matrix
├── tfidf_vectorizer.pkl             # fitted TF-IDF vectorizer
├── Books.csv                        # Goodreads ratings (for popularity ranking)
│
├── B_cosine_similarity.py           # classification + cosine similarity
│
├── fastapi_app.py                   #  FastAPI server with session memory
├── spacy_ner.py                     # SpaCy NER preference extraction
│
├── requirements.txt                 # Python dependencies
├── User_Guide.md                   # Detailed user manual 
└── workflow.json                   # N8N exported workflow
```

---

## Key Features

### 1. Content-Based Recommendation
- TF-IDF vectors represent book descriptions
- Cosine similarity measures semantic closeness
- Cluster-based boosting improves relevance

### 2. Intent Detection
Six competency question types:
- CQ1: Genre-based recommendation
- CQ2: Similar books ("similar to Dune")
- CQ3: Author search ("books by King")
- CQ4: Beginner-friendly books
- CQ5: Popular/highly-rated books
- CQ6: Personalized preference-based

### 3. User Memory & Preference Modelling
- Extracts preferences from natural language
- Session memory persists across turns
- Genre alias mapping (sci-fi → scientific)
- Preference conflict resolution

### 4. N8N Integration
- Self-hosted via Docker
- HTTP Request tool calls FastAPI
- Session memory node for persistence

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Start FastAPI
uvicorn fastapi_app:app --reload --port 8000 --host 0.0.0.0

# 3. Start N8N Docker
docker run -d --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n n8nio/n8n

# 4. Open http://localhost:5678, import workflow.json
```

For detailed instructions, see **[User_Guide.md](./User_Guide.md)**.

---

## Innovation: Dynamic User Persona

Unlike static recommenders, this system maintains a **dynamic user persona** across turns:

```json
{
  "preferred_genres": ["scientific", "fiction"],
  "disliked_genres": ["romantic"],
  "reading_level": "beginner",
  "past_feedback": [
    {"book": "Dune", "liked": true},
    {"book": "Foundation", "liked": false}
  ]
}
```

Preferences accumulate and refine throughout the conversation, enabling truly personalized recommendations.

---

## Limitations & Future Work

- **Collaborative filtering**: Currently content-based only (no SVD/KNN/NMF due to model serialization constraints)
- **Genre granularity**: Database has broad categories; user "mystery" maps to "fiction"
- **Cold start**: New users without expressed preferences fall back to general search
- **Database**: SQLite/PostgreSQL not implemented (flat CSV used)
