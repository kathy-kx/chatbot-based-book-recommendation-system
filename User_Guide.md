# User Guide - Book Recommendation Chatbot

## Overview

This is a conversational book recommendation system combining:
- **FastAPI** — recommendation engine API
- **SpaCy** — Named Entity Recognition (NER) for preference extraction
- **N8N** — chatbot orchestration with session memory (self-hosted via Docker)
- **Cosine Similarity** — content-based book recommendation (TF-IDF features)

**How recommendations work:** Each book's description is represented as a TF-IDF vector. When a user requests a recommendation, the system computes the cosine similarity between the query book's vector and all other books in the database, measuring how close their descriptions are in meaning. Results are returned in descending order of similarity score, so the most description-similar books appear first.

Based on the competency questions designed in Assignment 4, the chatbot also supports recommend books by genre, find books by a specific author, suggest beginner-friendly reads, surface popular titles, find books similar to one you enjoyed, and personalize results based on your stated preferences. It also remembers your preferences across conversation turns and filters recommendations accordingly.

---

## Installation

### 1. Python Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`requirements.txt` includes:

```
fastapi
uvicorn
pydantic
scikit-learn
scipy
pandas
numpy
spacy
joblib
matplotlib
```

### 2. Verify Installation

```bash
python -c "import fastapi, spacy, sklearn, scipy, pandas, numpy, joblib, matplotlib; print('All packages installed')"
```

### 3. Required Files in Project Root

```
books_with_clusters.csv      # book dataset with genre labels and cluster info
tfidf_matrix.npz             # TF-IDF feature matrix 
tfidf_vectorizer.pkl         # fitted TF-IDF vectorizer 
Books.csv                    # Goodreads ratings data (for popularity ranking)
fastapi_app.py               # FastAPI server
spacy_ner.py                 # SpaCy NER preference extraction
B_cosine_similarity.py       # cosine similarity recommender
```

---

## Running the System

### Step 1: Start FastAPI Server

```bash
uvicorn fastapi_app:app --reload --port 8000 --host 0.0.0.0
```

The API will be available at: `http://localhost:8000`

Interactive API documentation (Swagger UI): `http://localhost:8000/docs`

### Step 2: Start N8N

#### Using Docker (recommended)

```bash
docker run -d \
  --name n8n \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  n8nio/n8n
```

Open browser: `http://localhost:5678`

> Inside the Docker container, use `host.docker.internal:8000` to reach FastAPI on the host machine (supported on Mac and Windows Docker Desktop).

### Step 3: Import N8N Workflow

1. Open N8N → click **"..."** (top right) → **Import from file** → upload `workflow.json`
2. Open the **Chat** panel to start a conversation

### Step 4: Get Recommendations

1. Send a message to the chat. For example, "Show me books similar to Harry Potter"
2. You will get the response.

---

## API Endpoints

### GET /
Returns a welcome message and list of available genres.

**Response:**
```json
{
  "message": "Book Recommendation API",
  "genres": ["fiction", "history", "military", ...]
}
```

---

### GET /genres
Returns all book genres available in the database.

**Response:**
```json
{
  "genres": ["biography", "business", "fiction", "history", "medical",
             "military", "other", "psychology", "romantic", "scientific", "travel"]
}
```

---

### POST /chat
Main chatbot endpoint. Parses the user message, updates session memory, detects intent, and returns book recommendations.

**Request Body:**
```json
{
  "message": "I like sci-fi, show me books similar to Dune",
  "session_id": "user_001",
  "top_n": 5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | User's natural language input |
| `session_id` | string | No | Session identifier for memory persistence (default: `"default"`) |
| `top_n` | integer | No | Number of recommendations to return (default: `5`) |

**Response:**
```json
{
  "output": "[Memory: likes: scientific]\nBooks similar to 'Dune':\n\n1. ...",
  "recommendations": [
    {
      "Title": "Foundation",
      "Authors": "Isaac Asimov",
      "keyword_category": "scientific",
      "similarity_score": 0.4123
    }
  ],
  "detected_preferences": {
    "preferred_genres": ["scientific"],
    "disliked_genres": [],
    "liked_authors": [],
    "disliked_authors": [],
    "reading_level": null,
    "past_feedback": []
  },
  "session_memory": {
    "preferred_genres": ["scientific"],
    "disliked_genres": [],
    "liked_authors": [],
    "disliked_authors": [],
    "reading_level": null,
    "past_feedback": []
  },
  "intent": "recommend_similar"
}
```

---

### POST /recommend
Direct recommendation endpoint (also used internally by `/chat`). Accepts explicit preference parameters.

**Request Body:**
```json
{
  "query": "Dune",
  "preferred_genres": ["scientific"],
  "disliked_genres": ["romantic"],
  "reading_level": null,
  "top_n": 5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Book title or description keyword |
| `preferred_genres` | array | No | Genres to boost in results |
| `disliked_genres` | array | No | Genres to exclude from results |
| `reading_level` | string | No | `"beginner"`, `"intermediate"`, or `"advanced"` |
| `top_n` | integer | No | Number of results (default: `5`) |

**Response:**
```json
{
  "recommendations": [
    {
      "Title": "Foundation",
      "Authors": "Isaac Asimov",
      "keyword_category": "scientific",
      "similarity_score": 0.4123
    }
  ],
  "user_profile_snapshot": {
    "preferred_genres": ["scientific"],
    "disliked_genres": ["romantic"],
    "reading_level": null,
    "last_query": "Dune",
    "last_recommendations": ["Foundation", "..."]
  }
}
```

---

### GET /session/{session_id}
Returns the current accumulated preferences stored for a session.

```bash
curl http://localhost:8000/session/user_001
```

---

### DELETE /session/{session_id}
Clears all stored preferences for a session.

```bash
curl -X DELETE http://localhost:8000/session/user_001
```

---

## Supported Queries

Based on the competency questions designed in Assignment 4, the chatbot handles the following types of requests:

| What you want | Example Input |
|---------------|---------------|
| Recommend books in a genre | "Show me fiction books" |
| Find books similar to one you liked | "Books similar to Dune" |
| Find books by a specific author | "Books by Stephen King" |
| Get beginner-friendly reads | "Easy books for beginners" |
| Discover popular / highly rated books | "What are the best rated books?" |
| Personalized recommendations based on your preferences | "I like sci-fi, what should I read?" |

---

## Session Memory & Preference Detection

### Supported Genres (database values)
`fiction`, `history`, `military`, `travel`, `romantic`, `medical`, `business`, `scientific`, `psychology`, `biography`, `other`

### User Input Aliases (automatically mapped)

| User says | Mapped to |
|-----------|-----------|
| sci-fi, scifi, science fiction | scientific |
| romance | romantic |
| mystery, thriller, horror, fantasy | fiction |
| self-help | psychology |
| memoir, autobiography | biography |
| war | military |
| non-fiction | other |

### Preference Detection Examples

| Input | Detected |
|-------|----------|
| "I like sci-fi" | preferred_genres: ["scientific"] |
| "I don't like romance" | disliked_genres: ["romantic"] |
| "I dislike military books" | disliked_genres: ["military"] |
| "I prefer beginner books" | reading_level: "beginner" |
| "Books by Asimov" | liked_authors: ["Asimov"] |

### Session Memory Flow

1. User says "I like sci-fi" → stored in session
2. User says "I don't like romance" → stored in session
3. User asks "Show me books similar to Dune" → results filtered: scientific boosted, romantic excluded
4. Preferences persist for the entire conversation
5. User says "reset" / "start over" → session cleared

---

## Testing: Example Conversations

Open N8N Chat panel and try these conversations (use the same chat thread to test memory):

### Scenario 1: Dislike filtering + similar books
```
You: I don't like military
Bot: [Memory: dislikes: military] ...

You: Show me books similar to Harry Potter
Bot: Returns similar books, NO military books in results
```

### Scenario 2: Like boost + general search
```
You: I love history books
Bot: [Memory: likes: history] ...

You: The Diary of a Young Girl
Bot: Returns relevant books, history books boosted (score ×2)
```

### Scenario 3: Memory accumulates
```
You: I like sci-fi but don't like romance
Bot: [Memory: likes: scientific; dislikes: romantic]

You: books similar to Dune
Bot: Similar to Dune, no romantic books

You: I also don't like military books
Bot: [Memory: likes: scientific; dislikes: romantic, military]

You: books similar to Foundation
Bot: Similar to Foundation, no romantic or military books
```

### Scenario 4: Preference reversal
```
You: I like fiction
You: Actually I don't like fiction
You: books similar to Harry Potter
Bot: Returns similar books, NO fiction books (fiction was moved to dislikes)
```

### Scenario 5: Reset clears memory
```
You: I hate romance
You: reset
You: books similar to Harry Potter
Bot: Returns similar books normally, no [Memory] shown (preferences cleared)
```

---

## N8N Workflow Export / Import

### Export
1. Open the workflow in N8N
2. Click **"..."** (top right) → **Export** → **Download JSON**

### Import
1. Open N8N on the target machine
2. Click **Import from JSON** → upload the exported file

### Verify JSON
```bash
python -m json.tool workflow.json > /dev/null && echo "Valid JSON"
```

---

## Project Structure

```
chatbot-based-book-recommendation-system/
├── books_with_clusters.csv          # Book dataset 
├── tfidf_matrix.npz                 # TF-IDF feature matrix 
├── tfidf_vectorizer.pkl             # Fitted TF-IDF vectorizer 
├── Books.csv                        # Goodreads ratings data
├── B_cosine_similarity.py           # classification + cosine recommender
├── fastapi_app.py                   # FastAPI API server
├── spacy_ner.py                     # SpaCy NER preference extraction
├── requirements.txt                 # Python dependencies
├── User_Guide.md                    # This file
└── workflow.json                    # N8N exported workflow
```
