# User Guide — Book Recommendation Chatbot

## Overview

A conversational book recommendation system combining:
- **FastAPI** — recommendation engine REST API
- **SpaCy** — Named Entity Recognition (NER) for preference extraction
- **N8N** — chatbot orchestration with session memory
- **Cosine Similarity** — content-based book recommendation (TF-IDF features)
- **Groq API** *(optional)* — LLM-generated natural language responses

**How recommendations work:** Each book's description is represented as a TF-IDF vector. When a user requests a recommendation, the system computes cosine similarity between the query vector and all books in the database, returning results in descending order of similarity. Preference filtering then hard-excludes disliked genres and doubles the score of preferred genres.

Based on the competency questions designed in Assignment 4, the chatbot supports: genre-based recommendations, finding similar books, author search, beginner-friendly suggestions, popular titles, and personalized preference-based results. Preferences are remembered across conversation turns.

---

## Running the System

There are **two ways** to use this system:

---

### Option A — Cloud Version (no local setup required)

A live deployment is available without installing anything:

- **Chat interface:** hosted on N8N Cloud — open the chat link directly in your browser. [Chat link]((https://kathy-kx.app.n8n.cloud/workflow/ps1bfjRpaJ3OWZUg/621d3b?projectId=ivnuR5X22yOOs7ij&uiContext=workflow_list))
- **Backend API:** deployed on [Render](https://chatbot-based-book-recommendation-system.onrender.com/chat) 

> **Cold start notice:** The Render free tier pauses the server after a period of inactivity. **The first message after idle may take 30–60 seconds** to get a response while the server wakes up. This is normal — subsequent messages will be fast.

---

### Option B — Local Version

Run the full system on your own machine.

#### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### Step 2: (Optional but Recommended) Configure Groq API Key

To enable LLM-generated responses, create a `.env` file in the project root:

```bash
cp .env.example .env
# Then edit .env and add your key:
# GROQ_API_KEY=your_groq_api_key_here
```

If no key is set, the system returns a clean formatted recommendation list — all features work normally.

#### Step 3: Start the FastAPI Server

```bash
uvicorn fastapi_app:app --reload --port 8000 --host 0.0.0.0
```

- API available at: `http://localhost:8000`
- Interactive docs (Swagger UI): `http://localhost:8000/docs`

#### Step 4: Start N8N

##### Primary method — Docker

```bash
docker run -d \
  --name n8n \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  n8nio/n8n
```

Open: `http://localhost:5678`


#### Step 5: Import Workflow and Start Chatting

1. Open N8N → click **"..."** (top right) → **Import from file** → upload `workflow.json`
2. Activate the workflow
3. Open the **Chat** panel and start a conversation

---

## Required Files (local setup)

All files must be present in the project root:

```
books_with_clusters.csv      # 1,985 books with genre labels and cluster info
tfidf_matrix.npz             # TF-IDF sparse matrix (1985 × 5000)
tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
Books.csv                    # Goodreads ratings data (for popularity ranking)
fastapi_app.py               # FastAPI server
spacy_ner.py                 # SpaCy NER preference extraction
cosine_similarity.py         # Cosine similarity recommender
workflow.json                # N8N workflow
```

---

## Jupyter Notebooks (Google Colab)

The `.ipynb` notebooks in this project can be run in [Google Colab](https://colab.research.google.com/) without a local Python environment:

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. **File → Upload notebook** → select the `.ipynb` file
3. Upload the required data files (`books_with_clusters.csv`, `tfidf_matrix.npz`, etc.) to the Colab session storage
4. Run all cells

| Notebook | Purpose |
|----------|---------|
| `Classification_models.ipynb` | Genre classification: BoW / TF-IDF / LDA / Word2Vec features with LR / RF / SVM models |

---

## Python Dependencies

```
fastapi
uvicorn[standard]
pydantic
spacy
scikit-learn
scipy
pandas
numpy
joblib
matplotlib
groq              # optional — for LLM response generation
python-dotenv     # optional — for loading .env file locally
```

Install all at once:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Verify:

```bash
python -c "import fastapi, spacy, sklearn, scipy, pandas, numpy, joblib; print('OK')"
```

---


## Example Conversations

Open the N8N Chat panel (or cloud chat interface) and try these scenarios. Use the **same chat thread** to test memory accumulation.

### Scenario 1: Dislike filtering + similar books
```
You:  I don't like military
Bot:  [Memory: dislikes: military] Got it...

You:  Show me books similar to Harry Potter
Bot:  Returns similar books — NO military books in results
```

### Scenario 2: Like boost + general search
```
You:  I love history books
Bot:  [Memory: likes: history] ...

You:  The Diary of a Young Girl
Bot:  Returns relevant books — history books boosted (score ×2)
```

### Scenario 3: Memory accumulates across turns
```
You:  I like sci-fi but don't like romance
Bot:  [Memory: likes: scientific; dislikes: romantic]

You:  Books similar to Dune
Bot:  Similar to Dune, no romantic books

You:  I also don't like military
Bot:  [Memory: likes: scientific; dislikes: romantic, military]

You:  Books similar to Foundation
Bot:  Similar to Foundation, no romantic or military books
```

### Scenario 4: Preference reversal
```
You:  I like fiction
You:  Actually I don't like fiction
You:  Books similar to Harry Potter
Bot:  Returns similar books — NO fiction (moved from liked to disliked)
```

### Scenario 5: Reset clears memory
```
You:  I hate romance
You:  Reset
You:  Books similar to Harry Potter
Bot:  Returns results normally — no [Memory] shown, romance no longer excluded
```

---

## API Endpoints List

### GET /
Health check. Returns available genres.

```json
{ "message": "Book Recommendation API", "genres": ["fiction", "history", ...] }
```

---

### GET /genres
Returns all 11 genre labels in the database.

```json
{ "genres": ["biography", "business", "fiction", "history", "medical",
             "military", "other", "psychology", "romantic", "scientific", "travel"] }
```

---

### POST /chat
Primary chatbot endpoint used by N8N. Parses the message, updates session memory, detects intent, and returns recommendations.

**Request:**
```json
{
  "message": "I like sci-fi, show me books similar to Dune",
  "session_id": "user_001",
  "top_n": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | — | User's natural language input |
| `session_id` | string | No | `"default"` | Identifies the session for cross-turn memory |
| `top_n` | integer | No | `5` | Number of recommendations to return |

**Response:**
```json
{
  "output": "[Memory: likes: scientific]\nBooks similar to 'Dune':\n\n1. Foundation — Isaac Asimov [scientific]",
  "recommendations": [
    { "Title": "Foundation", "Authors": "Isaac Asimov", "keyword_category": "scientific", "similarity_score": 0.4123 }
  ],
  "detected_preferences": { "preferred_genres": ["scientific"], "disliked_genres": [], ... },
  "session_memory": { "preferred_genres": ["scientific"], "disliked_genres": [], ... },
  "intent": "recommend_similar"
}
```

---

### POST /recommend
Direct recommendation endpoint with explicit preference parameters (also used internally by `/chat`).

**Request:**
```json
{
  "query": "Dune",
  "preferred_genres": ["scientific"],
  "disliked_genres": ["romantic"],
  "reading_level": null,
  "top_n": 5
}
```

---

### GET /session/{session_id}
Returns the accumulated preferences for a session.

```bash
curl http://localhost:8000/session/user_001
```

---

### DELETE /session/{session_id}
Clears all preferences for a session.

```bash
curl -X DELETE http://localhost:8000/session/user_001
```

---

## Supported Queries

| What you want | Example input |
|---------------|---------------|
| Genre recommendation | "Show me fiction books" / "I like sci-fi" |
| Similar books | "Books similar to Dune" / "Something like Harry Potter" |
| Author search | "Books by Stephen King" / "Written by Tolkien" |
| Beginner-friendly | "Easy reads for beginners" / "Simple light read" |
| Popular / highly rated | "What are the best rated books?" / "Top books" |
| Preference-based | "I like history, what should I read?" |
| Reset session | "Reset" / "Start over" / "Forget everything" |

---

## Session Memory & Preference Detection

### Supported Genres (database values)
`fiction` · `history` · `military` · `travel` · `romantic` · `medical` · `business` · `scientific` · `psychology` · `biography` · `other`

### User Input Aliases (automatically mapped)

| User says | Mapped to |
|-----------|-----------|
| sci-fi, scifi, science fiction | scientific |
| romance | romantic |
| mystery, thriller, horror, fantasy | fiction |
| self-help | psychology |
| memoir, autobiography | biography |
| war | military |
| non-fiction, nonfiction | other |

### Preference Detection Examples

| Input | Detected |
|-------|----------|
| "I like sci-fi" | preferred_genres: ["scientific"] |
| "I don't like romance" | disliked_genres: ["romantic"] |
| "I like sci-fi but not romance" | preferred: ["scientific"], disliked: ["romantic"] |
| "I prefer beginner books" | reading_level: "beginner" |
| "Books by Asimov" | liked_authors: ["Asimov"] |

### Session Memory Flow

1. Turn 1: "I like sci-fi" → stored: `preferred_genres: ["scientific"]`
2. Turn 2: "I don't like romance" → stored: `disliked_genres: ["romantic"]`
3. Turn 3: "Books similar to Dune" → results filtered: scientific ×2 boost, romantic excluded
4. Preferences persist for the entire conversation
5. "Reset" / "Start over" → session cleared

---

## Project Structure

```
chatbot-based-book-recommendation-system/
├── fastapi_app.py               # FastAPI server — intent routing, session memory, all handlers
├── cosine_similarity.py         # Cosine similarity recommender (imported by fastapi_app.py)
├── spacy_ner.py                 # SpaCy NER — preference extraction and session memory merge
│
├── books_with_clusters.csv      # 1,985 books with genre labels
├── tfidf_matrix.npz             # TF-IDF sparse matrix
├── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
├── Books.csv                    # Goodreads ratings data
│
├── Classification.ipynb         # Genre classification experiments (run in Colab)
├── workflow.json                # N8N exported workflow
│
├── requirements.txt             # Python dependencies
├── .env.example                 # Groq API key template
└── User_Guide.md                # This file
```

---

## N8N Workflow Export / Import

### Export
1. Open workflow in N8N
2. Click **"..."** (top right) → **Export** → **Download JSON**

### Import
1. Open N8N on the target machine
2. Click **Import from JSON** → upload the exported file

### Verify JSON
```bash
python -m json.tool workflow.json > /dev/null && echo "Valid JSON"
```
