"""
FastAPI Recommendation Engine for Book Chatbot
Chatbot & System Integration

Features:
- Cosine Similarity recommendation
- User preference parameters (preferred_genres, disliked_genres, reading_level)
- Based on TF-IDF features and recommendation logic implemented in B_cosine_similarity.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
warnings.filterwarnings("ignore")

# Load .env for local development (no-op if file doesn't exist or dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import cosine similarity recommendation function
from B_cosine_similarity import recommend_books as _b_recommend_books

# ============ Groq LLM (optional — graceful fallback if key not set) ============
_groq_client = None
_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if _GROQ_API_KEY:
    try:
        from groq import Groq
        _groq_client = Groq(api_key=_GROQ_API_KEY)
        print("Groq LLM enabled")
    except ImportError:
        print("groq package not installed — LLM disabled, using formatted responses")

# ============ Load Data ============
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Book data + TF-IDF features (in project root)
books_df = pd.read_csv(os.path.join(DATA_DIR, "books_with_clusters.csv"))
tfidf_matrix = load_npz(os.path.join(DATA_DIR, "tfidf_matrix.npz"))
tfidf_vectorizer = joblib.load(os.path.join(DATA_DIR, "tfidf_vectorizer.pkl"))

# Merge Books.csv rating data (for CQ5 popular books)
books_raw = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"))
books_raw["_title_lower"] = books_raw["title"].str.lower().str.strip()
books_df["_title_lower"] = books_df["Title"].str.lower().str.strip()
# Deduplicate Books.csv (multiple rows per title), avoid extra rows from left merge
books_raw_dedup = books_raw.drop_duplicates(subset=["_title_lower"])
books_df = books_df.merge(
    books_raw_dedup[["_title_lower", "average_rating", "ratings_count"]],
    on="_title_lower", how="left"
).drop(columns=["_title_lower"])
books_df["average_rating"] = books_df["average_rating"].fillna(books_df["average_rating"].median())
books_df["ratings_count"] = books_df["ratings_count"].fillna(0)

# Title -> index mapping
indices = pd.Series(books_df.index, index=books_df["Title"]).drop_duplicates()

# All supported genres
ALL_GENRES = books_df["keyword_category"].unique().tolist()

# Description length median (for CQ4 beginner approximation)
DESC_LEN_MEDIAN = books_df["description_clean"].str.split().str.len().median()

print(f"Loaded {len(books_df)} books, {len(ALL_GENRES)} genres, ratings merged")

# ============ Pydantic Models ============
class RecommendRequest(BaseModel):
    query: str  # User input: can be book title or description
    preferred_genres: Optional[List[str]] = None  # Preferred genre list
    disliked_genres: Optional[List[str]] = None  # Disliked genre list
    reading_level: Optional[str] = None  # Preferred reading level
    top_n: int = 5

class BookRecommend(BaseModel):
    Title: str
    Authors: str
    keyword_category: str
    similarity_score: float

class RecommendResponse(BaseModel):
    recommendations: List[BookRecommend]
    user_profile_snapshot: dict  # Current user preference snapshot (for N8N storage)

# ============ Recommendation Logic ============
def get_book_index(title: str) -> Optional[int]:
    """Get index by book title"""
    if title in indices:
        return indices[title]
    # Fuzzy match: check if title contains keyword
    matches = [i for i, t in enumerate(books_df["Title"]) if title.lower() in t.lower()]
    return matches[0] if matches else None

def recommend_books_cosine(title: str, top_n: int = 5) -> List[tuple]:
    """
    Cosine Similarity recommendation based on book title.
    Delegates to B_cosine_similarity.recommend_books(), returns: List[(book_row, similarity_score)]
    """
    result = _b_recommend_books(title, top_n)
    if isinstance(result, str):  # B returns string if title not found
        return []
    return [(row, round(float(row["similarity_score"]), 4)) for _, row in result.iterrows()]

def recommend_by_description(description: str, top_n: int = 5) -> List[tuple]:
    """
    Description-based recommendation (used when user says "I like books about xxx").
    """
    desc_vector = tfidf_vectorizer.transform([description])
    sim_scores = cosine_similarity(desc_vector, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        if sim_scores[idx] > 0:
            book = books_df.iloc[idx]
            results.append((book, sim_scores[idx]))
    return results

def apply_preference_filtering(results: List[tuple],
                                preferred_genres: List[str] = None,
                                disliked_genres: List[str] = None,
                                reading_level: str = None) -> List[tuple]:
    """
    Adjust recommendations based on user preferences:
    - Preferred genres: boost weight
    - Disliked genres: reduce weight
    """
    if not preferred_genres and not disliked_genres and not reading_level:
        return results

    adjusted_results = []
    for book, score in results:
        genre = book.get('keyword_category', '')

        # Hard exclude disliked genres
        if disliked_genres and genre in disliked_genres:
            continue

        adjusted_score = score
        if preferred_genres and genre in preferred_genres:
            adjusted_score *= 2.0  # Double weight for preferred genres

        adjusted_results.append((book, adjusted_score))

    adjusted_results.sort(key=lambda x: x[1], reverse=True)
    return adjusted_results

# ============ CQ3: find_by_author ============
def recommend_by_author(author_name: str, top_n: int = 5) -> List[tuple]:
    """Find books by author name (fuzzy match) (CQ3)"""
    mask = books_df["Authors"].str.lower().str.contains(author_name.lower(), na=False)
    matched = books_df[mask]
    if matched.empty:
        return []
    results = []
    for _, book in matched.head(top_n).iterrows():
        results.append((book, 1.0))
    return results

# ============ CQ4: recommend_beginner ============
def recommend_beginner(genres: List[str] = None, top_n: int = 5) -> List[tuple]:
    """
    Recommend beginner-friendly books (CQ4).
    Dataset has no difficulty label, using description word count below median as proxy for "simple".
    """
    pool = books_df.copy()
    if genres:
        pool = pool[pool["keyword_category"].isin(genres)]
    pool = pool[pool["description_clean"].str.split().str.len() <= DESC_LEN_MEDIAN]
    if pool.empty:
        pool = books_df if not genres else books_df[books_df["keyword_category"].isin(genres)]
    sample = pool.sample(min(top_n, len(pool)))
    return [(row, 1.0) for _, row in sample.iterrows()]

# ============ CQ5: recommend_popular ============
def recommend_popular(genres: List[str] = None, top_n: int = 5) -> List[tuple]:
    """
    Recommend highly-rated books (CQ5).
    Uses average_rating * log1p(ratings_count) as weighted popularity score.
    """
    pool = books_df.copy()
    if genres:
        pool = pool[pool["keyword_category"].isin(genres)]
    if pool.empty:
        pool = books_df.copy()
    pool = pool.copy()
    pool["popularity_score"] = pool["average_rating"] * np.log1p(pool["ratings_count"])
    pool = pool.sort_values("popularity_score", ascending=False)
    results = []
    for _, book in pool.head(top_n).iterrows():
        results.append((book, round(book["popularity_score"], 4)))
    return results

# ============ Intent Detection ============
# Based on the 6 Competency Questions from Assignment 4
# TF-IDF seed queries for each genre (for relevance ranking within genre pools)
GENRE_SEED_QUERIES = {
    "scientific": "science fiction space future technology alien robot",
    "fiction":    "story novel adventure mystery thriller suspense",
    "history":    "history war historical world century event",
    "military":   "war battle soldier army military combat",
    "travel":     "journey travel adventure explore world",
    "romantic":   "love romance relationship heart feeling",
    "medical":    "doctor hospital medicine health disease",
    "business":   "business money success leadership management",
    "psychology": "mind psychology human behavior emotion self",
    "biography":  "life story memoir autobiography person",
    "other":      "book story world people",
}

POPULAR_SIGNALS  = ["popular", "highly rated", "well-reviewed", "best", "top rated",
                    "top books", "well rated", "most read", "highest rated", "trending"]
SIMILAR_SIGNALS  = ["similar to", "books like", "like the book", "just like",
                    "something like", "what else is like", "else like", "like it",
                    "could you suggest books similar", "suggest.*similar"]
AUTHOR_SIGNALS   = ["by ", "written by", "books by", "author", "'s books", "s books"]
BEGINNER_SIGNALS = ["beginner", "easy", "simple", "light read", "introductory",
                    "entry level", "entry-level", "for beginners", "starting out"]

# Regex patterns to extract reference book titles from similar queries
import re as _re
_SIMILAR_TITLE_PATTERNS = [
    _re.compile(r"similar to\s+(.+?)(?:\?|!|\.|$)", _re.I),
    _re.compile(r"books?\s+like\s+(.+?)(?:\?|!|\.|$)", _re.I),
    _re.compile(r"like the book\s+(.+?)(?:\?|!|\.|$)", _re.I),
    _re.compile(r"just like\s+(.+?)(?:\?|!|\.|$)", _re.I),
    _re.compile(r"something like\s+(.+?)(?:\?|!|\.|$)", _re.I),
    _re.compile(r"(?:loved|enjoyed|finished|read)\s+(.+?),", _re.I),  # "I loved Dune,"
    _re.compile(r"(?:loved|enjoyed|finished|read)\s+(.+?)(?:\?|!|\.|$)", _re.I),
]

def extract_reference_title(message: str) -> Optional[str]:
    """Extract and validate reference book title from similar queries"""
    for pattern in _SIMILAR_TITLE_PATTERNS:
        m = pattern.search(message)
        if m:
            candidate = m.group(1).strip().rstrip("?!.,")
            # Exact match
            if get_book_index(candidate) is not None:
                return candidate
            # Partial match (for abbreviated titles like "Python Crash Course")
            lower = candidate.lower()
            partial = [t for t in books_df["Title"] if lower in t.lower()]
            if partial:
                return partial[0]
    return None

def detect_intent(message: str, prefs) -> str:
    """
    Intent routing based on the 6 CQs from Assignment 4.
    Returns: 'find_by_author' | 'recommend_popular' | 'recommend_beginner' |
           'recommend_similar' | 'recommend_by_genre' | 'general_search'
    """
    text = message.lower()

    # CQ2: Similar books - prioritized over author to avoid person names in titles triggering CQ3
    if any(s in text for s in SIMILAR_SIGNALS):
        return "recommend_similar"

    # CQ5: Popular books
    if any(s in text for s in POPULAR_SIGNALS):
        return "recommend_popular"

    # CQ4: Beginner books
    if prefs.reading_level == "beginner" or any(s in text for s in BEGINNER_SIGNALS):
        return "recommend_beginner"

    # CQ3: Author query
    if prefs.liked_authors or any(s in text for s in AUTHOR_SIGNALS):
        return "find_by_author"

    # CQ1/CQ6: Genre / user preferences
    if prefs.preferred_genres or prefs.disliked_genres:
        return "recommend_by_genre"

    return "general_search"

# ============ FastAPI App ============
app = FastAPI(
    title="Book Recommendation API",
    description="For N8N Chatbot RAG - Cosine Similarity based Recommender",
    version="1.0"
)

# Allow browser requests from GitHub Pages and any other origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Book Recommendation API", "genres": ALL_GENRES}

@app.get("/genres")
def get_genres():
    """Return all available genre list"""
    return {"genres": sorted(ALL_GENRES)}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Main recommendation endpoint.
    Accepts user query and preferences, returns recommended books.
    """
    # 1. Try to match book title
    idx = get_book_index(request.query)

    if idx is not None:
        # If title found, use Cosine Similarity
        results = recommend_books_cosine(request.query, top_n=request.top_n * 2)
    else:
        # If not found, use description recommendation
        results = recommend_by_description(request.query, top_n=request.top_n * 2)

    if not results:
        raise HTTPException(status_code=404, detail="No similar books found")

    # 2. Apply preference filtering
    results = apply_preference_filtering(
        results,
        preferred_genres=request.preferred_genres,
        disliked_genres=request.disliked_genres,
        reading_level=request.reading_level
    )

    # 3. Take top_n
    results = results[:request.top_n]

    # 4. Build response
    recommendations = []
    for book, score in results:
        recommendations.append(BookRecommend(
            Title=book["Title"],
            Authors=book["Authors"],
            keyword_category=book["keyword_category"],
            similarity_score=float(round(score, 4))
        ))

    # 5. Return user preference snapshot (for N8N Session Memory storage)
    user_profile_snapshot = {
        "preferred_genres": request.preferred_genres or [],
        "disliked_genres": request.disliked_genres or [],
        "reading_level": request.reading_level,
        "last_query": request.query,
        "last_recommendations": [r.Title for r in recommendations]
    }

    return RecommendResponse(
        recommendations=recommendations,
        user_profile_snapshot=user_profile_snapshot
    )

class ChatRequest(BaseModel):
    message: str
    top_n: int = 5
    session_id: str = "default"

# ============ Session Memory (FastAPI in-memory, no API Key required) ============
# key: session_id, value: accumulated user preferences dict
_session_store: dict = {}

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Get current session's accumulated preferences"""
    return _session_store.get(session_id, {})

@app.delete("/session/{session_id}")
def reset_session(session_id: str):
    """Clear session preferences (for starting a new conversation)"""
    _session_store.pop(session_id, None)
    return {"message": f"Session '{session_id}' cleared."}

def _llm_generate(user_message: str, recs: list, memory_note: str, intent: str) -> Optional[str]:
    """
    Call Groq LLM to generate a natural response grounded in retrieved books.
    Returns None if LLM is unavailable — caller falls back to formatted string.
    """
    if _groq_client is None or not recs:
        return None
    try:
        book_lines = "\n".join(
            f"- {r['Title']} by {r['Authors']} "
            f"[{r['keyword_category']}] (similarity: {r['similarity_score']:.2f})"
            for r in recs
        )
        system_prompt = (
            "You are a helpful book recommendation assistant. "
            "Your responses are grounded strictly in the retrieved books provided. "
            "Do NOT mention or invent any books not in the list. "
            "Be friendly, concise, and explain briefly why each book fits the request."
        )
        user_prompt = (
            f"The user said: \"{user_message}\"\n\n"
            f"Retrieved books from database:\n{book_lines}\n\n"
            "Write a short, natural recommendation response (max 4 sentences intro, "
            "then list the books with a one-line reason each)."
        )
        response = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        return memory_note + text
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return None  # fallback to formatted string


@app.post("/chat")
def chat(request: ChatRequest):
    """
    N8N chat endpoint: automatically parses preferences from user message,
    merges with historical preferences, and returns recommendations.
    session_id enables cross-turn memory without API Key.
    """
    from spacy_ner import parse_user_input, merge_with_session_memory

    # 0. Detect reset intent
    _RESET_PHRASES = ["start over", "reset", "forget everything", "clear my preferences",
                      "forget what i said", "new conversation", "start fresh", "clear history"]
    if any(p in request.message.lower() for p in _RESET_PHRASES):
        _session_store.pop(request.session_id, None)
        return {
            "output": "Got it! I've cleared your preferences. Let's start fresh — tell me what you'd like to read.",
            "recommendations": [],
            "detected_preferences": {},
            "session_memory": {},
            "intent": "session_reset"
        }

    # 1. Parse current message preferences
    current_prefs = parse_user_input(request.message)

    # 2. Load historical preferences from session store and merge
    stored = _session_store.get(request.session_id, {})
    prefs = merge_with_session_memory(current_prefs, stored)

    # 3. Save merged preferences back to session store
    _session_store[request.session_id] = prefs.to_dict()

    intent = detect_intent(request.message, prefs)

    def _format(results: List[tuple], header: str) -> dict:
        # Build session memory summary (for display to user)
        mem = []
        if prefs.preferred_genres:
            mem.append(f"likes: {', '.join(prefs.preferred_genres)}")
        if prefs.disliked_genres:
            mem.append(f"dislikes: {', '.join(prefs.disliked_genres)}")
        if prefs.reading_level:
            mem.append(f"level: {prefs.reading_level}")
        memory_note = f"[Memory: {'; '.join(mem)}]\n" if mem else ""

        if not results:
            return {
                "output": memory_note + "Sorry, I couldn't find any books matching your request.",
                "recommendations": [],
                "detected_preferences": prefs.to_dict(),
                "session_memory": _session_store.get(request.session_id, {}),
                "intent": intent
            }
        recs = [
            BookRecommend(
                Title=str(book["Title"]),
                Authors=str(book["Authors"]),
                keyword_category=str(book["keyword_category"]),
                similarity_score=float(round(score, 4))
            )
            for book, score in results
        ]
        lines = [f"{i+1}. {r.Title} — {r.Authors} [{r.keyword_category}]"
                 for i, r in enumerate(recs)]
        fallback_output = memory_note + header + "\n\n" + "\n".join(lines)

        # Try LLM generation; fall back to formatted string if unavailable
        llm_output = _llm_generate(
            request.message, [r.model_dump() for r in recs], memory_note, intent
        )

        return {
            "output": llm_output if llm_output else fallback_output,
            "recommendations": [r.model_dump() for r in recs],
            "detected_preferences": prefs.to_dict(),
            "session_memory": _session_store.get(request.session_id, {}),
            "intent": intent
        }

    # ── CQ3: find_by_author ──
    if intent == "find_by_author":
        author = prefs.liked_authors[0] if prefs.liked_authors else None
        if not author:
            # Extract "by X" pattern from message
            import re
            m = re.search(r"(?:by|written by|books by)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)", request.message)
            author = m.group(1) if m else None
        if not author:
            return {"output": "Please specify an author name, e.g. 'books by Tolkien'.",
                    "recommendations": [], "detected_preferences": prefs.to_dict(), "intent": intent}
        results = recommend_by_author(author, request.top_n)
        return _format(results, f"Books by {author}:")

    # ── CQ5: recommend_popular ──
    if intent == "recommend_popular":
        results = recommend_popular(prefs.preferred_genres or None, request.top_n)
        genre_note = f" in {', '.join(prefs.preferred_genres)}" if prefs.preferred_genres else ""
        return _format(results, f"Top-rated books{genre_note} (sorted by rating × popularity):")

    # ── CQ4: recommend_beginner ──
    if intent == "recommend_beginner":
        results = recommend_beginner(prefs.preferred_genres or None, request.top_n)
        genre_note = f" in {', '.join(prefs.preferred_genres)}" if prefs.preferred_genres else ""
        return _format(results, f"Beginner-friendly books{genre_note} (shorter, accessible reads):")

    # ── CQ1/CQ6: recommend_by_genre / user preference ──
    if intent == "recommend_by_genre":
        # Only dislike, no like: record preference and guide user to express what they want
        if not prefs.preferred_genres and prefs.disliked_genres:
            dislike_str = ", ".join(prefs.disliked_genres)
            return {
                "output": (
                    f"Got it — I'll avoid {dislike_str} books for you.\n"
                    "What kind of books do you enjoy? For example:\n"
                    "• 'I like sci-fi'\n• 'I enjoy history and biography'\n• 'Show me fantasy books'"
                ),
                "recommendations": [],
                "detected_preferences": prefs.to_dict(),
                "session_memory": _session_store.get(request.session_id, {}),
                "intent": intent
            }

        genre_pool = books_df[books_df["keyword_category"].isin(prefs.preferred_genres)]
        if prefs.disliked_genres:
            genre_pool = genre_pool[~genre_pool["keyword_category"].isin(prefs.disliked_genres)]
        if genre_pool.empty:
            genre_pool = books_df

        # Use seed query for TF-IDF ranking within genre pool
        # Avoid random sampling of irrelevant books from mixed categories
        seed = " ".join(
            GENRE_SEED_QUERIES.get(g, g) for g in prefs.preferred_genres
        )
        pool_indices = genre_pool.index.tolist()
        seed_vec = tfidf_vectorizer.transform([seed])
        pool_matrix = tfidf_matrix[pool_indices]
        sims = cosine_similarity(seed_vec, pool_matrix).flatten()
        # Filter out books with very low similarity (threshold)
        RELEVANCE_THRESHOLD = 0.05
        valid_mask = sims >= RELEVANCE_THRESHOLD
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            valid_indices = sims.argsort()[::-1][:request.top_n]  # Fallback

        # Take top_n*3 candidates, randomly sample top_n to ensure relevance + diversity
        candidate_n = min(request.top_n * 3, len(valid_indices))
        top_candidates = valid_indices[sims[valid_indices].argsort()[::-1][:candidate_n]]
        chosen = np.random.choice(top_candidates, size=min(request.top_n, len(top_candidates)), replace=False)
        results = [
            (genre_pool.iloc[rank], round(float(sims[rank]), 4))
            for rank in chosen
        ]

        pref_note = f"genres you like: {', '.join(prefs.preferred_genres)}"
        if prefs.disliked_genres:
            pref_note += f"; excluding: {', '.join(prefs.disliked_genres)}"
        return _format(results, f"(Detected: {pref_note})\nRecommendations:")

    # ── CQ2: recommend_similar ── Extract title first, then cosine similarity
    if intent == "recommend_similar":
        ref_title = extract_reference_title(request.message)
        if ref_title is None:
            # Fallback: use entire message as query
            ref_title = request.message
        results = recommend_books_cosine(ref_title, request.top_n * 2)
        if not results:
            results = recommend_by_description(ref_title, request.top_n * 2)
        # Apply session preference filtering (disliked_genres hard exclude, preferred_genres boost)
        results = apply_preference_filtering(
            results,
            preferred_genres=prefs.preferred_genres or None,
            disliked_genres=prefs.disliked_genres or None,
        )
        results = results[:request.top_n]
        return _format(results, f"Books similar to '{ref_title}':")

    # ── User asks for "recommendations based on my preferences" but provides no specific preferences: prompt ──
    _PREF_PHRASES = ["based on my preferences", "my preferences", "based on my taste",
                     "my taste", "what i like", "recommend based on"]
    if intent == "general_search" and any(p in request.message.lower() for p in _PREF_PHRASES):
        return {
            "output": (
                "I'd love to recommend based on your preferences! "
                "Please tell me what you enjoy — for example:\n"
                "• 'I like sci-fi'\n• 'I enjoy fantasy and history'\n• 'I don't like romance'"
            ),
            "recommendations": [], "detected_preferences": prefs.to_dict(), "intent": "recommend_by_preference_prompt"
        }

    # ── general: TF-IDF cosine similarity ──
    rec_request = RecommendRequest(
        query=request.message,
        preferred_genres=prefs.preferred_genres or None,
        disliked_genres=prefs.disliked_genres or None,
        reading_level=prefs.reading_level,
        top_n=request.top_n
    )
    try:
        result = recommend(rec_request)
    except HTTPException:
        return {"output": "Sorry, I couldn't find any books matching your request.",
                "recommendations": [], "detected_preferences": prefs.to_dict(), "intent": intent}

    header_parts = []
    if prefs.preferred_genres:
        header_parts.append(f"genres you like: {', '.join(prefs.preferred_genres)}")
    if prefs.disliked_genres:
        header_parts.append(f"excluding: {', '.join(prefs.disliked_genres)}")
    header = f"(Detected: {'; '.join(header_parts)})\n" if header_parts else ""
    lines = [f"{i+1}. {r.Title} — {r.Authors} [{r.keyword_category}]"
             for i, r in enumerate(result.recommendations)]
    return {
        "output": header + "Here are your recommendations:\n\n" + "\n".join(lines),
        "recommendations": [r.model_dump() for r in result.recommendations],
        "detected_preferences": prefs.to_dict(),
        "intent": intent
    }

# ============ How to Run ============
# uvicorn fastapi_app:app --reload --port 8000
# Then in N8N, use HTTP Request Tool to call http://localhost:8000/chat
