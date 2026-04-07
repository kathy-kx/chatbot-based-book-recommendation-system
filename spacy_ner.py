"""
SpaCy NER - Named Entity Recognition for Book Recommendation
Chatbot & System Integration

Features:
- Extract entities: genre, author, topic, book title from user input
- Support preference expressions: positive ("I like sci-fi") and negative ("I don't like mystery")
"""

import spacy
from typing import Dict, List, Optional

# Load spaCy model (run first: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy model: python -m spacy download en_core_web_sm")
    nlp = None

# Known genres (matching keyword_category in books_with_clusters.csv)
KNOWN_GENRES = [
    "fiction", "history", "military", "travel",
    "romantic", "medical", "business", "scientific", "psychology",
    "biography", "other"
]

# User input aliases -> database genre mappings
GENRE_ALIASES = {
    # sci-fi variants
    "sci-fi": "scientific",
    "scifi": "scientific",
    "scify": "scientific",
    "science fiction": "scientific",
    "science-fiction": "scientific",
    # fiction sub-genres
    "mystery": "fiction",
    "mistery": "fiction",      # typo variant
    "thriller": "fiction",
    "horror": "fiction",
    "fantasy": "fiction",
    "fantasey": "fiction",     # typo variant
    "fantsy": "fiction",
    # other mappings
    "non-fiction": "other",
    "nonfiction": "other",
    "self-help": "psychology",
    "selfhelp": "psychology",
    "romance": "romantic",
    "war": "military",
    "historical": "history",
    "memoir": "biography",
    "autobiograph": "biography",
}

# Preference indicator words
POSITIVE_WORDS = [
    "like", "love", "enjoy", "prefer", "interested", "want", "looking for",
    "align with my taste", "based on what i enjoy", "based on my taste",
    "suggest", "show me", "recommend", "find me"
]
NEGATIVE_WORDS = ["don't like", "dont like", "do not like",
                  "don't enjoy", "dont enjoy",
                  "hate", "dislike", "not like", "avoid"]

class UserPreferences:
    """User preferences data class"""
    def __init__(self):
        self.preferred_genres: List[str] = []
        self.disliked_genres: List[str] = []
        self.liked_authors: List[str] = []
        self.disliked_authors: List[str] = []
        self.reading_level: Optional[str] = None  # beginner, intermediate, advanced
        self.past_feedback: List[Dict] = []  # [{"book": "Dune", "liked": True}]

    def to_dict(self) -> dict:
        return {
            "preferred_genres": self.preferred_genres,
            "disliked_genres": self.disliked_genres,
            "liked_authors": self.liked_authors,
            "disliked_authors": self.disliked_authors,
            "reading_level": self.reading_level,
            "past_feedback": self.past_feedback
        }

    def update(self, other: dict):
        """Update from N8N Session Memory"""
        for key in ["preferred_genres", "disliked_genres", "liked_authors", "disliked_authors"]:
            if key in other and isinstance(other[key], list):
                setattr(self, key, other[key])
        if "reading_level" in other:
            self.reading_level = other["reading_level"]
        if "past_feedback" in other and isinstance(other["past_feedback"], list):
            self.past_feedback = other["past_feedback"]

    def __repr__(self):
        return f"UserPreferences({self.to_dict()})"

def _sentiment_before(text_lower: str, pos: int) -> str:
    """
    Check if there are negative words in the 50 characters before the genre mention.
    Evaluates sentiment per genre mention to avoid whole-sentence misclassification.
    """
    window = text_lower[max(0, pos - 50):pos]
    if any(neg in window for neg in NEGATIVE_WORDS):
        return "negative"
    return "positive"

def extract_genre_mentions(text: str) -> Dict[str, List[str]]:
    """
    Extract genre mentions from text, evaluating sentiment per mention.
    Returns: {"positive": ["fiction"], "negative": ["romantic"]}
    """
    text_lower = text.lower()
    result = {"positive": [], "negative": []}

    def _add(genre: str, pos: int):
        key = _sentiment_before(text_lower, pos)
        if genre not in result[key]:
            result[key].append(genre)

    # Check aliases first (to avoid "science fiction" being matched by "fiction")
    for alias, mapped in GENRE_ALIASES.items():
        pos = text_lower.find(alias)
        if pos != -1:
            _add(mapped, pos)

    # Then check standard genres (skip positions already covered by aliases)
    for genre in KNOWN_GENRES:
        pos = text_lower.find(genre)
        if pos != -1:
            _add(genre, pos)

    return result

def extract_author_mentions(text: str, nlp_model=None) -> Dict[str, List[str]]:
    """
    Extract author names from text using spaCy NER.
    """
    if nlp_model is None:
        nlp_model = nlp

    result = {"positive": [], "negative": []}
    is_negative_context = any(neg in text.lower() for neg in NEGATIVE_WORDS)

    if nlp_model is None:
        return result

    doc = nlp_model(text)

    # spaCy's PERSON entity may be an author
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            author = ent.text.strip()
            if is_negative_context:
                if author not in result["negative"]:
                    result["negative"].append(author)
            else:
                if author not in result["positive"]:
                    result["positive"].append(author)

    return result

def extract_reading_level(text: str) -> Optional[str]:
    """
    Detect reading level preference.
    """
    text_lower = text.lower()

    if any(word in text_lower for word in ["easy", "beginner", "simple", "light"]):
        return "beginner"
    elif any(word in text_lower for word in ["complex", "advanced", "challenging", "difficult"]):
        return "advanced"
    elif any(word in text_lower for word in ["intermediate", "moderate"]):
        return "intermediate"

    return None

def extract_book_titles(text: str) -> List[str]:
    """
    Extract possible book titles (using quote detection).
    """
    import re
    # Match text enclosed in quotes
    quoted = re.findall(r'"([^"]+)"|"([^"]+)"', text)
    titles = []
    for match in quoted:
        title = match[0] or match[1]
        titles.append(title.strip())
    return titles

def parse_user_input(text: str, nlp_model=None) -> UserPreferences:
    """
    Parse user input and extract all preference information.
    """
    prefs = UserPreferences()

    # 1. Extract genre preferences
    genre_mentions = extract_genre_mentions(text)
    prefs.preferred_genres = genre_mentions["positive"]
    prefs.disliked_genres = genre_mentions["negative"]

    # 2. Extract author preferences
    author_mentions = extract_author_mentions(text, nlp_model)
    prefs.liked_authors = author_mentions["positive"]
    prefs.disliked_authors = author_mentions["negative"]

    # 3. Extract reading level
    prefs.reading_level = extract_reading_level(text)

    # 4. Extract book titles (for feedback)
    titles = extract_book_titles(text)
    for title in titles:
        # Detect positive or negative feedback
        if any(pos in text.lower() for pos in POSITIVE_WORDS):
            prefs.past_feedback.append({"book": title, "liked": True})
        elif any(neg in text.lower() for neg in NEGATIVE_WORDS):
            prefs.past_feedback.append({"book": title, "liked": False})

    return prefs

def merge_with_session_memory(current_prefs: UserPreferences, session_memory: dict) -> UserPreferences:
    """
    Merge current parsed preferences with historical preferences from N8N Session Memory.
    """
    merged = UserPreferences()

    # Load historical preferences from session memory
    merged.update(session_memory)

    # Add current preferences (deduplicated)
    # New dislike removes from liked and vice versa
    for genre in current_prefs.preferred_genres:
        if genre in merged.disliked_genres:
            merged.disliked_genres.remove(genre)
        if genre not in merged.preferred_genres:
            merged.preferred_genres.append(genre)

    for genre in current_prefs.disliked_genres:
        if genre in merged.preferred_genres:
            merged.preferred_genres.remove(genre)
        if genre not in merged.disliked_genres:
            merged.disliked_genres.append(genre)

    # Update reading level (latest takes priority)
    if current_prefs.reading_level:
        merged.reading_level = current_prefs.reading_level

    # Add new feedback
    for fb in current_prefs.past_feedback:
        if fb not in merged.past_feedback:
            merged.past_feedback.append(fb)

    return merged

# ============ Test ============
if __name__ == "__main__":
    test_inputs = [
        "I like sci-fi and fantasy, but I don't like mystery",
        "I'm looking for something by Stephen King",
        "I prefer beginner level books",
        "I loved 'Dune' but didn't like 'Foundation'",
        "Show me some romantic novels"
    ]

    print("Testing SpaCy NER extraction...\n")
    for text in test_inputs:
        prefs = parse_user_input(text)
        print(f"Input: {text}")
        print(f"Output: {prefs.to_dict()}")
        print("-" * 50)
