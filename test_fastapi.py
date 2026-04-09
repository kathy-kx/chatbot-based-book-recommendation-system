"""
Unit and Integration Tests — Chatbot-Based Book Recommendation System
Run from the project root directory:
    pytest test_fastapi.py -v
"""

import os
import sys
import pytest
import pandas as pd

# Ensure tests run with the project root as the working directory
# (cosine_similarity.py loads data files with relative paths)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — spacy_ner Unit Tests
# ═══════════════════════════════════════════════════════════════

class TestExtractGenreMentions:
    """Unit tests for spacy_ner.extract_genre_mentions()"""

    def test_positive_scifi_alias(self):
        """'sci-fi' alias maps to 'scientific' genre"""
        from spacy_ner import extract_genre_mentions
        result = extract_genre_mentions("I love sci-fi books")
        assert "scientific" in result["positive"]

    def test_positive_science_fiction(self):
        """'science fiction' alias maps to 'scientific'"""
        from spacy_ner import extract_genre_mentions
        result = extract_genre_mentions("I enjoy science fiction a lot")
        assert "scientific" in result["positive"]

    def test_negative_romance(self):
        """Negative sentiment correctly tags 'romance' → 'romantic' as disliked"""
        from spacy_ner import extract_genre_mentions
        result = extract_genre_mentions("I don't like romance books")
        assert "romantic" in result["negative"]
        assert "romantic" not in result["positive"]

    def test_mixed_positive_and_negative(self):
        """Per-mention window correctly splits positive and negative genres"""
        from spacy_ner import extract_genre_mentions
        result = extract_genre_mentions("I like sci-fi but I don't like romance")
        assert "scientific" in result["positive"]
        assert "romantic" in result["negative"]

    def test_typo_alias_fantasey(self):
        """Typo 'fantasey' is a registered alias for 'fiction'"""
        from spacy_ner import extract_genre_mentions
        result = extract_genre_mentions("I want a fantasey book")
        assert "fiction" in result["positive"]

    def test_no_genre_mentioned(self):
        """Returns empty lists when no genre keywords are present"""
        from spacy_ner import extract_genre_mentions
        result = extract_genre_mentions("I want something to read tonight")
        assert result["positive"] == []
        assert result["negative"] == []

    def test_history_alias(self):
        """'historical' maps to 'history'"""
        from spacy_ner import extract_genre_mentions
        result = extract_genre_mentions("I enjoy historical novels")
        assert "history" in result["positive"]

    def test_self_help_maps_to_psychology(self):
        """'self-help' maps to 'psychology'"""
        from spacy_ner import extract_genre_mentions
        result = extract_genre_mentions("I like self-help books")
        assert "psychology" in result["positive"]


class TestExtractReadingLevel:
    """Unit tests for spacy_ner.extract_reading_level()"""

    def test_beginner_keyword(self):
        from spacy_ner import extract_reading_level
        assert extract_reading_level("I'm a complete beginner") == "beginner"

    def test_easy_keyword(self):
        from spacy_ner import extract_reading_level
        assert extract_reading_level("something easy to read") == "beginner"

    def test_advanced_keyword(self):
        from spacy_ner import extract_reading_level
        assert extract_reading_level("I want something challenging") == "advanced"

    def test_intermediate_keyword(self):
        from spacy_ner import extract_reading_level
        assert extract_reading_level("I prefer intermediate level books") == "intermediate"

    def test_no_level_returns_none(self):
        from spacy_ner import extract_reading_level
        assert extract_reading_level("recommend me a book about history") is None


class TestParseUserInput:
    """Integration tests for spacy_ner.parse_user_input()"""

    def test_genre_and_reading_level(self):
        from spacy_ner import parse_user_input
        prefs = parse_user_input("I like fantasy books and I'm a beginner")
        assert "fiction" in prefs.preferred_genres
        assert prefs.reading_level == "beginner"

    def test_dislike_only(self):
        from spacy_ner import parse_user_input
        prefs = parse_user_input("I hate romance novels")
        assert "romantic" in prefs.disliked_genres
        assert "romantic" not in prefs.preferred_genres

    def test_multiple_likes(self):
        from spacy_ner import parse_user_input
        prefs = parse_user_input("I enjoy sci-fi and history")
        assert "scientific" in prefs.preferred_genres
        assert "history" in prefs.preferred_genres

    def test_empty_input(self):
        from spacy_ner import parse_user_input
        prefs = parse_user_input("")
        assert prefs.preferred_genres == []
        assert prefs.disliked_genres == []
        assert prefs.reading_level is None


class TestMergeWithSessionMemory:
    """Unit tests for spacy_ner.merge_with_session_memory()"""

    def test_accumulates_genres_across_turns(self):
        """Preferences from turn 1 persist into turn 2"""
        from spacy_ner import parse_user_input, merge_with_session_memory
        turn1 = parse_user_input("I like sci-fi")
        turn2 = parse_user_input("I also like history")
        merged = merge_with_session_memory(turn2, turn1.to_dict())
        assert "scientific" in merged.preferred_genres
        assert "history" in merged.preferred_genres

    def test_dislike_overrides_previous_like(self):
        """If a genre was liked before but is now disliked, it moves to dislikes"""
        from spacy_ner import parse_user_input, merge_with_session_memory
        turn1 = parse_user_input("I like sci-fi")
        turn2 = parse_user_input("actually I don't like science fiction anymore")
        merged = merge_with_session_memory(turn2, turn1.to_dict())
        assert "scientific" not in merged.preferred_genres
        assert "scientific" in merged.disliked_genres

    def test_reading_level_updates(self):
        """Latest reading level preference takes priority"""
        from spacy_ner import parse_user_input, merge_with_session_memory
        turn1 = parse_user_input("I prefer beginner books")
        turn2 = parse_user_input("actually give me something advanced")
        merged = merge_with_session_memory(turn2, turn1.to_dict())
        assert merged.reading_level == "advanced"

    def test_empty_session_returns_current(self):
        """Merging with empty session memory just returns current prefs"""
        from spacy_ner import parse_user_input, merge_with_session_memory
        prefs = parse_user_input("I like biography")
        merged = merge_with_session_memory(prefs, {})
        assert "biography" in merged.preferred_genres


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — fastapi_app Logic Unit Tests
# ═══════════════════════════════════════════════════════════════

class TestDetectIntent:
    """
    Unit tests for fastapi_app.detect_intent().
    One test per Competency Question (CQ1–CQ6) plus edge cases.
    """

    @pytest.fixture(autouse=True)
    def blank_prefs(self):
        from spacy_ner import UserPreferences
        self.prefs = UserPreferences()

    def test_cq1_genre_via_prefs(self):
        """CQ1: parsed genre preference routes to recommend_by_genre"""
        from fastapi_app import detect_intent
        from spacy_ner import parse_user_input
        prefs = parse_user_input("I like fantasy books")
        assert detect_intent("I like fantasy books", prefs) == "recommend_by_genre"

    def test_cq2_similar_books(self):
        """CQ2: 'similar to' phrase routes to recommend_similar"""
        from fastapi_app import detect_intent
        assert detect_intent("Could you suggest books similar to Dune?", self.prefs) == "recommend_similar"

    def test_cq2_books_like(self):
        """CQ2: 'books like' phrase also routes to recommend_similar"""
        from fastapi_app import detect_intent
        assert detect_intent("books like Harry Potter", self.prefs) == "recommend_similar"

    def test_cq3_author_by_keyword(self):
        """CQ3: 'books by' keyword routes to find_by_author"""
        from fastapi_app import detect_intent
        assert detect_intent("books by Stephen King", self.prefs) == "find_by_author"

    def test_cq3_written_by(self):
        """CQ3: 'written by' keyword routes to find_by_author"""
        from fastapi_app import detect_intent
        assert detect_intent("Please list all books written by J.K. Rowling", self.prefs) == "find_by_author"

    def test_cq4_beginner(self):
        """CQ4: 'beginner' keyword routes to recommend_beginner"""
        from fastapi_app import detect_intent
        assert detect_intent("What beginner-level books do you have?", self.prefs) == "recommend_beginner"

    def test_cq4_easy(self):
        """CQ4: 'easy' keyword also routes to recommend_beginner"""
        from fastapi_app import detect_intent
        assert detect_intent("I want an easy read", self.prefs) == "recommend_beginner"

    def test_cq5_best_rated(self):
        """CQ5: 'best' keyword routes to recommend_popular"""
        from fastapi_app import detect_intent
        assert detect_intent("What are the best rated books?", self.prefs) == "recommend_popular"

    def test_cq5_top_rated(self):
        """CQ5: 'top rated' phrase routes to recommend_popular"""
        from fastapi_app import detect_intent
        assert detect_intent("top rated sci-fi?", self.prefs) == "recommend_popular"

    def test_cq2_takes_priority_over_cq3(self):
        """CQ2 should fire before CQ3 even when an author name is present"""
        from fastapi_app import detect_intent
        assert detect_intent("books like Harry Potter by Rowling", self.prefs) == "recommend_similar"

    def test_general_search_fallback(self):
        """No signals → falls through to general_search"""
        from fastapi_app import detect_intent
        assert detect_intent("recommend me something", self.prefs) == "general_search"


class TestApplyPreferenceFiltering:
    """Unit tests for fastapi_app.apply_preference_filtering()"""

    @pytest.fixture(autouse=True)
    def sample_results(self):
        self.results = [
            (pd.Series({"Title": "Sci-Fi A", "Authors": "Author1", "keyword_category": "scientific"}), 0.8),
            (pd.Series({"Title": "Romance B", "Authors": "Author2", "keyword_category": "romantic"}), 0.7),
            (pd.Series({"Title": "History C", "Authors": "Author3", "keyword_category": "history"}), 0.6),
        ]

    def test_disliked_genre_excluded(self):
        from fastapi_app import apply_preference_filtering
        filtered = apply_preference_filtering(self.results, disliked_genres=["romantic"])
        titles = [b["Title"] for b, _ in filtered]
        assert "Romance B" not in titles

    def test_preferred_genre_score_boosted(self):
        """Preferred genre score ×2 should push History C (0.6→1.2) above Sci-Fi A (0.8)"""
        from fastapi_app import apply_preference_filtering
        filtered = apply_preference_filtering(self.results, preferred_genres=["history"])
        assert filtered[0][0]["Title"] == "History C"

    def test_no_filter_returns_all(self):
        from fastapi_app import apply_preference_filtering
        filtered = apply_preference_filtering(self.results)
        assert len(filtered) == 3

    def test_dislike_and_prefer_together(self):
        """Disliked excluded + preferred boosted in one call"""
        from fastapi_app import apply_preference_filtering
        filtered = apply_preference_filtering(
            self.results, preferred_genres=["history"], disliked_genres=["scientific"]
        )
        titles = [b["Title"] for b, _ in filtered]
        assert "Sci-Fi A" not in titles
        assert filtered[0][0]["Title"] == "History C"


class TestRecommendByAuthor:
    """Unit tests for fastapi_app.recommend_by_author()"""

    def test_known_author_returns_results(self):
        from fastapi_app import recommend_by_author
        results = recommend_by_author("Rowling")
        assert len(results) > 0

    def test_partial_author_name(self):
        """Partial name match should also work (fuzzy contains)"""
        from fastapi_app import recommend_by_author
        results = recommend_by_author("King")
        assert len(results) > 0

    def test_unknown_author_returns_empty(self):
        from fastapi_app import recommend_by_author
        results = recommend_by_author("XYZ_NoSuchAuthor_99999")
        assert results == []

    def test_result_contains_author_name(self):
        """All returned books should have the searched author in Authors field"""
        from fastapi_app import recommend_by_author
        results = recommend_by_author("Rowling", top_n=3)
        for book, _ in results:
            assert "rowling" in book["Authors"].lower()


class TestRecommendPopular:
    """Unit tests for fastapi_app.recommend_popular()"""

    def test_returns_correct_count(self):
        from fastapi_app import recommend_popular
        results = recommend_popular(top_n=5)
        assert len(results) == 5

    def test_sorted_by_popularity_score(self):
        from fastapi_app import recommend_popular
        results = recommend_popular(top_n=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_genre_filter_respected(self):
        from fastapi_app import recommend_popular
        results = recommend_popular(genres=["fiction"], top_n=5)
        genres = [b["keyword_category"] for b, _ in results]
        assert all(g == "fiction" for g in genres)

    def test_returns_results_without_genre(self):
        from fastapi_app import recommend_popular
        results = recommend_popular()
        assert len(results) > 0


class TestRecommendBeginner:
    """Unit tests for fastapi_app.recommend_beginner()"""

    def test_returns_results(self):
        from fastapi_app import recommend_beginner
        results = recommend_beginner(top_n=5)
        assert len(results) > 0

    def test_genre_filter_respected(self):
        from fastapi_app import recommend_beginner
        results = recommend_beginner(genres=["history"], top_n=5)
        genres = [b["keyword_category"] for b, _ in results]
        assert all(g == "history" for g in genres)

    def test_all_scores_equal_one(self):
        """recommend_beginner assigns a flat score of 1.0 to all results"""
        from fastapi_app import recommend_beginner
        results = recommend_beginner(top_n=5)
        assert all(s == 1.0 for _, s in results)


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — FastAPI Integration Tests (TestClient)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from fastapi_app import app
    return TestClient(app)


class TestAPIEndpoints:
    """End-to-end integration tests via FastAPI TestClient"""

    # ── Health / metadata endpoints ──────────────────────────

    def test_root_returns_genres(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "genres" in r.json()

    def test_genres_endpoint(self, client):
        r = client.get("/genres")
        assert r.status_code == 200
        genres = r.json()["genres"]
        assert isinstance(genres, list)
        assert len(genres) > 0

    # ── CQ1 — genre recommendation ───────────────────────────

    def test_chat_cq1_formal(self, client):
        """CQ1 formal: 'Can you recommend a science fiction book?'"""
        r = client.post("/chat", json={
            "message": "Can you recommend a science fiction book?",
            "session_id": "cq1_formal"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "recommend_by_genre"
        assert len(data["recommendations"]) > 0

    def test_chat_cq1_casual(self, client):
        """CQ1 casual: 'I wanna read something in the fantasy genre.'"""
        r = client.post("/chat", json={
            "message": "I wanna read something in the fantasy genre.",
            "session_id": "cq1_casual"
        })
        assert r.status_code == 200
        assert r.json()["intent"] == "recommend_by_genre"

    # ── CQ2 — similar books ──────────────────────────────────

    def test_chat_cq2_formal(self, client):
        """CQ2 formal: 'Could you suggest books similar to Dune?'"""
        r = client.post("/chat", json={
            "message": "Could you suggest books similar to Dune?",
            "session_id": "cq2_formal"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "recommend_similar"
        assert len(data["recommendations"]) > 0

    def test_chat_cq2_short(self, client):
        """CQ2 short: 'books like The Hobbit?'"""
        r = client.post("/chat", json={
            "message": "books like The Hobbit?",
            "session_id": "cq2_short"
        })
        assert r.status_code == 200
        assert r.json()["intent"] == "recommend_similar"

    # ── CQ3 — author search ──────────────────────────────────

    def test_chat_cq3_formal(self, client):
        """CQ3 formal: 'Please list all books written by J.K. Rowling.'"""
        r = client.post("/chat", json={
            "message": "Please list all books written by J.K. Rowling.",
            "session_id": "cq3_formal"
        })
        assert r.status_code == 200
        assert r.json()["intent"] == "find_by_author"

    def test_chat_cq3_short(self, client):
        """CQ3 short: 'Frank Herbert books?'"""
        r = client.post("/chat", json={
            "message": "books by Frank Herbert",
            "session_id": "cq3_short"
        })
        assert r.status_code == 200
        assert r.json()["intent"] == "find_by_author"

    # ── CQ4 — beginner books ─────────────────────────────────

    def test_chat_cq4_formal(self, client):
        """CQ4 formal: 'What beginner-level books do you have on programming?'"""
        r = client.post("/chat", json={
            "message": "What beginner-level books do you have on programming?",
            "session_id": "cq4_formal"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "recommend_beginner"
        assert len(data["recommendations"]) > 0

    def test_chat_cq4_casual(self, client):
        """CQ4 casual: 'I'm new to machine learning, any easy books for beginners?'"""
        r = client.post("/chat", json={
            "message": "I'm new to machine learning, any easy books for beginners?",
            "session_id": "cq4_casual"
        })
        assert r.status_code == 200
        assert r.json()["intent"] == "recommend_beginner"

    def test_chat_cq4_starter_new_to(self, client):
        """CQ4: 'starter' and 'new to' added to BEGINNER_SIGNALS — now correctly routed"""
        r = client.post("/chat", json={
            "message": "I'm new to machine learning, any good starter books?",
            "session_id": "cq4_starter"
        })
        assert r.status_code == 200
        assert r.json()["intent"] == "recommend_beginner"

    # ── CQ5 — popular books ──────────────────────────────────

    def test_chat_cq5_formal(self, client):
        """CQ5 formal: 'What are the highest rated fantasy books you know?' (no hyphen)"""
        r = client.post("/chat", json={
            "message": "What are the highest rated fantasy books you know?",
            "session_id": "cq5_formal"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "recommend_popular"
        assert len(data["recommendations"]) > 0

    def test_chat_cq5_hyphenated_highest_rated(self, client):
        """CQ5: 'highest-rated' (hyphen) added to POPULAR_SIGNALS — now correctly routed"""
        r = client.post("/chat", json={
            "message": "What are the highest-rated fantasy books you know?",
            "session_id": "cq5_hyphen"
        })
        assert r.status_code == 200
        assert r.json()["intent"] == "recommend_popular"

    def test_chat_cq5_short(self, client):
        """CQ5 short: 'top rated sci-fi?'"""
        r = client.post("/chat", json={
            "message": "top rated sci-fi?",
            "session_id": "cq5_short"
        })
        assert r.status_code == 200
        assert r.json()["intent"] == "recommend_popular"

    # ── CQ6 — preference-based recommendation ────────────────

    def test_chat_cq6_formal(self, client):
        """CQ6 formal: 'Based on my interest in science fiction, what would you recommend?'"""
        r = client.post("/chat", json={
            "message": "Based on my interest in science fiction, what would you recommend?",
            "session_id": "cq6_formal"
        })
        assert r.status_code == 200
        assert r.json()["intent"] == "recommend_by_genre"

    def test_chat_cq6_casual(self, client):
        """CQ6 casual: 'I'm really into fantasy, got any suggestions for me?'"""
        r = client.post("/chat", json={
            "message": "I'm really into fantasy, got any suggestions for me?",
            "session_id": "cq6_casual"
        })
        assert r.status_code == 200
        assert r.json()["intent"] == "recommend_by_genre"

    # ── Session memory ───────────────────────────────────────

    def test_session_memory_accumulates(self, client):
        """Preferences from turn 1 should appear in turn 2's session_memory"""
        sid = "test_session_accumulate"
        client.post("/chat", json={"message": "I like sci-fi", "session_id": sid})
        r2 = client.post("/chat", json={"message": "show me popular books", "session_id": sid})
        assert r2.status_code == 200
        mem = r2.json()["session_memory"]
        assert "scientific" in mem.get("preferred_genres", [])

    def test_session_reset_clears_memory(self, client):
        """'start over' should clear the session and return session_reset intent"""
        sid = "test_session_reset"
        client.post("/chat", json={"message": "I like history", "session_id": sid})
        r = client.post("/chat", json={"message": "start over", "session_id": sid})
        assert r.status_code == 200
        assert r.json()["intent"] == "session_reset"
        # Confirm session is empty after reset
        r2 = client.get(f"/session/{sid}")
        assert r2.json() == {}

    def test_session_get_endpoint(self, client):
        """GET /session/{id} should return the stored preferences"""
        sid = "test_get_session"
        client.post("/chat", json={"message": "I like biography books", "session_id": sid})
        r = client.get(f"/session/{sid}")
        assert r.status_code == 200
        assert "biography" in r.json().get("preferred_genres", [])

    def test_session_delete_endpoint(self, client):
        """DELETE /session/{id} should remove the session"""
        sid = "test_delete_session"
        client.post("/chat", json={"message": "I like fiction", "session_id": sid})
        client.delete(f"/session/{sid}")
        r = client.get(f"/session/{sid}")
        assert r.json() == {}

    # ── Response structure ───────────────────────────────────

    def test_chat_response_has_required_keys(self, client):
        r = client.post("/chat", json={"message": "recommend fantasy books", "session_id": "test_struct"})
        assert r.status_code == 200
        data = r.json()
        for key in ("output", "recommendations", "detected_preferences", "session_memory", "intent"):
            assert key in data, f"Missing key: {key}"

    def test_recommendation_item_structure(self, client):
        """Each recommendation object should have Title, Authors, keyword_category, similarity_score"""
        r = client.post("/chat", json={"message": "I like sci-fi", "session_id": "test_rec_struct"})
        assert r.status_code == 200
        recs = r.json()["recommendations"]
        assert len(recs) > 0
        for rec in recs:
            assert "Title" in rec
            assert "Authors" in rec
            assert "keyword_category" in rec
            assert "similarity_score" in rec

    def test_dislike_preference_reflected_in_response(self, client):
        """Books from disliked genres should not appear in recommendations"""
        sid = "test_dislike_filter"
        # First turn: express dislike
        client.post("/chat", json={"message": "I don't like romance books", "session_id": sid})
        # Second turn: ask for recommendations
        r = client.post("/chat", json={"message": "recommend me something", "session_id": sid})
        assert r.status_code == 200
        recs = r.json()["recommendations"]
        genres = [rec["keyword_category"] for rec in recs]
        assert "romantic" not in genres
