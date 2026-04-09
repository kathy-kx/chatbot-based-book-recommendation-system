### Competency Questions (CQ)

- **CQ1:** Recommend a book by genre  
- **CQ2:** Recommend books similar to a given book  
- **CQ3:** Find books by a specific author  
- **CQ4:** Suggest beginner-level books on a topic  
- **CQ5:** Recommend popular books in a genre  
- **CQ6:** Recommend books based on user preferences  

### Test Cases

| # | CQ  | Test Question | Style   | Expected Intent                  | Actual Intent                    | Result | Failure Reason |
|---|-----|--------------|--------|--------------------------------|----------------------------------|--------|----------------|
| 1 | CQ1 | Can you recommend a science fiction book? | Formal | recommend_by_genre | recommend_by_genre | ✅ | |
| 2 | CQ1 | I wanna read something in the fantasy genre. | Casual | recommend_by_genre | recommend_by_genre | ✅ | |
| 3 | CQ1 | mystery books? | Short | recommend_by_genre | recommend_by_genre | ✅ | |
| 4 | CQ1 | Give me a suggeston in sci-fi genere. | Typo | recommend_by_genre | recommend_by_genre | ✅ | |
| 5 | CQ1 | What novels fall under the horror category? | Synonym | recommend_by_genre | Default Fallback Intent | ❌ | "novels"/"category"/"horror" not in training phrases |
| 6 | CQ2 | Could you suggest books similar to Dune? | Formal | recommend_similar_books | recommend_similar_books | ✅ | |
| 7 | CQ2 | I loved Harry Potter, what else is like it? | Casual | recommend_similar_books | recommend_similar_books | ✅ | |
| 8 | CQ2 | books like The Hobbit? | Short | recommend_similar_books | recommend_similar_books | ✅ | |
| 9 | CQ2 | Recommend somthing simlar to Gone Girl. | Typo | recommend_similar_books | recommend_similar_books | ✅ | |
| 10 | CQ2 | What titles are comparable to Python Crash Course? | Synonym | recommend_similar_books | recommend_similar_books | ✅ | |
| 11 | CQ3 | Please list all books written by J.K. Rowling. | Formal | find_books_by_author | find_books_by_author | ✅ | |
| 12 | CQ3 | What has Tolkien written? | Casual | find_books_by_author | find_books_by_author | ✅ | |
| 13 | CQ3 | Frank Herbert books? | Short | find_books_by_author | find_books_by_author | ✅ | |
| 14 | CQ3 | Show me books by Gillian Flyn. | Typo | find_books_by_author | find_books_by_author | ✅ | |
| 15 | CQ3 | Which works are authored by Eric Matthes? | Synonym | find_books_by_author | find_books_by_author | ✅ | |
| 16 | CQ4 | What beginner-level books do you have on programming? | Formal | recommend_beginner_books | recommend_beginner_books | ✅ | |
| 17 | CQ4 | I'm new to machine learning, any good starter books? | Casual | recommend_beginner_books | recommend_beginner_books | ✅ | |
| 18 | CQ4 | beginner data science book? | Short | recommend_beginner_books | recommend_beginner_books | ✅ | |
| 19 | CQ4 | Suggest an intro-level book on programing. | Typo | recommend_beginner_books | recommend_beginner_books | ✅ | |
| 20 | CQ4 | What entry-level books cover machine learning topics? | Synonym | recommend_beginner_books | recommend_beginner_books | ✅ | |
| 21 | CQ5 | What are the highest-rated fantasy books you know? | Formal | recommend_popular_books | recommend_popular_books | ✅ | |
| 22 | CQ5 | Which mystery books have the best ratings? | Casual | recommend_popular_books | recommend_popular_books | ✅ | |
| 23 | CQ5 | top rated sci-fi? | Short | recommend_popular_books | recommend_popular_books | ✅ | |
| 24 | CQ5 | What fantasey books are most popular? | Typo | recommend_popular_books | recommend_similar_books | ❌ | Typo "fantasey" misclassified |
| 25 | CQ5 | Which well-reviewed titles belong to the mystery genre? | Synonym | recommend_popular_books | recommend_by_genre | ❌ | "well-reviewed" not trained |
| 26 | CQ6 | Based on my interest in science fiction, what would you recommend? | Formal | recommend_by_user_preference | recommend_by_user_preference | ✅ | |
| 27 | CQ6 | I'm really into fantasy, got any suggestions for me? | Casual | recommend_by_user_preference | recommend_by_user_preference | ✅ | |
| 28 | CQ6 | recommend based on my preferences? | Short | recommend_by_user_preference | recommend_by_user_preference | ✅ | |
| 29 | CQ6 | I like mistery novels, what shoud I read? | Typo | recommend_by_user_preference | Default Fallback Intent | ❌ | Multiple typos |
| 30 | CQ6 | Suggest titles that align with my taste in science fiction. | Synonym | recommend_by_user_preference | recommend_by_genre | ❌ | Phrase not trained |

