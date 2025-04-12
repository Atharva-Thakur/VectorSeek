import time
import numpy as np

def perform_vector_search(conn, query, model, author_filter=None, title_keyword=None, min_content_length=None, top_k=5):
    try:
        cursor = conn.cursor()

        print(f"Generating embedding for the query: '{query}'")
        start_time = time.time()
        # Generate query embedding using the sentence-transformer model
        query_embedding = model.encode([query])[0]
        end_time = time.time()
        print(f"Generating embedding completed. Time taken: {end_time - start_time:.2f} seconds.")

        filters = []
        params = []

        if author_filter:
            filters.append("author ILIKE %s")
            params.append(f"%{author_filter}%")
        if title_keyword:
            filters.append("title ILIKE %s")
            params.append(f"%{title_keyword}%")
        if min_content_length:
            filters.append("char_length(content) >= %s")
            params.append(min_content_length)

        where_clause = "WHERE " + " AND ".join(filters) if filters else ""

        sql = f"""
            SELECT id, title, author, content, embedding <=> %s::vector AS similarity
            FROM embeddings
            {where_clause}
            ORDER BY similarity
            LIMIT %s;
        """
        # Embedding must be the first param, and LIMIT is the last
        query_params = [query_embedding.tolist()] + params + [top_k]

        print("Performing vector search for the most similar documents...")
        start_time = time.time()

        cursor.execute(sql, query_params)
        rows = cursor.fetchall()

        end_time = time.time()
        print(f"Vector search completed. Time taken: {end_time - start_time:.2f} seconds.")
        for row in rows:
            print(f"Title: {row[1]}, Author: {row[2]}")
            print("------")
        return [
            {
                "id": row[0],
                "title": row[1],
                "author": row[2],
                "content": row[3],
                "distance": row[4]
            }
            for row in rows
        ]

    except Exception as e:
        print(f"Error during vector search: {e}")
        return []

def get_books_by_title(conn, title: str):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, author, content 
            FROM embeddings 
            WHERE title ILIKE %s 
            LIMIT 20;
        """, (f"%{title}%",))
        rows = cursor.fetchall()
        return [{"id": row[0], "title": row[1], "author": row[2], "content": row[3]} for row in rows]
        
    except Exception as e:
        print(f"Error during title search: {e}")
        return []