def test_vector_search_with_filters(conn, model, query, author_filter=None, title_keyword=None, min_content_length=None):
    try:
        print(f"\nRunning filtered search for: '{query}'")
        query_embedding = model.encode([query])[0]
        query_sql = """
        SELECT title, author, content, embedding, 
               embedding <=> %s::vector AS similarity
        FROM embeddings
        WHERE 1=1
        """
        params = [list(query_embedding)]
        if author_filter:
            query_sql += " AND author ILIKE %s"
            params.append(f"%{author_filter}%")
        if title_keyword:
            query_sql += " AND title ILIKE %s"
            params.append(f"%{title_keyword}%")
        if min_content_length:
            query_sql += " AND length(content) >= %s"
            params.append(min_content_length)

        query_sql += " ORDER BY similarity LIMIT 5;"

        with conn.cursor() as cur:
            cur.execute(query_sql, tuple(params))
            results = cur.fetchall()
            for result in results:
                print(f"Title: {result[0]}, Author: {result[1]}, Similarity: {result[4]:.4f}")
                print("------")
    except Exception as e:
        print(f"Error in filtered search: {e}")
