import time
import numpy as np

def perform_vector_search(conn, query, model):
    try:
        start_time = time.time()
        print(f"Generating embedding for the query: '{query}'")
        # Generate query embedding using the sentence-transformer model
        query_embedding = model.encode([query])[0]  # single query, so extract the first item
        end_time = time.time()  # End time measurement
        print(f"Generating embedding completed. Time taken: {end_time - start_time:.2f} seconds.")

        start_time = time.time()  # Start time measurement
        print("Performing vector search for the most similar documents...")
        with conn.cursor() as cur:
            # Perform a vector search on the embeddings table
            search_query = """
            SELECT title, author, content, 
                   embedding <=> %s::vector AS similarity
            FROM embeddings
            ORDER BY similarity
            LIMIT 5;
            """
            # Pass the query embedding as a list, and cast it to the 'vector' type in SQL
            cur.execute(search_query, (query_embedding.tolist(),))  # Pass the list directly
            results = cur.fetchall()

            # Output results
            end_time = time.time()  # End time measurement
            print(f"Vector search completed. Time taken: {end_time - start_time:.2f} seconds.")
            print("Top 5 most similar documents:")
            for result in results:
                print(f"Title: {result[0]}, Author: {result[1]}")
                print(f"Similarity Score: {result[3]}")
                # print(f"Content: {result[2]}\n")
                print("------")
    except Exception as e:
        print(f"Error during vector search: {e}")