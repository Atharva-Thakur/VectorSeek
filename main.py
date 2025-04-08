import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values
import psycopg2.extensions
import time  # Import the time module for measuring execution time

# Load data and embeddings
df = pd.read_csv("data/data1000000/data.csv")
embeddings = np.load("data/data1000000/embeddings.npy")
df['embeddings'] = list(embeddings)

# Database connection parameters
host = "localhost"
user = "postgres"
password = "password"
port = 5432

# Connect to the PostgreSQL database
def create_db_connection():
    try:
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )
        print(f"Database connection successful.")
        return conn
    except Exception as e:
        print(f"Error while connecting to database: {e}")
        return None

# Create the embeddings table if it doesn't exist
def create_table(conn):
    try:
        start_time = time.time()  # Start time measurement
        print("Creating the embeddings table if it doesn't exist...")
        with conn.cursor() as cur:
            # Install the pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("pgvector extension checked/installed.")
            
            # Create the table with the proper structure
            table_create_command = """
            CREATE TABLE IF NOT EXISTS embeddings (
                id bigserial primary key, 
                title text,
                author text,
                content text,
                embedding vector(384)
            );
            """
            cur.execute(table_create_command)
            conn.commit()
            print("Table created successfully.")
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
                ON embeddings USING hnsw (embedding vector_l2_ops);
            """)
            conn.commit()
            print("HNSW index created successfully.")
            
            end_time = time.time()  # End time measurement
            print(f"Table and index creation completed. Time taken: {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()

# Insert embeddings and metadata from DataFrame into PostgreSQL in batch
def insert_embeddings(conn, df, batch_size=10000):
    try:
        total_rows = len(df)
        print(f"Inserting {total_rows} embeddings into the database in batches of {batch_size}...")

        start_time = time.time()
        inserted = 0

        with conn.cursor() as cur:
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]

                # Prepare data for insertion
                data_list = [
                    (row['title'], row['author'], row['description'], np.array(row['embeddings']).tolist()) 
                    for _, row in batch.iterrows()
                ]

                insert_command = """
                INSERT INTO embeddings (title, author, content, embedding) 
                VALUES %s
                """
                try:
                    execute_values(cur, insert_command, data_list)
                    conn.commit()
                    inserted += len(data_list)
                    print(f"✅ Inserted batch {i // batch_size + 1}: {len(data_list)} rows (Total so far: {inserted})")
                except Exception as batch_e:
                    print(f"⚠️ Error inserting batch {i // batch_size + 1}: {batch_e}")
                    conn.rollback()

        end_time = time.time()
        print(f"Finished inserting embeddings. Total inserted: {inserted}/{total_rows}. Time taken: {end_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"Fatal error during embedding insertion: {e}")
        conn.rollback()

# Perform a vector search for a query
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
            SELECT title, author, content, embedding, 
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
                print(f"Similarity Score: {result[4]}")
                # print(f"Content: {result[2]}\n")
                print("------")
    except Exception as e:
        print(f"Error during vector search: {e}")

def test_vector_search_with_filters(conn, model, query, author_filter=None, title_keyword=None, min_content_length=None):
    try:
        print(f"\nTesting filtered vector search for query: '{query}'")
        query_embedding = model.encode([query])[0]

        base_query = """
        SELECT title, author, content, embedding, 
               embedding <=> %s::vector AS similarity
        FROM embeddings
        WHERE 1=1
        """
        params = [query_embedding.tolist()]

        # Add filters dynamically
        if author_filter:
            base_query += " AND author ILIKE %s"
            params.append(f"%{author_filter}%")
        if title_keyword:
            base_query += " AND title ILIKE %s"
            params.append(f"%{title_keyword}%")
        if min_content_length:
            base_query += " AND length(content) >= %s"
            params.append(min_content_length)

        # Append ordering and limit
        base_query += " ORDER BY similarity LIMIT 5;"

        with conn.cursor() as cur:
            cur.execute(base_query, tuple(params))
            results = cur.fetchall()
            print(f"Found {len(results)} results with filters:")
            for result in results:
                print(f"Title: {result[0]}, Author: {result[1]}, Similarity: {result[4]:.4f}")
                print("------")

    except Exception as e:
        print(f"Error during filtered vector search: {e}")

def run_tests():
    print("Running test cases...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    conn = create_db_connection()
    if conn is None:
        return

    test_vector_search_with_filters(conn, model, query="neural networks")
    test_vector_search_with_filters(conn, model, query="artificial intelligence", author_filter="John")
    test_vector_search_with_filters(conn, model, query="data mining", title_keyword="mining")
    test_vector_search_with_filters(conn, model, query="deep learning", min_content_length=500)

    conn.close()

def main():
    # Load sentence transformer model
    print("Loading the Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 1: Connect to the database
    conn = create_db_connection()
    if conn is None:
        return

    # Step 2: Create the table and the index if they don't exist
    # print("Creating table and index...")
    # create_table(conn) 

    # # # Step 3: Insert embeddings into the table
    # print("Inserting embeddings into the table...")
    # insert_embeddings(conn, df) 

    # Step 4: Perform vector search for a sample query
    # query = "Machine learning"
    # print(f"Performing vector search with query: '{query}'")
    # perform_vector_search(conn, query, model)

    run_tests()

    # Close the connection
    print("Closing the database connection.")
    conn.close()

if __name__ == "__main__":
    main()
