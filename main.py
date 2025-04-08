import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values
import psycopg2.extensions

# Load data and embeddings
df = pd.read_csv("data/data.csv")
embeddings = np.load("data/embeddings.npy")
df['embeddings'] = list(embeddings)

# Database connection parameters
host = "localhost"
user = "postgres"
password = "password"
port = 5432

# Connect to the PostgreSQL database
def create_db_connection():
    try:
        conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )
        return conn
    except Exception as e:
        print(f"Error while connecting to database: {e}")
        return None

# Create the embeddings table if it doesn't exist
def create_table(conn):
    try:
        with conn.cursor() as cur:
            # Install the pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
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
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()

# Insert embeddings and metadata from DataFrame into PostgreSQL in batch
def insert_embeddings(conn, df):
    try:
        with conn.cursor() as cur:
            # Prepare data for insertion (embedding is converted to list)
            data_list = [(row['title'], row['author'], row['description'], np.array(row['embeddings']).tolist()) 
                         for _, row in df.iterrows()]
            insert_command = """
            INSERT INTO embeddings (title, author, content, embedding) 
            VALUES %s
            """
            execute_values(cur, insert_command, data_list)
            conn.commit()
            print("Embeddings inserted successfully!")
    except Exception as e:
        print(f"Error inserting embeddings: {e}")
        conn.rollback()

# Perform a vector search for a query
def perform_vector_search(conn, query, model):
    try:
        # Generate query embedding using the sentence-transformer model
        query_embedding = model.encode([query])[0]  # single query, so extract the first item

        with conn.cursor() as cur:
            # Perform a vector search on the embeddings table
            search_query = """
            SELECT title, author, content, embedding, 
                   embedding <=> %s AS similarity
            FROM embeddings
            ORDER BY similarity
            LIMIT 5;
            """
            cur.execute(search_query, (query_embedding.tolist(),))  # execute query with embedding
            results = cur.fetchall()

            # Output results
            print("Top 5 most similar documents:")
            for result in results:
                print(f"Title: {result[0]}, Author: {result[1]}")
                print(f"Similarity Score: {result[4]}")
                print(f"Content: {result[2]}\n")
    except Exception as e:
        print(f"Error during vector search: {e}")

def main():
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 1: Connect to the database
    conn = create_db_connection()
    if conn is None:
        return

    # Step 2: Create the table if it doesn't exist
    create_table(conn)

    # Step 3: Insert embeddings into the table
    insert_embeddings(conn, df)

    # Step 4: Perform vector search for a sample query
    query = "What is machine learning?"
    perform_vector_search(conn, query, model)

    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()
