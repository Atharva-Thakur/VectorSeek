import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values
import psycopg2.extensions
import time  # Import the time module for measuring execution time

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
        start_time = time.time()  # Start time measurement
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )
        end_time = time.time()  # End time measurement
        print(f"Database connection successful. Time taken: {end_time - start_time:.2f} seconds.")
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
            end_time = time.time()  # End time measurement
            print(f"Table created successfully. Time taken: {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()

# Insert embeddings and metadata from DataFrame into PostgreSQL in batch
def insert_embeddings(conn, df):
    try:
        start_time = time.time()  # Start time measurement
        print(f"Inserting {len(df)} embeddings into the database...")
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
            end_time = time.time()  # End time measurement
            print(f"{len(df)} embeddings inserted successfully! Time taken: {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error inserting embeddings: {e}")
        conn.rollback()

# Perform a vector search for a query
def perform_vector_search(conn, query, model):
    try:
        start_time = time.time()  # Start time measurement
        print(f"Generating embedding for the query: '{query}'")
        # Generate query embedding using the sentence-transformer model
        query_embedding = model.encode([query])[0]  # single query, so extract the first item
        
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
                print(f"Content: {result[2]}\n")
    except Exception as e:
        print(f"Error during vector search: {e}")

def main():
    # Load sentence transformer model
    print("Loading the Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 1: Connect to the database
    conn = create_db_connection()
    if conn is None:
        return

    # Step 2: Create the table if it doesn't exist
    print("Creating table...")
    create_table(conn) 

    # Step 3: Insert embeddings into the table
    print("Inserting embeddings into the table...")
    insert_embeddings(conn, df) 

    # Step 4: Perform vector search for a sample query
    query = "What is machine learning?"
    print(f"Performing vector search with query: '{query}'")
    perform_vector_search(conn, query, model)

    # Close the connection
    print("Closing the database connection.")
    conn.close()

if __name__ == "__main__":
    main()
