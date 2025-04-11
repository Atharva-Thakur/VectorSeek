import psycopg2
from config import DB_CONFIG
import time

def create_db_connection():
    try:
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**DB_CONFIG)
        print("Database connection successful.")
        return conn
    except Exception as e:
        print(f"Error while connecting to database: {e}")
        return None

def create_table(conn):
    try:
        with conn.cursor() as cur:
            start_time = time.time()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id bigserial primary key, 
                    title text,
                    author text,
                    content text,
                    embedding vector(384)
                );
            """)
            # Drop the existing index if it exists (optional safety step)
            cur.execute("""
                DROP INDEX IF EXISTS embeddings_vector_idx;
            """)
            # Create HNSW index with tuned parameters
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
                ON embeddings USING hnsw (embedding vector_l2_ops)
                WITH (m = 16, ef_construction = 200);
            """)
            # Analyze for planner statistics (not strictly necessary for HNSW)
            cur.execute("ANALYZE embeddings;")
            conn.commit()
            print(f"Table and index created in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
