from sentence_transformers import SentenceTransformer
from database import create_db_connection, create_table
from embedding_loader import load_data_and_embeddings
from insert import insert_embeddings
from search import perform_vector_search
from config import MODEL_NAME

def run_tests(conn, model):
    perform_vector_search(conn, model, "neural networks")
    perform_vector_search(conn, model, "artificial intelligence", author_filter="John")
    perform_vector_search(conn, model, "data mining", title_keyword="Mining")
    perform_vector_search(conn, model, "deep learning", min_content_length=500)

def main():
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    conn = create_db_connection()
    if conn is None:
        return

    # create_table(conn)

    # df = load_data_and_embeddings()
    # insert_embeddings(conn, df)

    # perform_vector_search(conn, model, "neural networks", "Christopher M. Bishop" )
    run_tests(conn, model)

    print("Closing connection.")
    conn.close()

if __name__ == "__main__":
    main()
