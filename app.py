from sentence_transformers import SentenceTransformer
from database import create_db_connection, create_table
from embedding_loader import load_data_and_embeddings
from insert import insert_embeddings
from search import perform_vector_search
from test_cases import test_vector_search_with_filters
from config import MODEL_NAME

def run_tests(conn, model):
    test_vector_search_with_filters(conn, model, "neural networks")
    test_vector_search_with_filters(conn, model, "artificial intelligence", author_filter="John")
    test_vector_search_with_filters(conn, model, "data mining", title_keyword="mining")
    test_vector_search_with_filters(conn, model, "deep learning", min_content_length=500)

def main():
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    conn = create_db_connection()
    if conn is None:
        return

    create_table(conn)

    df = load_data_and_embeddings()
    insert_embeddings(conn, df)

    perform_vector_search(conn, "Machine learning", model)
    # run_tests(conn, model)

    print("Closing connection.")
    conn.close()

if __name__ == "__main__":
    main()
