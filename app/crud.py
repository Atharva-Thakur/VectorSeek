import time
import numpy as np
import pandas as pd
from sqlalchemy import func
from app.models import Book
from app.services import compute_embedding, format_description

def format_book(book: Book, similarity: float = None) -> dict:
    return {
        "book_id": book.book_id,
        "title": book.title,
        "author": book.author,
        "average_rating": book.average_rating,
        "publisher": book.publisher,
        "description": format_description(book.description),
        "image_url": book.image_url,
        "similarity_score": similarity
    }

def vector_search(db, query_text: str, top_k: int):
    query_embedding = compute_embedding(query_text)
    results = db.query(
        Book,
        func.l2_distance(Book.embedding, query_embedding).label("distance")
    ).order_by("distance").limit(top_k).all()

    books = [
        format_book(book, 1 - min(distance / 10, 1))
        for book, distance in results
    ]
    return books

def text_search(db, query_text: str, top_k: int):
    query_tokens = ' & '.join(query_text.split())
    text_query = func.to_tsquery('english', query_tokens)
    results = db.query(
        Book,
        func.ts_rank(
            func.to_tsvector('english', Book.title + ' ' + Book.author + ' ' + func.coalesce(Book.description, '')),
            text_query
        ).label("rank")
    ).filter(
        func.to_tsvector('english', Book.title + ' ' + Book.author + ' ' + func.coalesce(Book.description, '')).op('@@')(text_query)
    ).order_by(func.desc("rank")).limit(top_k).all()

    books = [
        format_book(book, float(rank))
        for book, rank in results
    ]
    return books

def hybrid_search(db, query_text: str, top_k: int):
    query_embedding = compute_embedding(query_text)

    # Create vector search CTE
    vector_cte = db.query(
        Book.id,
        func.l2_distance(Book.embedding, query_embedding).label("vector_distance")
    ).cte("vector_results")

    query_tokens = ' & '.join(query_text.split())
    text_query = func.to_tsquery('english', query_tokens)
    
    # Create text search CTE
    text_cte = db.query(
        Book.id,
        func.ts_rank(
            func.to_tsvector('english', Book.title + ' ' + Book.author + ' ' + func.coalesce(Book.description, '')),
            text_query
        ).label("text_rank")
    ).filter(
        func.to_tsvector('english', Book.title + ' ' + Book.author + ' ' + func.coalesce(Book.description, '')).op('@@')(text_query)
    ).cte("text_results")

    # Combine results
    results = db.query(
        Book,
        (0.7 * (1 - func.min(vector_cte.c.vector_distance / 10, 1)) +
         0.3 * func.coalesce(text_cte.c.text_rank, 0)).label("hybrid_score")
    ).join(
        vector_cte, Book.id == vector_cte.c.id
    ).outerjoin(
        text_cte, Book.id == text_cte.c.id
    ).order_by(func.desc("hybrid_score")).limit(top_k).all()

    books = [
        format_book(book, float(score))
        for book, score in results
    ]
    return books

def get_book_by_id(db, book_id: int):
    return db.query(Book).filter(Book.book_id == book_id).first()

def get_similar_books(db, book_id: int, limit: int):
    book = get_book_by_id(db, book_id)
    if not book:
        return None

    results = db.query(
        Book,
        func.l2_distance(Book.embedding, book.embedding).label("distance")
    ).filter(Book.book_id != book_id).order_by("distance").limit(limit).all()

    books = [
        format_book(similar_book, 1 - min(distance / 10, 1))
        for similar_book, distance in results
    ]
    return books

def import_data(db, data_path: str, embeddings_path: str, batch_size: int = 1000):
    # Load data and embeddings
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_json(data_path)
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    if len(df) != len(embeddings):
        raise ValueError(f"Data ({len(df)} rows) and embeddings ({len(embeddings)} rows) length mismatch")
    
    df['embedding'] = embeddings.tolist()
    total_records = len(df)
    print(f"Importing {total_records} records in batches of {batch_size}")
    
    for i in range(0, total_records, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch = []
        for _, row in batch_df.iterrows():
            book = Book(
                book_id=row.get('book_id'),
                title=row.get('title'),
                author=row.get('author'),
                author_id=row.get('author_id'),
                work_id=row.get('work_id'),
                language=row.get('language'),
                average_rating=row.get('average_rating'),
                ratings_count=row.get('ratings_count'),
                publication_date=row.get('publication_date'),
                original_publication_date=row.get('original_publication_date'),
                format=row.get('format'),
                edition_information=row.get('edition_information'),
                publisher=row.get('publisher'),
                num_pages=row.get('num_pages'),
                series_name=row.get('series_name'),
                series_position=row.get('series_position'),
                description=row.get('description'),
                image_url=row.get('image_url'),
                shelves=row.get('shelves'),
                rating_distribution=row.get('rating_distribution'),
                embedding=row.get('embedding')
            )
            batch.append(book)
        db.bulk_save_objects(batch)
        db.commit()
        print(f"Imported records {i} to {i + len(batch_df)}")
    print("Data import completed.")

def create_indexes():
    from app.database import engine
    conn = engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_books_title_gin ON books USING gin(to_tsvector('english', title));")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_books_author_gin ON books USING gin(to_tsvector('english', author));")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_books_description_gin ON books USING gin(to_tsvector('english', description));")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_books_embedding ON books USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);")
    conn.commit()
    cursor.close()
    conn.close()
    print("Database indexes created.")
