import pandas as pd
import numpy as np
from .models import Book
from .db import SessionLocal, engine, Base

Base.metadata.create_all(bind=engine)

def ingest():
    df = pd.read_csv("/app/data/data.csv")
    embeddings = np.load("/app/data/embeddings.npy")

    db = SessionLocal()
    if db.query(Book).first():  # Skip if data already exists
        print("Data already ingested. Skipping.")
        db.close()
        return
    try:
        for i, row in df.iterrows():
            book = Book(
                book_id=row['book_id'],
                title=row['title'],
                author=row['author'],
                description=row['description'],
                average_rating=row['average_rating'],
                num_pages=row['num_pages'],
                embedding=embeddings[i].tolist()
            )
            db.add(book)
            if i % 1000 == 0:
                db.commit()
        db.commit()
    finally:
        db.close()
