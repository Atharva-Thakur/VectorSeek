from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from .db import SessionLocal
from .models import Book
from .utils import embed_text

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/books/{book_id}")
def get_book(book_id: int):
    db = next(get_db())
    book = db.query(Book).filter(Book.book_id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@router.get("/search")
def search_books(query: str, top_k: int = 5):
    db = next(get_db())
    vector = embed_text(query)
    results = db.execute(
        f"""
        SELECT *, embedding <#> cube(array{vector}) AS distance
        FROM books
        ORDER BY distance ASC
        LIMIT {top_k}
        """
    ).fetchall()
    return [dict(row) for row in results]
