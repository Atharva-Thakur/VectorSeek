import time
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db
from app.services import compute_embedding  # if needed

app = FastAPI(
    title="Book Vector Search API",
    description="API for searching books using vector embeddings and hybrid search",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    return {"message": "Book Vector Search API", "status": "running"}

@app.post("/search", response_model=list[schemas.BookResponse])
async def search_books(search_query: schemas.SearchQuery, db: Session = Depends(get_db)):
    start_time = time.time()
    query_text = search_query.query
    top_k = search_query.top_k
    search_type = search_query.search_type

    try:
        if search_type == "vector":
            books = crud.vector_search(db, query_text, top_k)
        elif search_type == "text":
            books = crud.text_search(db, query_text, top_k)
        elif search_type == "hybrid":
            books = crud.hybrid_search(db, query_text, top_k)
        else:
            raise HTTPException(status_code=400, detail="Invalid search type")
        
        execution_time = time.time() - start_time
        print(f"Search executed in {execution_time:.2f} seconds")
        return books
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/book/{book_id}", response_model=schemas.BookResponse)
async def get_book(book_id: int, db: Session = Depends(get_db)):
    book = crud.get_book_by_id(db, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return crud.format_book(book)

@app.get("/similar/{book_id}", response_model=list[schemas.BookResponse])
async def get_similar_books(book_id: int, limit: int = Query(5, ge=1, le=100), db: Session = Depends(get_db)):
    books = crud.get_similar_books(db, book_id, limit)
    if books is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return books
