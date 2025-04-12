from fastapi import APIRouter, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from database import create_db_connection
from search import perform_vector_search, get_books_by_title
from config import MODEL_NAME

router = APIRouter()
model = SentenceTransformer(MODEL_NAME)
conn = create_db_connection()

class VectorSearchRequest(BaseModel):
    query: str
    author_filter: str = None
    title_keyword: str = None
    min_content_length: int = None
    top_k: int = 5

class TitleRequest(BaseModel):
    title: str

@router.post("/search")
def vector_search(request: VectorSearchRequest):
    if not conn:
        return {"error": "Database connection failed."}

    results = perform_vector_search(
        conn,
        query=request.query,
        model=model,
        author_filter=request.author_filter,
        title_keyword=request.title_keyword,
        min_content_length=request.min_content_length,
        top_k=request.top_k
    )
    return {"results": results}

@router.post("/by-title")
def books_by_title(request: TitleRequest):
    if not conn:
        return {"error": "Database connection failed."}

    results = get_books_by_title(conn, request.title)
    return {"results": results}