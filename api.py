from fastapi import APIRouter, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from database import create_db_connection
from search import perform_vector_search
from config import MODEL_NAME

router = APIRouter()
model = SentenceTransformer(MODEL_NAME)
conn = create_db_connection()

class SearchRequest(BaseModel):
    query: str
    author_filter: str = None
    title_keyword: str = None
    min_content_length: int = None
    top_k: int = 5

@router.post("/search")
def vector_search(request: SearchRequest):
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
