import logging
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field

from app.config.settings import settings
from app.data.loader import load_data_to_db
from app.db.operations import (
    get_book_by_id, 
    get_books, 
    count_books, 
    search_books_by_text,
    count_search_results
)
from app.services.search import SearchService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for requests and responses
class VectorSearchRequest(BaseModel):
    vector: List[float] = Field(..., description="Query vector for similarity search")
    distance_strategy: str = Field(
        default="cosine", 
        description="Distance strategy: 'cosine', 'l2', or 'inner_product'"
    )
    threshold: float = Field(
        default=settings.DEFAULT_SIMILARITY_THRESHOLD,
        description="Similarity threshold (lower is more restrictive)"
    )
    limit: int = Field(
        default=settings.DEFAULT_MAX_RESULTS,
        description="Maximum number of results to return"
    )

class BookResponse(BaseModel):
    book_id: int
    title: str
    author: str
    average_rating: float
    description: str
    image_url: str
    similarity_score: Optional[float] = None
    
    class Config:
        from_attributes = True

class PaginatedBooksResponse(BaseModel):
    items: List[BookResponse]
    total: int
    page: int
    size: int
    pages: int

class DataLoadResponse(BaseModel):
    status: str
    records_loaded: Optional[int] = None
    message: Optional[str] = None

# API endpoints
@router.post("/load-data", response_model=DataLoadResponse, status_code=201)
async def api_load_data():
    """Load data from CSV and embeddings into the database."""
    logger.info("Loading data into database")
    result = await load_data_to_db()
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result

@router.get("/books", response_model=PaginatedBooksResponse)
async def api_get_books(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE, description="Page size")
):
    """Get a paginated list of books."""
    skip = (page - 1) * size
    books = await get_books(skip=skip, limit=size)
    total = await count_books()
    
    # Convert to response model
    items = [
        BookResponse(
            book_id=book.book_id,
            title=book.title,
            author=book.author,
            average_rating=book.average_rating,
            description=book.description,
            image_url=book.image_url
        ) for book in books
    ]
    
    return PaginatedBooksResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        pages=(total + size - 1) // size  # Calculate total pages
    )

@router.get("/books/{book_id}", response_model=BookResponse)
async def api_get_book(
    book_id: int = Path(..., description="Book ID")
):
    """Get a book by its ID."""
    book = await get_book_by_id(book_id)
    
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    return BookResponse(
        book_id=book.book_id,
        title=book.title,
        author=book.author,
        average_rating=book.average_rating,
        description=book.description,
        image_url=book.image_url
    )

@router.get("/search", response_model=PaginatedBooksResponse)
async def api_search_books(
    q: str = Query(..., min_length=1, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE, description="Page size")
):
    """Search books by title, author, or description."""
    skip = (page - 1) * size
    books = await search_books_by_text(q, skip=skip, limit=size)
    total = await count_search_results(q)
    
    # Convert to response model
    items = [
        BookResponse(
            book_id=book.book_id,
            title=book.title,
            author=book.author,
            average_rating=book.average_rating,
            description=book.description,
            image_url=book.image_url
        ) for book in books
    ]
    
    return PaginatedBooksResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        pages=(total + size - 1) // size  # Calculate total pages
    )

@router.post("/vector-search", response_model=List[BookResponse])
async def api_vector_search(request: VectorSearchRequest):
    """Perform vector similarity search."""
    try:
        # Validate distance strategy
        if request.distance_strategy not in ["cosine", "l2", "inner_product"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid distance strategy. Must be one of: 'cosine', 'l2', 'inner_product'"
            )
        
        # Validate vector dimensions
        if len(request.vector) != settings.VECTOR_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension must be {settings.VECTOR_DIMENSION}"
            )
        
        # Perform vector search
        results = await SearchService.vector_search(
            query_vector=request.vector,
            distance_strategy=request.distance_strategy,
            limit=request.limit,
            threshold=request.threshold
        )
        
        # Convert to response model
        return [
            BookResponse(
                book_id=book.book_id,
                title=book.title,
                author=book.author,
                average_rating=book.average_rating,
                description=book.description,
                image_url=book.image_url,
                similarity_score=distance
            ) for book, distance in results
        ]
    
    except Exception as e:
        logger.error(f"Vector search API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))