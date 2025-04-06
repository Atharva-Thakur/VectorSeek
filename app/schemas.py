from pydantic import BaseModel, Field
from typing import Optional

class BookResponse(BaseModel):
    book_id: int
    title: str
    author: str
    average_rating: float
    publisher: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    similarity_score: Optional[float] = None

    class Config:
        orm_mode = True

class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=100)
    search_type: str = Field("hybrid", regex="^(vector|text|hybrid)$")
