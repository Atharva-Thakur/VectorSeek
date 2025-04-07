from sqlalchemy import Column, Integer, String, Float, Text
from pgvector.sqlalchemy import Vector
from .db import Base

class Book(Base):
    __tablename__ = 'books'
    
    book_id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    author = Column(String)
    description = Column(Text)
    average_rating = Column(Float)
    num_pages = Column(Float)
    embedding = Column(Vector(384))  # assuming all-MiniLM-L6-v2 (384 dims)