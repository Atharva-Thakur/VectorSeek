from sqlalchemy import Column, Integer, String, Float, Text
from sqlalchemy.dialects.postgresql import ARRAY, REAL
from app.database import Base

class Book(Base):
    __tablename__ = "books"
    
    id = Column(Integer, primary_key=True, index=True)
    book_id = Column(Integer, unique=True, index=True)
    title = Column(String(500), index=True)
    author = Column(String(255), index=True)
    author_id = Column(Integer)
    work_id = Column(Integer)
    language = Column(String(50))
    average_rating = Column(Float)
    ratings_count = Column(Integer)
    publication_date = Column(String(50))
    original_publication_date = Column(String(50))
    format = Column(String(100))
    edition_information = Column(Text, nullable=True)
    publisher = Column(String(255))
    num_pages = Column(String(50), nullable=True)
    series_name = Column(String(255), nullable=True)
    series_position = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=True)
    shelves = Column(String(1000), nullable=True)
    rating_distribution = Column(String(255), nullable=True)
    embedding = Column(ARRAY(REAL))  # pgvector embedding
