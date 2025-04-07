from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector

# Create base model
Base = declarative_base()

class Book(Base):
    __tablename__ = "books"
    
    book_id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    author = Column(String(255), nullable=False, index=True)
    author_id = Column(Integer, nullable=False)
    work_id = Column(Integer, nullable=False)
    language = Column(String(50), nullable=True)
    average_rating = Column(Float, nullable=False)
    ratings_count = Column(Integer, nullable=False)
    publication_date = Column(String(50), nullable=False)
    original_publication_date = Column(String(50), nullable=True)
    format = Column(String(100), nullable=True)
    edition_information = Column(Text, nullable=True)
    publisher = Column(String(255), nullable=True)
    num_pages = Column(Float, nullable=True)
    series_name = Column(String(255), nullable=True)
    series_position = Column(String(50), nullable=True)
    description = Column(Text, nullable=False)
    image_url = Column(String(500), nullable=False)
    shelves = Column(Text, nullable=False)  # Stored as JSON string
    rating_distribution = Column(Text, nullable=False)  # Stored as JSON string
    
    # Vector embedding for similarity search
    embedding = Column(Vector(384), nullable=True)  # Adjust dimension as needed
    
    def __repr__(self):
        return f"<Book(book_id={self.book_id}, title='{self.title}', author='{self.author}')>"