import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.db.models import Base, Book

# Configure logging
logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
)

# Create async session
async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

@asynccontextmanager
async def get_db_session():
    """Get a database session."""
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            await session.close()

async def init_db():
    """Initialize database by creating tables and extensions if needed."""
    try:
        # Create pgvector extension if not exists
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

async def get_book_by_id(book_id: int):
    """Get a book by its ID."""
    async with get_db_session() as session:
        result = await session.get(Book, book_id)
        return result

async def get_books(skip: int = 0, limit: int = 20):
    """Get a list of books with pagination."""
    async with get_db_session() as session:
        query = session.query(Book).offset(skip).limit(limit)
        result = await session.execute(query)
        return result.scalars().all()

async def count_books():
    """Count the total number of books."""
    async with get_db_session() as session:
        query = session.query(Book.book_id).count()
        result = await session.execute(query)
        return result.scalar()

async def search_books_by_text(search_term: str, skip: int = 0, limit: int = 20):
    """Search books by title, author, or description."""
    search_pattern = f"%{search_term}%"
    async with get_db_session() as session:
        query = session.query(Book).filter(
            (Book.title.ilike(search_pattern)) | 
            (Book.author.ilike(search_pattern)) | 
            (Book.description.ilike(search_pattern))
        ).offset(skip).limit(limit)
        result = await session.execute(query)
        return result.scalars().all()

async def count_search_results(search_term: str):
    """Count the total number of search results."""
    search_pattern = f"%{search_term}%"
    async with get_db_session() as session:
        query = session.query(Book.book_id).filter(
            (Book.title.ilike(search_pattern)) | 
            (Book.author.ilike(search_pattern)) | 
            (Book.description.ilike(search_pattern))
        ).count()
        result = await session.execute(query)
        return result.scalar()

async def delete_all_books():
    """Delete all books from the database."""
    async with get_db_session() as session:
        await session.execute(text("TRUNCATE TABLE books"))
        await session.commit()