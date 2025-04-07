import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sqlalchemy import text

from app.config.settings import settings
from app.db.models import Book
from app.db.operations import get_db_session, delete_all_books

# Configure logging
logger = logging.getLogger(__name__)

async def load_data_to_db(batch_size=1000):
    """
    Load data from CSV and embeddings from NPY file into the database.
    Process in batches to manage memory usage.
    """
    try:
        # Check if files exist
        if not os.path.exists(settings.CSV_PATH):
            raise FileNotFoundError(f"CSV file not found: {settings.CSV_PATH}")
        
        if not os.path.exists(settings.EMBEDDINGS_PATH):
            raise FileNotFoundError(f"Embeddings file not found: {settings.EMBEDDINGS_PATH}")
        
        # Load embeddings
        logger.info(f"Loading embeddings from {settings.EMBEDDINGS_PATH}")
        embeddings = np.load(settings.EMBEDDINGS_PATH)
        
        # Check embedding dimensions
        if embeddings.shape[1] != settings.VECTOR_DIMENSION:
            logger.warning(
                f"Embedding dimension in config ({settings.VECTOR_DIMENSION}) "
                f"doesn't match actual dimension ({embeddings.shape[1]})"
            )
        
        # Load CSV in chunks to manage memory
        logger.info(f"Loading CSV data from {settings.CSV_PATH}")
        
        # First, clean the database
        logger.info("Clearing existing data from database")
        await delete_all_books()
        
        # Read CSV in chunks
        chunk_iter = pd.read_csv(settings.CSV_PATH, chunksize=batch_size)
        total_records = 0
        
        for i, chunk in enumerate(chunk_iter):
            start_idx = i * batch_size
            end_idx = min(start_idx + len(chunk), len(embeddings))
            
            # Get corresponding embeddings for this chunk
            chunk_embeddings = embeddings[start_idx:end_idx]
            
            if len(chunk) != len(chunk_embeddings):
                logger.warning(
                    f"Mismatch between chunk size ({len(chunk)}) and "
                    f"embeddings size ({len(chunk_embeddings)})"
                )
                # Adjust to the smaller size
                min_size = min(len(chunk), len(chunk_embeddings))
                chunk = chunk.iloc[:min_size]
                chunk_embeddings = chunk_embeddings[:min_size]
            
            # Process the chunk
            await process_data_chunk(chunk, chunk_embeddings)
            
            total_records += len(chunk)
            logger.info(f"Processed {total_records} records so far")
        
        logger.info(f"Successfully loaded {total_records} records into the database")
        return {"status": "success", "records_loaded": total_records}
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return {"status": "error", "message": str(e)}

async def process_data_chunk(df_chunk, embeddings_chunk):
    """Process a chunk of data and insert into the database."""
    async with get_db_session() as session:
        # Prepare list of book records
        book_records = []
        
        for i, row in df_chunk.iterrows():
            # Convert shelves and rating_distribution to JSON strings if they're not already
            shelves = row["shelves"]
            if not isinstance(shelves, str):
                shelves = json.dumps(shelves)
                
            rating_distribution = row["rating_distribution"]
            if not isinstance(rating_distribution, str):
                rating_distribution = json.dumps(rating_distribution)
            
            # Create book record
            book = Book(
                book_id=int(row["book_id"]),
                title=str(row["title"]),
                author=str(row["author"]),
                author_id=int(row["author_id"]),
                work_id=int(row["work_id"]),
                language=str(row["language"]) if pd.notna(row["language"]) else None,
                average_rating=float(row["average_rating"]),
                ratings_count=int(row["ratings_count"]),
                publication_date=str(row["publication_date"]),
                original_publication_date=str(row["original_publication_date"]) if pd.notna(row["original_publication_date"]) else None,
                format=str(row["format"]) if pd.notna(row["format"]) else None,
                edition_information=str(row["edition_information"]) if pd.notna(row["edition_information"]) else None,
                publisher=str(row["publisher"]) if pd.notna(row["publisher"]) else None,
                num_pages=float(row["num_pages"]) if pd.notna(row["num_pages"]) else None,
                series_name=str(row["series_name"]) if pd.notna(row["series_name"]) else None,
                series_position=str(row["series_position"]) if pd.notna(row["series_position"]) else None,
                description=str(row["description"]),
                image_url=str(row["image_url"]),
                shelves=shelves,
                rating_distribution=rating_distribution,
                embedding=embeddings_chunk[i % len(embeddings_chunk)].tolist()
            )
            book_records.append(book)
        
        # Bulk insert records
        session.add_all(book_records)
        await session.commit()