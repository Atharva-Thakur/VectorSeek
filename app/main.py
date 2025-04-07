import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.routes import router
from app.db.operations import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database connection and create tables if needed
    logger.info("Initializing database connection")
    await init_db()
    yield
    logger.info("Shutting down application")

# Create FastAPI app
app = FastAPI(
    title="Book Vector Search API",
    description="API for searching books using vector similarity with pgvector",
    version="1.0.0",
    lifespan=lifespan,
)

# Include API routes
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)