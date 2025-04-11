from fastapi import FastAPI
from api import router

app = FastAPI(
    title="Book Vector Search API",
    description="API for searching books using semantic vector similarity",
    version="1.0.0"
)

app.include_router(router)