from fastapi import FastAPI
from .api import router
from .ingest import ingest
import threading
import os

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
def on_startup():
    if os.getenv("RUN_INGEST", "false").lower() == "true":
        threading.Thread(target=ingest).start()