from fastapi import FastAPI
from app.api import api_router

app = FastAPI(title="TriMem Agent API - Scalable Architecture")

# Include the refactored routes
app.include_router(api_router)