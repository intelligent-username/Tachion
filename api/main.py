"""
Tachion API Server

Run with: uvicorn api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.predict import router as predict_router

app = FastAPI(
    title="Tachion API",
    description="Market forecasting API",
    version="1.0.0"
)

# CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Tachion API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
