from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import your routers or modules
from query import router as query_router

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Create FastAPI app
app = FastAPI()

# CORS setup
origins = [
    "https://knowledge-platform-nb1egs.flutterflow.app",  # your FlutterFlow app
    "http://localhost:3000",  # local testing (optional)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # allows POST + handles OPTIONS preflight
    allow_headers=["*"],   # allows any headers
)

# Include routers
app.include_router(query_router)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Backend is running"}



