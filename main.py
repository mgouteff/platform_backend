from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import query   # import your query router from routes/query.py

# Create FastAPI app
app = FastAPI(
    title="Accounting Support App",
    description="Ask questions, get answers from uploaded documents",
    version="1.0.0"
)

# ✅ Add CORS middleware right after creating the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your FlutterFlow domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(query.router)

# Root endpoint (just to test if server is running)
@app.get("/")
async def root():
    return {"message": "FastAPI is running! Go to /docs for Swagger UI."}

