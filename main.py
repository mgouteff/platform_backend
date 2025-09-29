from fastapi import FastAPI
from routes import query   # import your query router from routes/query.py

# Create FastAPI app
app = FastAPI(
    title="Accounting Support App",
    description="Ask questions, get answers from uploaded documents",
    version="1.0.0"
)

# Register routes
app.include_router(query.router)

# Root endpoint (just to test if server is running)
@app.get("/")
async def root():
    return {"message": "FastAPI is running! Go to /docs for Swagger UI."}

