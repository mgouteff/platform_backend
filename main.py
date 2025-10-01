from fastapi import FastAPI
import routes.query as query   # safer import style for Render
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Accounting Support App",
    description="Ask questions, get answers from uploaded documents",
    version="1.0.0"
)

# Enable CORS (so FlutterFlow can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your FlutterFlow domain in production
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
