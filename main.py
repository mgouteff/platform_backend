from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import query   # import your query router from routes/query.py

# Create FastAPI app
app = FastAPI(
    title="Accounting Support App",
    description="Ask questions, get answers from uploaded documents",
    version="1.0.0"
)

# Allowed origins for FlutterFlow and local dev
origins = [
    "https://yourapp.flutterflow.app",   # replace with your actual FlutterFlow app URL
    "http://localhost:8080",             # optional: local dev in FlutterFlow
    "http://localhost",                  # optional: plain localhost
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # only these domains can call your API
    allow_credentials=True,       # allow cookies/authorization
    allow_methods=["*"],          # allow all HTTP methods
    allow_headers=["*"],          # allow all headers
)

# Register routes
app.include_router(query.router)

# Root endpoint (just to test if server is running)
@app.get("/")
async def root():
    return {"message": "FastAPI is running! Go to /docs for Swagger UI."}


