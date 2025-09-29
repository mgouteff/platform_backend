# routes/query.py

from fastapi import APIRouter, HTTPException
from models.query_request import QueryRequest
from services.embeddings import get_embedding
from services.supabase_client import match_documents

router = APIRouter()

@router.post("/query", summary="Query Docs")
async def query_docs(req: QueryRequest):
    """
    Takes a natural language question from the user,
    creates an embedding with OpenAI,
    then queries Supabase for the most relevant document chunks.
    """

    try:
        print("🔹 Incoming request:", req.dict())

        # Step 1: Create embedding for the question
        print("🔹 Generating embedding...")
        embedding = get_embedding(req.question)
        print(f"✅ Embedding created. First 5 values: {embedding[:5]}")

        # Step 2: Query Supabase with embedding
        print("🔹 Querying Supabase for matches...")
        results = match_documents(embedding, req.top_k)
        print(f"✅ Supabase returned {len(results)} matches")

        return {
            "question": req.question,
            "results": results
        }

    except Exception as e:
        print("❌ ERROR in query_docs:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
