# routes/query.py
from dotenv import load_dotenv

load_dotenv()

from fastapi import APIRouter, HTTPException
from models.query_request import QueryRequest
from services.embeddings import extract_source_ids_from_res, get_embedding, get_ai_response, remove_uuid_line
from services.supabase_client import match_documents, match_knowledge_base
from openai import OpenAI
import os
import json

router = APIRouter()

# Init OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.post("/query", summary="Query Docs (raw chunks)")
async def query_docs(req: QueryRequest):
    """
    Takes a natural language question from the user,
    creates an embedding with OpenAI,
    then queries Supabase for the most relevant document chunks.
    Returns raw matches only.
    """
    try:
        print("🔹 Incoming request:", req.dict())

        print("🔹 Generating embedding...")
        embedding = get_embedding(req.question)
        print(f"✅ Embedding created. First 5 values: {embedding[:5]}")

        print("🔹 Querying Supabase for matches...")
        results = match_knowledge_base(embedding, 15)
        print(f"✅ Supabase returned {len(results)} matches")

        response = get_ai_response(knowledge_base=results, question=req.question)

        used_ids = extract_source_ids_from_res(response.output_text)

        return {
            "text": remove_uuid_line(response.output_text) if used_ids else response.output_text,
            "sources": [{"id": r.get("id"), "title": r.get("title")} for r in results if r.get("id") in used_ids],
        }

    except Exception as e:
        print("❌ ERROR in query_docs:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", summary="Ask GPT with context")
async def ask_gpt(req: QueryRequest):
    """
    Takes a natural language question,
    fetches relevant document chunks from Supabase,
    then sends them as context to GPT for a polished answer.
    """
    try:
        print("🔹 Incoming request:", req.dict())

        # Step 1: Create embedding for the question
        print("🔹 Generating embedding...")
        embedding = get_embedding(req.question)

        # Step 2: Query Supabase with embedding
        print("🔹 Querying Supabase for matches...")
        results = match_documents(embedding, req.top_k)

        # Step 3: Build GPT prompt with context
        context = "\n\n".join([r.get("content", "") for r in results])
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
        ]

        # Step 4: Call GPT
        print("🔹 Calling OpenAI GPT...")
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3
        )

        answer = completion.choices[0].message.content
        print("✅ GPT response generated")

        # Step 5: Return structured response
        return {
            "question": req.question,
            "answer": answer,
            "sources": results  # keep the original chunks for traceability
        }

    except Exception as e:
        print("❌ ERROR in ask_gpt:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

