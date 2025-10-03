# routes/query.py
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

load_dotenv()

from fastapi import APIRouter, HTTPException
from models.query_request import QueryRequest
from services.embeddings import expand_user_query, extract_source_ids_from_res, get_embedding, get_ai_response, remove_uuid_line, stream_openai_response
from services.supabase_client import match_documents, match_knowledge_base
from openai import OpenAI
import os
import json
import asyncio

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
        expanded_q = expand_user_query(req.question)

        print("🔹 Generating embedding...")
        embedding = get_embedding(expanded_q)
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


@router.post("/stream", summary="Ask GPT with context. Streamed response")
async def stream(req: QueryRequest):
    try:
        async def event_stream():
            try:
                yield f"data: {json.dumps({'status': 'Analyzing users question...'})}\n\n"
                await asyncio.sleep(0)

                expanded_q = expand_user_query(req.question)

                yield f"data: {json.dumps({'status': 'Preparing search query...'})}\n\n"
                await asyncio.sleep(0)

                #generate vectors from users question. knowledge_base - larger model(3072 vector size), documents - smaller model(1536 vector size)
                # embedding = get_embedding(expanded_q, model="text-embedding-3-large") # for knowledge_base table larger model
                embedding = get_embedding(expanded_q, model="text-embedding-3-small") # for documents table smaller model

                yield f"data: {json.dumps({'status': 'Searching relevant sources...'})}\n\n"
                await asyncio.sleep(0)

                # results = match_knowledge_base(embedding, req.top_k) # vector search in knowledge_base table
                results = match_documents(embedding, req.top_k, threshold=0.1) # vector search in documents table
                print(f"✅ Supabase returned {len(results)} matches")


                yield f"data: {json.dumps({'status': 'Analyzing sources...'})}\n\n"
                await asyncio.sleep(0)

                # for final chat gpt response changing sources table does not change anything
                async for chunk in stream_openai_response(results, req.question):
                    yield chunk
            except Exception as e:
                yield f"data: {json.dumps({'error': 'Something went wrong', 'exception': str(e)})}"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Something went wrong")

