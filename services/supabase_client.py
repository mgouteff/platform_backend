# services/supabase_client.py

import os
from supabase import create_client, Client

# Load Supabase environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Missing Supabase credentials in environment variables")
else:
    print("✅ DEBUG: Supabase URL:", SUPABASE_URL)
    print("✅ DEBUG: Supabase Key starts with:", SUPABASE_KEY[:6])

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def match_documents(query_embedding, top_k: int = 3, threshold=0.4):
    """
    Calls the 'match_documents' Postgres function in Supabase to find similar chunks.
    """
    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "match_threshold": threshold # test with other values in future, when there will be more data in db
        }
    ).execute()

    if hasattr(response, "data"):
        return response.data
    else:
        print("❌ ERROR: Supabase response did not contain 'data'. Full response:", response)
        return []

def match_knowledge_base(embedding, limit):
    response = supabase.rpc(
        "match_knowledge_base",
        {
            "query_embedding": embedding,
            "match_count": limit
        }
    ).execute()

    return response.data
