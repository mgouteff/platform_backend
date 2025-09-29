from openai import OpenAI
from supabase import create_client
import os

# --- Setup ---
# OpenAI client (uses your API key from environment variable)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Supabase connection
SUPABASE_URL = "https://lwvntviqpkfogdgmtnvp.supabase.co"
# Service role key from Supabase dashboard (starts with eyJ…)
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx3dm50dmlxcGtmb2dkZ210bnZwIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1ODgxOTIzMiwiZXhwIjoyMDc0Mzk1MjMyfQ.6A2VFHCfXtW68D4IDg0fxD6OTUjRpuzJSnq0pZPsqKQ"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Functions ---
def get_embedding(text: str):
    """Generate embedding from OpenAI for a given text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_docs(query: str, threshold=0.7, count=5):
    """Search Supabase for documents matching the query embedding."""
    query_embedding = get_embedding(query)
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": count
        }
    ).execute()
    return result.data

# --- Test run ---
if __name__ == "__main__":
    query = "example search text"
    results = search_docs(query)

    if results:
        print("Results:", results)
    else:
        print("No relevant matches found in your knowledge base.")
