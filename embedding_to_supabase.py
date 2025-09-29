from openai import OpenAI
from supabase import create_client
import os
import uuid

# --- Setup ---
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Replace with your project URL + service role key
SUPABASE_URL = "https://lwvntviqpkfogdgmtnvp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx3dm50dmlxcGtmb2dkZ210bnZwIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1ODgxOTIzMiwiZXhwIjoyMDc0Mzk1MjMyfQ.6A2VFHCfXtW68D4IDg0fxD6OTUjRpuzJSnq0pZPsqKQ"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Functions ---
def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def store_doc(content: str):
    embedding = get_embedding(content)
    supabase.table("documents").insert({
        "id": str(uuid.uuid4()),   # auto-generate UUID
        "content": content,
        "embedding": embedding
    }).execute()
    print("✅ Document stored in Supabase!")

# --- Run ---
if __name__ == "__main__":
    sample_text = "This is a test about parental leave in Czech labor law."
    store_doc(sample_text)
