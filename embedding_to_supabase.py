from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv
import os
import uuid

# --- Load environment variables ---
load_dotenv()

# --- Setup ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Supabase connection (values come from .env now)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Functions ---
def get_embedding(text: str):
    """Generate embedding from OpenAI for a given text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def store_doc(content: str):
    """Store a document + its embedding in Supabase."""
    embedding = get_embedding(content)
    supabase.table("documents").insert({
        "id": str(uuid.uuid4()),  # auto-generate UUID
        "content": content,
        "embedding": embedding
    }).execute()
    print("✅ Document stored in Supabase!")

# --- Run (example) ---
if __name__ == "__main__":
    sample_text = "This is a test about parental leave in Czech labor law."
    store_doc(sample_text)
