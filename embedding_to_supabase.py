from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv
import os
import uuid

# --- Load environment variables ---
load_dotenv()

# --- Setup ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Functions ---
def get_embedding(text: str):
    """Generate an embedding from OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def store_doc(content: str):
    """Store a document + its embedding in Supabase."""
    embedding = get_embedding(content)
    supabase.table("documents").insert({
        "id": str(uuid.uuid4()),
        "title": "Sample Test Doc",
        "source_file": "manual_test",
        "version": "1.0",
        "doc_type": "internal",
        "uploaded_at": "now()",
        # embedding not stored since no embedding column in your schema
    }).execute()
    print("✅ Document stored in Supabase!")

# --- Sanity Check Block ---
if __name__ == "__main__":
    print("⚡ Running Supabase sanity check...")

    try:
        response = supabase.table("documents").select("*").limit(1).execute()
        print("✅ Supabase connection successful!")
        print("Response:", response.data)
    except Exception as e:
        print("❌ Supabase connection failed!")
        print("Error:", e)

