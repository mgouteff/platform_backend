import os
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str):
    """Generate embedding for a given text."""
    resp = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding


def seed_embeddings(batch_size=50):
    rows = supabase.table("knowledge_base").select("id, chunk_text").is_("embedding", None).execute()

    if not rows.data:
        print("No rows left without embeddings.")
        return

    print(f"Found {len(rows.data)} rows to update...")

    for i, row in enumerate(rows.data):
        embedding = get_embedding(row["chunk_text"])

        supabase.table("knowledge_base").update({"embedding": embedding}).eq("id", row["id"]).execute()

        print(f"Updated row {i+1}/{len(rows.data)}: {row['id']}")



if __name__ == "__main__":
    seed_embeddings()
