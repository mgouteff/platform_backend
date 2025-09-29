# services/embeddings.py

import os
from openai import OpenAI

# Load API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in environment variables")
else:
    print("✅ DEBUG: OpenAI API Key starts with:", OPENAI_API_KEY[:8])

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to get embeddings for a given text
def get_embedding(text: str):
    """
    Generate an embedding vector for the given input text using OpenAI embeddings API.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",  # or text-embedding-3-large
        input=text
    )
    return response.data[0].embedding
