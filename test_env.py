# test_env.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("SUPABASE_URL:", os.getenv("SUPABASE_URL"))
print("SUPABASE_KEY:", os.getenv("SUPABASE_KEY")[:6] + "...")  # print only first chars for safety
