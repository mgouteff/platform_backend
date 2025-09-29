import os
import requests

# Make sure your API key is set in the environment first:
#   setx OPENAI_API_KEY "sk-your-real-key-here"   (Windows, permanent)
#   set OPENAI_API_KEY=sk-your-real-key-here      (Windows, one session)
#   export OPENAI_API_KEY=sk-your-real-key-here   (Mac/Linux)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ No OPENAI_API_KEY found in environment variables.")
    exit(1)

print("✅ Found API key (hidden for security). Testing connection...")

url = "https://api.openai.com/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}

try:
    response = requests.get(url, headers=headers, timeout=10)
    print("Status code:", response.status_code)

    if response.status_code == 200:
        print("✅ Connection successful!")
        models = response.json().get("data", [])
        print(f"Retrieved {len(models)} models. Example: {models[0]['id']}")
    else:
        print("❌ Error response from API:")
        print(response.text)

except requests.exceptions.RequestException as e:
    print("❌ Connection error:", e)
