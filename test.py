from openai import OpenAI
import os
import sys
from env_loader import load_env

load_env()

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not api_key:
    print("[ERROR] Please set OPENAI_API_KEY in .env before running test.py")
    sys.exit(1)

client_kwargs = {"api_key": api_key}
if base_url:
    client_kwargs["base_url"] = base_url

client = OpenAI(**client_kwargs)

print(f"Testing OpenAI API connection with model: {model}...")
try:
    # First, let's list available models on your API key
    available = [m.id for m in client.models.list().data if "whisper" not in m.id and "guard" not in m.id]
    print(f"Connected to Groq! Available chat models: {available[:4]}")
    if model not in available and available:
        model = available[0]
        print(f"Using active model: {model}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Explain what an API is in simple terms."}
        ],
        max_tokens=150
    )
    print("\n[SUCCESS] Response received from Groq/OpenAI:")
    print("-" * 60)
    print(response.choices[0].message.content)
    print("-" * 60)
except Exception as e:
    import traceback
    print(f"\n[FAILED] Error: {e}")
    if hasattr(e, "__cause__") and e.__cause__:
        print(f"Cause: {e.__cause__}")
    traceback.print_exc()
