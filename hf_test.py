from huggingface_hub import InferenceClient
import os
import sys
from env_loader import load_env

load_env()

token = os.getenv("HF_TOKEN")
model = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

if not token or "your_" in token:
    print("[ERROR] Please set your free HF_TOKEN in .env (get it from https://huggingface.co/settings/tokens)")
    sys.exit(1)

# Use HF's own free serverless infrastructure (hf-inference) instead of paid third-party router providers
client = InferenceClient(provider="hf-inference", api_key=token)

print(f"Testing Hugging Face connection with model: {model}...")
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Explain what an API is in simple terms."}
        ],
        max_tokens=150
    )
    print("\n[SUCCESS] Response received:")
    print(response.choices[0].message.content)
except Exception as e:
    import traceback
    print(f"\n[FAILED] Error: {e}")
    if hasattr(e, "__cause__") and e.__cause__:
        print(f"Cause: {e.__cause__}")
    traceback.print_exc()
