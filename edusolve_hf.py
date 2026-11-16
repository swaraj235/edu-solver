from huggingface_hub import InferenceClient
import os
import sys
import time
from env_loader import load_env

# 1. Load environment variables from .env
load_env()

hf_token = os.getenv("HF_TOKEN")
model_name = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

if not hf_token or "your_" in hf_token:
    print("\n" + "!" * 70)
    print("ERROR: Hugging Face Token (HF_TOKEN) not configured in '.env'!")
    print("\nHow to get a 100% FREE token in 1 minute:")
    print("  1. Create a free account at https://huggingface.co")
    print("  2. Go to: https://huggingface.co/settings/tokens")
    print("  3. Click 'Create new token' (Select Type: 'Read')")
    print("  4. Paste the token into your .env file: HF_TOKEN=hf_...")
    print("!" * 70 + "\n")
    sys.exit(1)

# Initialize Hugging Face Inference Client on free serverless tier (hf-inference)
client = InferenceClient(provider="hf-inference", api_key=hf_token)

# 2. Get assignment question from user
default_question = "Explain the difference between supervised and unsupervised machine learning."
print(f"Default Question: {default_question}")
user_input = input("Enter the assignment question (or press Enter to use default):\n").strip()
assignment = user_input if user_input else default_question

# 3. Structure prompt engineering roles (System Persona vs User Request)
system_instructions = (
    "You are an academic assistant working for EduSolve AI.\n"
    "Solve the following assignment question clearly and accurately.\n"
    "Requirements:\n"
    "1. Explain concepts in simple language.\n"
    "2. Use appropriate headings.\n"
    "3. Include real-world examples wherever appropriate.\n"
    "4. Keep the answer strictly relevant to the question.\n"
    "5. Structure the answer clearly.\n"
    "6. Provide technically accurate information."
)

user_prompt = f"Assignment Question:\n{assignment}"

print(f"\n[EduSolve AI] Sending prompt to Hugging Face API ({model_name})...\n")

start_time = time.time()

try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=800,
        temperature=0.3
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    answer = response.choices[0].message.content

    print("=" * 70)
    print("EDUSOLVE AI - HUGGING FACE GENERATED SOLUTION")
    print("=" * 70)
    print(answer)
    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print(f"API Response Time: {elapsed_time:.2f} seconds")
    print("=" * 70)

except Exception as e:
    print(f"\nAPI Execution Error: {e}")
    print("\nTip: If this model is busy or unavailable, try another free serverless model in .env:")
    print("  - Qwen/Qwen2.5-7B-Instruct")
    print("  - mistralai/Mistral-7B-Instruct-v0.3")
    print("  - HuggingFaceH4/zephyr-7b-beta")
