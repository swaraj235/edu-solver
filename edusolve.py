from openai import OpenAI
import os
import sys
import time
from env_loader import load_env

# 1. Load environment variables from .env
load_env()

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")  # e.g., for GitHub Models or Groq
model_name = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")

if not api_key:
    print("\n" + "!" * 70)
    print("ERROR: OpenAI API Key not found!")
    print("Please set OPENAI_API_KEY in your '.env' file.")
    print("Tip: If you don't have a paid OpenAI key, see .env.example for")
    print("free options like Hugging Face (edusolve_hf.py) or GitHub Models.")
    print("!" * 70 + "\n")
    sys.exit(1)

# Initialize OpenAI client (supports official OpenAI, GitHub Models, or Groq)
client_kwargs = {"api_key": api_key}
if base_url:
    client_kwargs["base_url"] = base_url

client = OpenAI(**client_kwargs)

# 2. Get assignment question from user
default_question = "Explain the difference between supervised and unsupervised machine learning."
print(f"Default Question: {default_question}")
user_input = input("Enter the assignment question (or press Enter to use default):\n").strip()
assignment = user_input if user_input else default_question

# 3. Structure Prompt Engineering roles (System vs User)
system_instructions = (
    "You are an academic assistant working for EduSolve AI.\n"
    "Solve assignment questions clearly and accurately with these requirements:\n"
    "1. Explain concepts in simple language.\n"
    "2. Use appropriate headings.\n"
    "3. Include real-world examples wherever appropriate.\n"
    "4. Keep the answer strictly relevant to the question.\n"
    "5. Structure the answer clearly with markdown formatting."
)

user_prompt = f"Assignment Question:\n{assignment}"

print(f"\n[EduSolve AI] Sending prompt to API ({model_name})...\n")

start_time = time.time()

try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,  # Deterministic, academic output
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    answer = response.choices[0].message.content
    usage = getattr(response, "usage", None)

    print("=" * 70)
    print("EDUSOLVE AI - OPENAI API GENERATED SOLUTION")
    print("=" * 70)
    print(answer)
    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print(f"API Response Time: {elapsed_time:.2f} seconds")
    if usage:
        print(f"Prompt Tokens: {usage.prompt_tokens} | Completion Tokens: {usage.completion_tokens} | Total: {usage.total_tokens}")
    print("=" * 70)

except Exception as e:
    print(f"\nAPI Error: {e}")
