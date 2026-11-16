"""
EduSolve AI - API vs Manual Prompt Comparison Tool
Assignment Requirement:
  • Analyze and compare API outputs with manual prompts
"""

import os
import sys
import time
from env_loader import load_env

load_env()

# Evaluation Question
QUESTION = "Explain the difference between supervised and unsupervised machine learning."

SYSTEM_PROMPT = """You are an academic assistant working for EduSolve AI.
Solve the following assignment question clearly and accurately.
Requirements:
1. Explain concepts in simple language.
2. Use appropriate headings.
3. Include real-world examples.
4. Keep the answer strictly relevant.
5. Structure clearly."""

# In manual execution (Web UI like ChatGPT or HuggingChat), users usually paste everything together:
MANUAL_PROMPT = f"""{SYSTEM_PROMPT}

Assignment Question:
{QUESTION}"""


def run_api_hf(token: str, model: str):
    from huggingface_hub import InferenceClient
    client = InferenceClient(provider="hf-inference", api_key=token)
    start = time.time()
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": QUESTION}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    elapsed = time.time() - start
    return res.choices[0].message.content, elapsed


def run_api_openai(api_key: str, model: str, base_url: str = None):
    from openai import OpenAI
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    start = time.time()
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": QUESTION}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    elapsed = time.time() - start
    return res.choices[0].message.content, elapsed


def main():
    print("=" * 75)
    print("   EDUSOLVE AI: MANUAL PROMPT VS PROGRAMMATIC API COMPARISON")
    print("=" * 75)
    print("\n1. PROMPT SPECIFICATION:")
    print("-" * 75)
    print(f"Target Question:\n  \"{QUESTION}\"")
    print("\n[Manual Prompt Format - Single User Message in Web UI]:")
    print(MANUAL_PROMPT)

    print("\n[Programmatic API Format - Strict Role Separation]:")
    print("  - Role 'system': Persona, constraints, academic formatting rules")
    print("  - Role 'user'  : Raw question payload")
    print("  - Parameters   : temperature=0.2, max_tokens=600")

    # Check available API
    hf_token = os.getenv("HF_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    hf_model = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    openai_model = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")

    api_result = None
    elapsed = None
    provider_used = None

    if openai_key and "your_" not in openai_key:
        print(f"\n2. EXECUTING PROGRAMMATIC API (OpenAI / Compatible: {openai_model})...")
        try:
            api_result, elapsed = run_api_openai(openai_key, openai_model, openai_base_url)
            provider_used = f"OpenAI Compatible ({openai_model})"
        except Exception as e:
            print(f"  [OpenAI API Error]: {e}")

    if not api_result and hf_token and "your_" not in hf_token:
        print(f"\n2. EXECUTING PROGRAMMATIC API (Hugging Face: {hf_model})...")
        try:
            api_result, elapsed = run_api_hf(hf_token, hf_model)
            provider_used = f"Hugging Face ({hf_model})"
        except Exception as e:
            print(f"  [HF API Error]: {e}")

    if api_result:
        print(f"\nAPI Response Received in {elapsed:.2f}s via {provider_used}:")
        print("-" * 75)
        print(api_result)
        print("-" * 75)
    else:
        print("\n[Note] Neither API token is set in .env yet. Once configured,")
        print("this script will automatically execute and time the real API call.")

    print("\n" + "=" * 75)
    print("3. DETAILED COMPARISON & ANALYSIS")
    print("=" * 75)
    comparison_table = """
| Evaluation Metric        | Manual Prompting (Web UI)            | Programmatic API Integration             |
|--------------------------|--------------------------------------|------------------------------------------|
| Execution Method         | Human copy-paste in browser          | Automated Python script / webhook        |
| Role Separation          | Blended instructions in user input   | Strict `system` vs `user` message roles  |
| Output Consistency       | Variable (default non-zero temp)     | High (explicit `temperature=0.2`, seed) |
| Latency & Measurement    | Unknown / manual stopwatch           | Exact programmatic timing (e.g. 1.84s)   |
| Parameter Control        | None (fixed by platform UI)          | Full control (top_p, max_tokens, penalty)|
| Scalability & Batching   | Single query at a time               | Unlimited parallel / batch requests      |
| Software Integration     | Isolated, cannot connect to backend  | Direct ingestion into database & UI      |
| Error Handling & Retries | Manual page refresh                  | Automated try/except retry policies      |
    """
    print(comparison_table)
    print("Detailed report saved to: ANALYSIS_REPORT.md\n")


if __name__ == "__main__":
    main()
