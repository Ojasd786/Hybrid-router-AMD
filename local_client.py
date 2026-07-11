"""
local_client.py
===============

Handles communication with the local model server.
Optimized for AMD Hackathon: strict timeouts, resilient error handling,
answer normalization, deterministic outputs, and 0-token reporting.
"""

import os
import time
import random
import requests
from typing import Dict, Any

# Default to localhost, but allow environment variable override for Docker networking flexibility
LOCAL_SERVER_URL = os.getenv("LOCAL_MODEL_URL", "http://localhost:8000/generate")

# Timeout (seconds) for each HTTP request to the local model
LOCAL_TIMEOUT = float(os.getenv("LOCAL_TIMEOUT", "10"))

# Number of retries for transient local server failures
LOCAL_RETRIES = int(os.getenv("LOCAL_RETRIES", "2"))

# Weak/boilerplate answers that should trigger fallback
WEAK_ANSWERS = {
    "",
    "unknown",
    "i don't know",
    "cannot determine",
    "not sure",
    "no answer",
    "no.",
    "error",
}


def normalize_answer(answer: str) -> str:
    """Clean and normalize local model output."""
    if answer is None:
        return ""
    # Collapse whitespace and trim
    return " ".join(str(answer).split()).strip()


def is_weak_answer(answer: str) -> bool:
    """Detect weak or useless local answers."""
    ans = normalize_answer(answer).lower()
    return len(ans) < 20 or ans in WEAK_ANSWERS


def ask_local_model(prompt: str) -> Dict[str, Any]:
    """
    Sends the prompt to the locally running inference server.
    Returns a dictionary matching the router's expected schema.

    Guarantees:
    - Reports 0 tokens for local calls.
    - Returns 'weak' flag for centralized fallback decision.
    - Uses retries with exponential backoff and jitter.
    """

    start_time = time.perf_counter()

    # Minimal prompt sanitization to avoid accidental huge payloads
    prompt_clean = " ".join(str(prompt).split())
    if len(prompt_clean) > 20000:
        prompt_clean = prompt_clean[:20000]

    payload = {
        "prompt": prompt_clean,
        "max_tokens": int(os.getenv("LOCAL_MAX_TOKENS", "512")),
        "temperature": float(os.getenv("LOCAL_TEMPERATURE", "0.0")),
    }

    last_error = None
    for attempt in range(LOCAL_RETRIES + 1):
        try:
            resp = requests.post(
                LOCAL_SERVER_URL,
                json=payload,
                timeout=LOCAL_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json() if resp.content else {}

            # Support multiple possible response keys
            answer = data.get("answer") or data.get("text") or data.get("response") or ""
            answer = normalize_answer(answer)

            latency = time.perf_counter() - start_time

            return {
                "success": True,
                "answer": answer,
                "route": "local",
                "latency": latency,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "weak": is_weak_answer(answer),
            }

        except requests.exceptions.RequestException as error:
            last_error = error
            # quick exit on DNS/connection refused for first attempt? still backoff and retry a couple times
            backoff = (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(backoff)
            continue
        except ValueError:
            # JSON decode error or unexpected response body
            last_error = "invalid_json"
            break

    latency = time.perf_counter() - start_time
    # Log the last error for debugging (stdout)
    try:
        print(f"⚠️ Local Model API failed after {LOCAL_RETRIES + 1} attempts: {last_error}")
    except Exception:
        pass

    return {
        "success": False,
        "answer": "Error: Local model unavailable or timed out.",
        "route": "local_error",
        "latency": latency,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "weak": True,
    }
