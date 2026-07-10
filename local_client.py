"""
local_client.py
===============

Handles communication with Person 2's local model server.
Optimized for AMD Hackathon: strict timeouts, resilient error handling, and 0-token reporting.
"""

import os
import time
import requests

# Default to localhost, but allow environment variable override for Docker networking flexibility
LOCAL_SERVER_URL = os.getenv(
    "LOCAL_MODEL_URL",
    "http://localhost:8000/generate"
)

# Prevent container timeout
LOCAL_TIMEOUT = int(
    os.getenv(
        "LOCAL_TIMEOUT",
        "120"
    )
)


def ask_local_model(prompt: str) -> dict:
    """
    Sends the prompt to the locally running inference server.
    Returns a dictionary matching the router's expected schema.
    """

    start_time = time.perf_counter()

    payload = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.1
    }

    try:

        response = requests.post(
            LOCAL_SERVER_URL,
            json=payload,
            timeout=LOCAL_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "answer",
            data.get(
                "text",
                data.get(
                    "response",
                    ""
                )
            )
        )

        latency = time.perf_counter() - start_time

        return {
            "success": True,
            "answer": str(answer).strip(),
            "route": "local",
            "latency": latency,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    except requests.exceptions.RequestException as error:

        latency = time.perf_counter() - start_time

        print(f"Local Model API failed: {error}")

        return {
            "success": False,
            "answer": "Error: Local model unavailable or timed out.",
            "route": "local_error",
            "latency": latency,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }