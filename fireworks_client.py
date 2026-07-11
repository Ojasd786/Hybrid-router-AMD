"""
fireworks_client.py
===================

Handles all Fireworks API communication.
Optimized for token efficiency, lazy-loading, retries,
answer normalization, and weak-answer detection.
"""

import time
import random
from openai import OpenAI
from config import settings

class FireworksClientError(Exception):
    """Raised when Fireworks inference fails after retries."""
    pass


# Weak/boilerplate answers that should be flagged
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
    """Clean and normalize Fireworks output."""
    return str(answer).strip()


def is_weak_answer(answer: str) -> bool:
    """Detect weak or useless Fireworks answers."""
    ans = normalize_answer(answer).lower()
    return len(ans) < 20 or ans in WEAK_ANSWERS


class FireworksClient:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization ensures env vars are loaded before connection."""
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.fireworks_api_key,
                base_url=settings.fireworks_base_url
            )
        return self._client

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,    # deterministic, concise outputs
        max_tokens: int = 1024,      # reduced to limit token burn
        retries: int = 5
    ) -> dict:

        if not getattr(settings, "allowed_models", None):
            raise ValueError("No allowed models configured in settings.allowed_models.")
        if model not in settings.allowed_models:
            raise ValueError(f"'{model}' is not present in ALLOWED_MODELS.")

        last_error = None

        for attempt in range(retries):
            try:
                start_time = time.perf_counter()

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise AI assistant. Answer accurately and concisely. Prefer one-sentence factual answers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=["\n\n"]
                )

                latency = time.perf_counter() - start_time
                usage = getattr(response, "usage", None)

                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)

                # Safely extract answer and model name
                answer = ""
                try:
                    if getattr(response, "choices", None):
                        choice0 = response.choices[0]
                        answer = getattr(getattr(choice0, "message", None), "content", None) or getattr(choice0, "text", "") or ""
                except Exception:
                    answer = ""

                answer = normalize_answer(answer)
                model_name = getattr(response, "model", model)

                return {
                    "success": True,
                    "answer": answer,
                    "model": model_name,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "latency": latency,
                    "weak": is_weak_answer(answer),
                    "retries_used": attempt,
                }

            except Exception as e:
                last_error = e
                # Exponential backoff with jitter to handle rate limits gracefully
                backoff = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(backoff)

        # After retries exhausted
        err_msg = f"Fireworks API failed after {retries} attempts: {str(last_error)}"
        return {
            "success": False,
            "error": err_msg,
            "model": model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency": 0.0,
            "weak": True,
            "retries_used": retries,
        }


# Global singleton
fireworks_client = FireworksClient()
