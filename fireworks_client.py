"""
fireworks_client.py

Handles all Fireworks API communication.
Token efficiency, lazy-loading, and retries.
"""

import time
from openai import OpenAI
from config import settings

class FireworksClientError(Exception):
    """Raised when Fireworks inference fails after retries."""
    pass


class FireworksClient:
    def __init__(self):
        # Do not initialize the client here to prevent startup crashes.
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
        temperature: float = 0.1, # Lower temp = more concise/deterministic answers
        max_tokens: int = 2048,
        retries: int = 3
    ) -> dict:

        # Ensure router is using only allowed models
        if model not in settings.allowed_models:
            raise ValueError(f"'{model}' is not present in ALLOWED_MODELS.")

        last_error = None

        for attempt in range(retries):
            try:
                start_time = time.perf_counter()

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a highly efficient assistant. Provide direct, factual, and concise answers. Do not use conversational filler, pleasantries, or preamble."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                latency = time.perf_counter() - start_time
                usage = getattr(response, "usage", None)

                # Safely extract tokens
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)
                
                # Safely extract answer
                answer = response.choices[0].message.content if response.choices else ""

                return {
                    "success": True,
                    "answer": answer.strip(),
                    "model": response.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "latency": latency
                }

            except Exception as e:
                last_error = e
                # Network failed, wait before retrying (exponential backoff)
                time.sleep(1.0 * (attempt + 1))

        # If all retries fail, return a structured error dictionary
        return {
            "success": False,
            "error": f"Fireworks API failed after {retries} attempts: {str(last_error)}"
        }

# Global singleton
fireworks_client = FireworksClient()