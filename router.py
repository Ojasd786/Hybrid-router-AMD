"""
router.py
=========

Smart complexity-based task router for AMD Developer Hackathon.

Responsibilities:
- Classify tasks into the official 8 NLP capability domains without using an LLM.
- Calculate prompt complexity, constraints, and length.
- Decide between Local Model (0 tokens) vs Fireworks API.
- Execute inference using local_client or fireworks_client without breaking API contracts.
- Track metrics using stats.py.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from config import settings
from fireworks_client import fireworks_client, FireworksClientError
from local_client import ask_local_model
from stats import stats


class TaskRouter:
    """Intelligent router optimizing accuracy vs token cost."""

    def __init__(self) -> None:
        self.code_gen_patterns = re.compile(
            r"\b(write|create|implement|generate|construct)\s+(a\s+)?(python|js|cpp|java|function|script|class|code)\b",
            re.IGNORECASE,
        )

        self.code_debug_patterns = re.compile(
            r"\b(debug|fix|bug|correct|error|find\s+the\s+bug|syntax|typeerror|valueerror|exception)\b",
            re.IGNORECASE,
        )

        self.logic_patterns = re.compile(
            r"\b(puzzle|riddle|deduce|deduction|all\s+conditions|must\s+be\s+satisfied|who\s+owns|sits\s+next|truth|liar|constraint)\b",
            re.IGNORECASE,
        )

        self.math_patterns = re.compile(
            r"(\b(calculate|compute|solve|equation|percentage|interest|ratio|probability)\b|\d+\s*[\+\-\*\/\%]\s*\d+|[\$€£]\d+)",
            re.IGNORECASE,
        )

        self.ner_patterns = re.compile(
            r"\b(extract|identify|named\s+entities|entity|entities|person|organization|location|date)\b",
            re.IGNORECASE,
        )

        self.sentiment_patterns = re.compile(
            r"\b(sentiment|classify|review|positive|negative|neutral|attitude|opinion)\b",
            re.IGNORECASE,
        )

        self.summarize_patterns = re.compile(
            r"\b(summarize|summary|condense|abstract|brief|in\s+one\s+sentence|shorten)\b",
            re.IGNORECASE,
        )

    def classify_domain(self, prompt: str) -> str:
        if self.code_debug_patterns.search(prompt):
            return "code_debugging"

        if self.code_gen_patterns.search(prompt) or "def " in prompt:
            return "code_generation"

        if self.logic_patterns.search(prompt):
            return "logical_reasoning"

        if self.math_patterns.search(prompt):
            return "mathematical_reasoning"

        if self.ner_patterns.search(prompt):
            return "ner"

        if self.sentiment_patterns.search(prompt):
            return "sentiment"

        if self.summarize_patterns.search(prompt):
            return "summarization"

        return "factual_knowledge"

    def estimate_complexity(self, prompt: str, domain: str) -> float:
        score = 0.0

        word_count = len(prompt.split())

        if word_count > 150:
            score += 0.3
        elif word_count > 70:
            score += 0.15

        domain_weights = {
            "code_generation": 0.6,
            "code_debugging": 0.6,
            "logical_reasoning": 0.5,
            "mathematical_reasoning": 0.4,
            "summarization": 0.2,
            "factual_knowledge": 0.2,
            "ner": 0.1,
            "sentiment": 0.1,
        }

        score += domain_weights.get(domain, 0.2)

        constraint_triggers = [
            "if and only if",
            "exactly",
            "must contain",
            "step by step",
            "handling duplicates",
            "edge cases",
            "justify",
        ]

        prompt_lower = prompt.lower()

        for trigger in constraint_triggers:
            if trigger in prompt_lower:
                score += 0.15

        return min(score, 1.0)

    def select_fireworks_model(self, domain: str, complexity: float) -> str:
        """
        Select the best Fireworks model based on task complexity.
        """

        if not settings.allowed_models:
            return "default"

        # Stronger model for difficult tasks
        if (
            domain in (
                "code_generation",
                "code_debugging",
                "logical_reasoning",
            )
            or complexity >= 0.7
        ):
            return settings.allowed_models[0]

        # Smaller / cheaper model for easier tasks
        if len(settings.allowed_models) > 1:
            return settings.allowed_models[1]

        return settings.allowed_models[0]

    def route(self, prompt: str) -> Dict[str, Any]:

        domain = self.classify_domain(prompt)
        complexity = self.estimate_complexity(prompt, domain)

        use_fireworks = (
            complexity >= 0.5
            or domain in (
                "code_generation",
                "code_debugging",
                "logical_reasoning",
                "mathematical_reasoning",
            )
        )

        selected_model = self.select_fireworks_model(
            domain,
            complexity,
        )

        if use_fireworks:

            try:

                fw_response = fireworks_client.generate(
                    prompt=prompt,
                    model=selected_model,
                )

                if not fw_response.get("success", False):
                    raise FireworksClientError(
                        fw_response.get("error", "Unknown Fireworks error")
                    )

                stats.record_fireworks(
                    prompt_tokens=fw_response.get("prompt_tokens", 0),
                    completion_tokens=fw_response.get("completion_tokens", 0),
                    latency=fw_response.get("latency", 0.0),
                    route=f"fireworks:{fw_response.get('model', 'unknown')}",
                )

                return {
                    "success": True,
                    "answer": fw_response.get("answer", ""),
                    "route": f"fireworks:{fw_response.get('model', 'unknown')}",
                    "prompt_tokens": fw_response.get("prompt_tokens", 0),
                    "completion_tokens": fw_response.get("completion_tokens", 0),
                    "total_tokens": fw_response.get("total_tokens", 0),
                    "latency": fw_response.get("latency", 0.0),
                }

            except FireworksClientError as e:

                print(
                    f"⚠️ Fireworks API failed: {e}. Falling back to local model."
                )

        local_response = ask_local_model(prompt)

        stats.record_local(
            latency=local_response.get("latency", 0.0),
            route="local",
        )

        return local_response


router = TaskRouter()
