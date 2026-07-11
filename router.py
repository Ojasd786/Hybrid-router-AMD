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


def normalize_output(answer: str) -> str:
    """Minimal normalization: trim whitespace only."""
    return answer.strip() if answer else ""


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
            r"\b(puzzle|riddle|deduce|deduction|truth\s+table|constraint\s+satisfaction|all\s+conditions|must\s+be\s+satisfied|who\s+owns|sits\s+next|truth|liar|constraint|logic)\b",
            re.IGNORECASE,
        )
        self.math_patterns = re.compile(
            r"(\b(calculate|compute|solve|equation|percentage|interest|ratio|probability|statistics|mean|variance|average|integral|derivative|algebra|geometry|calculus|matrix)\b|\d+\s*[\+\-\*\/\%]\s*\d+|[\$€£]\d+)",
            re.IGNORECASE,
        )
        self.ner_patterns = re.compile(
            r"\b(extract|identify|named\s+entities|entity|entities|person|organization|location|date)\b",
            re.IGNORECASE,
        )
        self.sentiment_patterns = re.compile(
            r"\b(sentiment|classify|review|positive|negative|neutral|attitude|opinion|emotion|tone)\b",
            re.IGNORECASE,
        )
        self.summarize_patterns = re.compile(
            r"\b(summarize|summary|condense|abstract|brief|in\s+one\s+sentence|shorten|overview|digest|tl;dr)\b",
            re.IGNORECASE,
        )

    def classify_domain(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if self.code_debug_patterns.search(prompt_lower):
            return "code_debugging"
        if self.code_gen_patterns.search(prompt_lower) or "def " in prompt_lower or "function" in prompt_lower:
            return "code_generation"
        if self.logic_patterns.search(prompt_lower) or "sudoku" in prompt_lower:
            return "logical_reasoning"
        if self.math_patterns.search(prompt_lower):
            return "mathematical_reasoning"
        if self.ner_patterns.search(prompt_lower):
            return "ner"
        if self.sentiment_patterns.search(prompt_lower):
            return "sentiment"
        if self.summarize_patterns.search(prompt_lower):
            return "summarization"
        return "factual_knowledge"

    def estimate_complexity(self, prompt: str, domain: str) -> float:
        score = 0.0
        word_count = len(prompt.split())
        if word_count > 150:
            score += 0.5
        elif word_count > 70:
            score += 0.3
        elif word_count > 30:
            score += 0.15
        domain_weights = {
            "code_generation": 0.8,
            "code_debugging": 0.8,
            "logical_reasoning": 0.7,
            "mathematical_reasoning": 0.7,
            "summarization": 0.5,
            "factual_knowledge": 0.3,
            "ner": 0.3,
            "sentiment": 0.3,
        }
        score += domain_weights.get(domain, 0.3)
        constraint_triggers = [
            "if and only if",
            "exactly",
            "must contain",
            "step by step",
            "handling duplicates",
            "edge cases",
            "justify",
            "prove",
            "explain why",
            "detailed reasoning",
            "multiple conditions",
        ]
        prompt_lower = prompt.lower()
        for trigger in constraint_triggers:
            if trigger in prompt_lower:
                score += 0.25
        return min(score, 1.0)

    def select_fireworks_model(self, domain: str, complexity: float) -> str:
        if not settings.allowed_models:
            return "default"
        # Default: strongest model first
        if domain in ("code_generation", "code_debugging", "logical_reasoning", "mathematical_reasoning"):
            return settings.allowed_models[0]
        if complexity >= 0.7:
            return settings.allowed_models[0]
        if len(settings.allowed_models) > 1:
            return settings.allowed_models[1]
        return settings.allowed_models[0]

    def route(self, prompt: str) -> Dict[str, Any]:
        domain = self.classify_domain(prompt)
        complexity = self.estimate_complexity(prompt, domain)

        # Raised threshold to reduce unnecessary Fireworks calls while keeping accuracy
        use_fireworks = (
            complexity >= 0.35
            or domain in (
                "code_generation",
                "code_debugging",
                "logical_reasoning",
                "mathematical_reasoning",
                "summarization",
                "factual_knowledge",
            )
        )

        selected_model = self.select_fireworks_model(domain, complexity)

        if use_fireworks:
            try:
                fw_response = fireworks_client.generate(
                    prompt=prompt,
                    model=selected_model,
                )
                if not fw_response.get("success", False):
                    raise FireworksClientError(fw_response.get("error", "Unknown Fireworks error"))
                stats.record_fireworks(
                    prompt_tokens=fw_response.get("prompt_tokens", 0),
                    completion_tokens=fw_response.get("completion_tokens", 0),
                    latency=fw_response.get("latency", 0.0),
                    route=f"fireworks:{fw_response.get('model', 'unknown')}",
                    domain=domain,
                )
                return {
                    "success": True,
                    "answer": normalize_output(fw_response.get("answer", "")),
                    "route": f"fireworks:{fw_response.get('model', 'unknown')}",
                    "prompt_tokens": fw_response.get("prompt_tokens", 0),
                    "completion_tokens": fw_response.get("completion_tokens", 0),
                    "total_tokens": fw_response.get("total_tokens", 0),
                    "latency": fw_response.get("latency", 0.0),
                    "domain": domain,
                }
            except FireworksClientError as e:
                print(f"⚠️ Fireworks API failed: {e}. Falling back to local model.")

        # Local model inference
        local_response = ask_local_model(prompt)

        # If local is weak, attempt a single Fireworks retry
        if local_response.get("weak", False):
            try:
                fw_response = fireworks_client.generate(
                    prompt=prompt,
                    model=selected_model,
                )
                if fw_response.get("success", False):
                    # record that we fell back from local to fireworks
                    try:
                        stats.record_fallback()
                    except Exception:
                        pass
                    stats.record_fireworks(
                        prompt_tokens=fw_response.get("prompt_tokens", 0),
                        completion_tokens=fw_response.get("completion_tokens", 0),
                        latency=fw_response.get("latency", 0.0),
                        route=f"fireworks:{fw_response.get('model', 'unknown')}",
                        domain=domain,
                    )
                    return {
                        "success": True,
                        "answer": normalize_output(fw_response.get("answer", "")),
                        "route": f"fireworks:{fw_response.get('model', 'unknown')}",
                        "prompt_tokens": fw_response.get("prompt_tokens", 0),
                        "completion_tokens": fw_response.get("completion_tokens", 0),
                        "total_tokens": fw_response.get("total_tokens", 0),
                        "latency": fw_response.get("latency", 0.0),
                        "domain": domain,
                    }
            except FireworksClientError:
                # If retry fails, continue and return local response
                pass

        # Record local call metrics including domain
        try:
            stats.record_local(
                latency=local_response.get("latency", 0.0),
                route="local",
                domain=domain,
            )
        except TypeError:
            # Fallback if stats.record_local signature doesn't accept domain
            try:
                stats.record_local(latency=local_response.get("latency", 0.0), route="local")
            except Exception:
                pass

        # Ensure local_response contains normalized answer and route
        local_response.setdefault("route", "local")
        if "answer" in local_response:
            local_response["answer"] = normalize_output(local_response.get("answer", ""))
        local_response.setdefault("prompt_tokens", 0)
        local_response.setdefault("completion_tokens", 0)
        local_response.setdefault("total_tokens", local_response.get("prompt_tokens", 0) + local_response.get("completion_tokens", 0))
        local_response.setdefault("latency", local_response.get("latency", 0.0))
        local_response.setdefault("success", True)

        # Attach domain for downstream metrics/analysis
        local_response["domain"] = domain

        return local_response


router = TaskRouter()
