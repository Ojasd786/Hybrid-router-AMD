"""
stats.py

Enhanced routing statistics for the Hybrid Router.
Optimized for AMD Hackathon: Telemetry, console logging, domain tracking, and fallback monitoring.
"""

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Stats:
    total_tasks: int = 0
    fireworks_calls: int = 0
    local_calls: int = 0
    fallback_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency: float = 0.0
    routes: list = field(default_factory=list)
    domains: dict = field(default_factory=lambda: defaultdict(int))
    domain_fallbacks: dict = field(default_factory=lambda: defaultdict(int))


class StatsManager:
    def __init__(self):
        self.stats = Stats()

    def record_fireworks(self, prompt_tokens: int, completion_tokens: int, latency: float, route: str, domain: str = "unknown"):
        """Wrapper specifically for Fireworks API calls."""
        self.record(
            route=route,
            latency=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            domain=domain
        )

    def record_local(self, latency: float, route: str = "local", domain: str = "unknown"):
        """Wrapper specifically for Local Model calls (0 tokens)."""
        self.record(
            route=route,
            latency=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            domain=domain
        )

    def record_fallback(self, domain: str = "unknown"):
        """Track when local failed and Fireworks rescued."""
        self.stats.fallback_used += 1
        self.stats.domain_fallbacks[domain] += 1

    def record(self, route: str, latency: float, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0, domain: str = "unknown"):
        """Core internal method for recording task telemetry."""
        self.stats.total_tasks += 1
        self.stats.total_latency += latency
        self.stats.prompt_tokens += prompt_tokens
        self.stats.completion_tokens += completion_tokens
        self.stats.total_tokens += total_tokens
        self.stats.routes.append(route)
        self.stats.domains[domain] += 1

        # Flexible matching for dynamic routes (e.g., "fireworks:qwen-coder")
        if route.startswith("fireworks"):
            self.stats.fireworks_calls += 1
        elif route.startswith("local"):
            self.stats.local_calls += 1

    def average_latency(self) -> float:
        if self.stats.total_tasks == 0:
            return 0.0
        return self.stats.total_latency / self.stats.total_tasks

    def summary(self) -> dict:
        """Returns the summary as a dictionary."""
        return {
            "total_tasks": self.stats.total_tasks,
            "fireworks_calls": self.stats.fireworks_calls,
            "local_calls": self.stats.local_calls,
            "fallback_used": self.stats.fallback_used,
            "prompt_tokens": self.stats.prompt_tokens,
            "completion_tokens": self.stats.completion_tokens,
            "total_tokens": self.stats.total_tokens,
            "average_latency": round(self.average_latency(), 4),
            "routes": list(self.stats.routes),
            "domains": dict(self.stats.domains),
            "domain_fallbacks": dict(self.stats.domain_fallbacks),
        }

    def print_summary(self):
        """Prints a highly visible, formatted summary to the Docker logs."""
        print("========== HYBRID ROUTER STATS ==========")
        print(f"   Total Tasks Processed: {self.stats.total_tasks}")
        print(f"   Local Model Calls:     {self.stats.local_calls} (0 Tokens)")
        print(f"   Fireworks API Calls:   {self.stats.fireworks_calls}")
        print(f"   Fallbacks Triggered:   {self.stats.fallback_used}")
        print("   -------------------------------------------")
        print(f"   Total Prompt Tokens:       {self.stats.prompt_tokens}")
        print(f"   Total Completion Tokens:   {self.stats.completion_tokens}")
        print(f"   TOTAL HACKATHON TOKENS:    {self.stats.total_tokens}")
        print("   -------------------------------------------")
        print(f"   Average Task Latency:  {self.average_latency():.3f}s")
        print("   -------------------------------------------")
        print("   Domain Distribution:")
        for domain, count in dict(self.stats.domains).items():
            print(f"      {domain}: {count}")
        print("   -------------------------------------------")
        print("   Domain Fallbacks:")
        for domain, count in dict(self.stats.domain_fallbacks).items():
            print(f"      {domain}: {count}")
        print("===========================================")

    def reset(self):
        self.stats = Stats()


# Global singleton
stats = StatsManager()
