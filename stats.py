"""
stats.py

Stores routing statistics for the Hybrid Router.
Optimized for AMD Hackathon: Telemetry, console logging, and dynamic route tracking.
"""

from dataclasses import dataclass, field


@dataclass
class Stats:
    total_tasks: int = 0
    fireworks_calls: int = 0
    local_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency: float = 0.0
    routes: list = field(default_factory=list)


class StatsManager:
    def __init__(self):
        self.stats = Stats()

    def record_fireworks(self, prompt_tokens: int, completion_tokens: int, latency: float, route: str):
        """Wrapper specifically for Fireworks API calls."""
        self.record(
            route=route,
            latency=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )

    def record_local(self, latency: float, route: str = "local"):
        """Wrapper specifically for Local Model calls (0 tokens)."""
        self.record(
            route=route,
            latency=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )

    def record(self, route: str, latency: float, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        """Core internal method for recording task telemetry."""
        self.stats.total_tasks += 1
        self.stats.total_latency += latency
        self.stats.prompt_tokens += prompt_tokens
        self.stats.completion_tokens += completion_tokens
        self.stats.total_tokens += total_tokens
        self.stats.routes.append(route)

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
            "prompt_tokens": self.stats.prompt_tokens,
            "completion_tokens": self.stats.completion_tokens,
            "total_tokens": self.stats.total_tokens,
            "average_latency": round(self.average_latency(), 4),
            "routes": self.stats.routes
        }

    def print_summary(self):
        """Prints a highly visible, formatted summary to the Docker logs."""
        print(f"   Total Tasks Processed: {self.stats.total_tasks}")
        print(f"   Local Model Calls:     {self.stats.local_calls} (0 Tokens)")
        print(f"   Fireworks API Calls:   {self.stats.fireworks_calls}")
        print(f"   -------------------------------------------")
        print(f"   Total Prompt Tokens:       {self.stats.prompt_tokens}")
        print(f"   Total Completion Tokens:   {self.stats.completion_tokens}")
        print(f"   TOTAL HACKATHON TOKENS:    {self.stats.total_tokens}")
        print(f"   -------------------------------------------")
        print(f"   Average Task Latency:  {self.average_latency():.3f}s")

    def reset(self):
        self.stats = Stats()


# Global singleton
stats = StatsManager()