"""
agent.py
========

Main entry point for the AMD Developer Hackathon.
Reads tasks, routes them, and writes results.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

from router import router
from stats import stats

# Default Hackathon paths (can be overridden for local testing)
INPUT_FILE = os.getenv("INPUT_FILE", "/input/tasks.json")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "/output/results.json")


def load_tasks(input_path: str) -> List[Dict[str, Any]]:
    """Load tasks safely from JSON input file."""

    if not os.path.exists(input_path):
        print(f"CRITICAL ERROR: Required input file does not exist at '{input_path}'")
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        if not isinstance(tasks, list):
            raise ValueError("Input JSON root must be a list of task objects.")

        return tasks

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to parse {input_path}: {e}")
        sys.exit(1)


def write_results(output_path: str, results: List[Dict[str, str]]) -> None:
    """Create output directory (if needed) and write results."""

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Successfully saved {len(results)} results to {output_path}")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed writing results to {output_path}: {e}")
        sys.exit(1)


def main() -> None:

    start_time = time.perf_counter()

    tasks = load_tasks(INPUT_FILE)

    print(f"Loaded {len(tasks)} tasks from {INPUT_FILE}")

    results: List[Dict[str, str]] = []

    for idx, task in enumerate(tasks, start=1):

        task_id = task.get("task_id", f"unknown-task-{idx}")
        prompt = task.get("prompt", "")

        print(
            f"[{idx}/{len(tasks)}] Processing Task ID: {task_id} ...",
            end=" ",
            flush=True,
        )

        if not prompt:
            print("SKIPPED (Empty prompt)")
            results.append(
                {
                    "task_id": task_id,
                    "answer": "",
                }
            )
            continue

        try:

            route_res = router.route(prompt)

            answer_text = str(
                route_res.get("answer", "")
            ).strip()

            results.append(
                {
                    "task_id": task_id,
                    "answer": answer_text,
                }
            )

            print(f"DONE ({route_res.get('route', 'unknown')})")

        except Exception as err:

            print(f"ERROR: {err}")

            results.append(
                {
                    "task_id": task_id,
                    "answer": "",
                }
            )

    write_results(OUTPUT_FILE, results)

    total_duration = time.perf_counter() - start_time

    print("\n====================================================")
    print("Execution Summary")
    print("====================================================")
    print(f"Total Tasks Processed : {len(results)}")
    print(f"Execution Time        : {total_duration:.2f} seconds")

    if hasattr(stats, "summary"):
        stats.summary()
    elif hasattr(stats, "print_summary"):
        stats.print_summary()

    print("====================================================")

    sys.exit(0)


if __name__ == "__main__":
    main()