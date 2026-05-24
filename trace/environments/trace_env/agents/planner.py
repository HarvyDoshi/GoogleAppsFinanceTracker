"""
environments/trace_env/agents/planner.py

PlannerAgent — decomposes a long-horizon instruction into a sub-task list.

Uses a local LLM call (or rule-based fallback) to generate a structured plan.
The plan is stored in episode state and scored by the reward function.
"""

from __future__ import annotations
import re


class PlannerAgent:
    def __init__(self, config: dict):
        self.config = config

    def decompose(self, instruction: str, task: dict) -> list[str]:
        """
        Decompose the instruction into ordered sub-tasks.

        In the full system, this calls a local LLM.
        Here we use a rule-based decomposer for environment stability.
        """
        sources = task.get("available_sources", ["gmail"])
        steps = []

        # Step 1: Always start with retrieval planning
        for source in sources:
            steps.append(f"Query {source} for: {instruction[:60]}...")

        # Step 2: Cross-reference if multiple sources
        if len(sources) > 1:
            steps.append("Cross-reference findings across sources")

        # Step 3: Synthesize
        steps.append("Synthesize all retrieved findings")

        # Step 4: Verify
        steps.append("Verify the synthesized answer against world model")

        # Step 5: Format and answer
        steps.append("Format and submit final answer")

        return steps
