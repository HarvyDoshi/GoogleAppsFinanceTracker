"""
training/dataset.py

TaskCurriculum — generates tasks with progressive difficulty.

Per hackathon guide: "Make success possible early."
Start with short horizons, then gradually remove scaffolding.

Uses weighted sampling for smoother transitions between difficulty tiers,
and chat-template formatting for instruct models.

Difficulty tiers:
  easy   — single source, short horizon, explicit answer
  medium — 2 sources, moderate horizon, partial ground truth
  hard   — 2+ sources, long horizon, schema drift, ambiguous
"""

from __future__ import annotations
import random
import json
from typing import Optional


# ── Instruction pools (no Drive references — Gmail + Sheets only) ────────────

EASY_INSTRUCTIONS = [
    "Find all travel and ride receipts from Gmail in the last 10 days and list them.",
    "Summarize all Uber and Rapido emails from 2023.",
    "What is the total amount spent on shopping invoices in January 2022?",
    "Find emails with the word 'flight' or 'hotel' in 2022.",
    "Read my historical transactions from Google Sheets.",
    "Retrieve the latest 20 financial emails from Gmail.",
    "Check Google Sheets for the total spend last month.",
    "Find all food delivery receipts in Gmail from last week.",
    "List all Rapido ride receipts from the past month.",
    "How many Uber trips did I take in December 2023?",
    "Find all Swiggy or Zomato receipts from January 2024.",
    "Retrieve the most recent 5 payment confirmation emails.",
    "What was my total ride spending in the last 30 days?",
    "Find all emails with 'invoice' in the subject from 2023.",
    "List all shopping order confirmations from Amazon in Gmail.",
]

MEDIUM_INSTRUCTIONS = [
    "Audit all ride receipts from Gmail between 2022 and 2023, and calculate the total spend by vendor.",
    "Find all travel-related emails from 2022-2023 and summarize total expenses and dates.",
    "Identify any recurring subscriptions and their monthly costs across my email history from 2022-2024.",
    "Summarize my Gmail shopping invoices from 2023 and sum the totals.",
    "Retrieve transactions from Gmail, then check Google Sheets to see if they are already logged.",
    "Compare the travel expenses found in Gmail with the records in Google Sheets.",
    "Aggregate my food orders from Gmail and update the summary in Google Sheets.",
    "Find all ride receipts from multiple vendors (Uber, Rapido, Ola) and categorize by vendor.",
    "Calculate the monthly average spend on food delivery from Jan-Jun 2023.",
    "Cross-reference Gmail transaction amounts with Google Sheets ledger entries.",
    "Find all hotel bookings from 2022-2023 and calculate the total accommodation spend.",
    "Identify the top 3 merchants by transaction count from the last 6 months.",
]

HARD_INSTRUCTIONS = [
    "Perform a full financial audit of my travel and ride footprint from 2022 to 2024, flag any missing receipts, and produce a summary report with exact amounts.",
    "Build a complete breakdown of my financial transactions across categories (rides, travel, shopping) from 2022-2024, identifying the top 5 vendors by spend.",
    "Analyze all receipts, tax invoices, and bookings across Gmail from 2022-2024. Extract all numeric totals and aggregate them by category.",
    "Given my financial history across 2022-2024, estimate total annual ride costs.",
    "Retrieve all financial transactions from Gmail, sync them to Google Sheets, then generate a merged financial dashboard summary.",
    "Audit the Google Sheets ledger against raw Gmail receipts to find discrepancies, and output a final reconciled total spend.",
    "Fetch all unlogged invoices from Gmail, sync them to Sheets, and summarize the top spending categories across both sources.",
    "Perform a complete audit: retrieve all Gmail transactions, compare with Sheets, identify missing entries, and produce a reconciliation report with exact amounts and dates.",
    "Calculate year-over-year spending trends from 2022-2024 across all categories, identifying anomalies and missing receipts.",
]


def _make_ground_truth(instruction: str, difficulty: str) -> dict:
    """
    Generate verifiable ground truth structure for the task.

    Per guide: "if the task is verifiable, build the verifier first."
    Ground truth defines WHAT to verify, not hardcoded answers.
    """
    inst_lower = instruction.lower()

    # ── Structure-based ground truth (verifiable without knowing exact amounts) ──
    gt = {
        "answer": None,
        "expected_numeric_target": None,
        "expected_sources": ["gmail"],
        "required_action_sequence": ["PLAN", "RETRIEVE"],
        "min_retrieval_count": 1,
    }

    # Adjust expectations by difficulty
    if difficulty == "easy":
        gt["expected_steps"] = 5
        gt["min_retrieval_count"] = 1
        gt["required_action_sequence"] = ["PLAN", "RETRIEVE", "ANSWER"]

    elif difficulty == "medium":
        gt["expected_steps"] = 10
        gt["expected_sources"] = ["gmail", "sheets"]
        gt["min_retrieval_count"] = 2
        gt["required_action_sequence"] = ["PLAN", "RETRIEVE", "MEMORIZE", "ANSWER"]

    else:  # hard
        gt["expected_steps"] = 15
        gt["expected_sources"] = ["gmail", "sheets"]
        gt["min_retrieval_count"] = 3
        gt["schema_drift"] = True
        gt["required_action_sequence"] = ["PLAN", "RETRIEVE", "VERIFY", "ANSWER"]

    # ── Content-specific expectations ──
    if "sheets" in inst_lower or "ledger" in inst_lower or "sync" in inst_lower:
        gt["expected_sources"] = ["gmail", "sheets"]
    if "compare" in inst_lower or "cross-reference" in inst_lower or "reconcil" in inst_lower:
        gt["expected_sources"] = ["gmail", "sheets"]
        gt["required_action_sequence"].insert(-1, "VERIFY")
    if any(w in inst_lower for w in ["audit", "flag", "discrepan"]):
        if "VERIFY" not in gt["required_action_sequence"]:
            gt["required_action_sequence"].insert(-1, "VERIFY")

    return gt


class TaskCurriculum:
    """
    Samples tasks from easy → medium → hard based on training progress.

    Uses weighted probability sampling for smoother transitions instead
    of hard cutoffs. Per guide: "easy tasks with short horizons first,
    medium tasks with a little more branching, harder tasks only after
    the model starts getting non-zero reward."
    """

    def __init__(self, config: dict):
        self.config = config
        self._global_step = 0
        self._easy_until = config.get("easy_until_step", 50)
        self._medium_until = config.get("medium_until_step", 150)
        self._transition_window = config.get("transition_window", 20)

    def advance(self, step: int):
        """Update the curriculum based on training step."""
        self._global_step = step

    def sample(self) -> dict:
        """Sample a task with weighted difficulty based on current step."""
        difficulty = self._sample_difficulty()

        if difficulty == "easy":
            instruction = random.choice(EASY_INSTRUCTIONS)
            sources = ["gmail"]
        elif difficulty == "medium":
            instruction = random.choice(MEDIUM_INSTRUCTIONS)
            sources = ["gmail", "sheets"]
        else:
            instruction = random.choice(HARD_INSTRUCTIONS)
            sources = ["gmail", "sheets"]

        return {
            "instruction": instruction,
            "difficulty": difficulty,
            "available_sources": sources,
            "ground_truth": _make_ground_truth(instruction, difficulty),
        }

    def _sample_difficulty(self) -> str:
        """
        Weighted sampling with smooth transitions.

        Instead of hard cutoffs:
          step < easy_until - window:       100% easy
          step in [easy_until-window, easy_until+window]: blend easy/medium
          step < medium_until - window:     100% medium
          step in [medium_until-window, medium_until+window]: blend medium/hard
          step > medium_until + window:     100% hard (with some easy for stability)
        """
        step = self._global_step
        w = self._transition_window

        if step < self._easy_until - w:
            return "easy"
        elif step < self._easy_until + w:
            # Blend easy → medium
            progress = (step - (self._easy_until - w)) / (2 * w)
            weights = [1.0 - progress, progress, 0.0]
        elif step < self._medium_until - w:
            return "medium"
        elif step < self._medium_until + w:
            # Blend medium → hard
            progress = (step - (self._medium_until - w)) / (2 * w)
            weights = [0.05, 1.0 - progress, progress]  # keep 5% easy for stability
        else:
            # Mostly hard, with some easy/medium for diversity
            weights = [0.1, 0.2, 0.7]

        return random.choices(["easy", "medium", "hard"], weights=weights, k=1)[0]

    def _current_difficulty(self) -> str:
        """Legacy method — returns the dominant difficulty for the current step."""
        step = self._global_step
        if step < self._easy_until:
            return "easy"
        elif step < self._medium_until:
            return "medium"
        else:
            return "hard"


def format_prompt_for_chat(instruction: str, sources: list[str], step: int = 0,
                           context: str = "", memory: str = "") -> list[dict]:
    """
    Format a training prompt using chat template structure.
    Returns a list of messages suitable for tokenizer.apply_chat_template().
    """
    system_prompt = (
        "You are an expert financial auditor. Accurately retrieve, verify, "
        "and aggregate transactions from digital sources.\n"
        "You MUST respond ONLY with valid JSON in this exact format:\n"
        '{"action_type": "PLAN|RETRIEVE|MEMORIZE|VERIFY|ANSWER", '
        '"content": "...", "source": "gmail|sheets|null"}\n'
        "Do NOT include any text outside the JSON object."
    )

    sources_str = ", ".join(sources)
    user_content = (
        f"[STEP {step}]\n"
        f"GOAL: {instruction}\n"
        f"SOURCES: {sources_str}\n"
    )
    if context:
        user_content += f"LAST RESULT: {context[:500]}\n"
    if memory:
        user_content += f"MEMORY: {memory}\n"
    user_content += "\nChoose your next action and respond with valid JSON only."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
