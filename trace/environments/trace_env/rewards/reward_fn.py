"""
environments/trace_env/rewards/reward_fn.py

Multi-component reward function for Trace.

Per the hackathon guide: "Your reward function is your task specification."
We use MULTIPLE independent reward signals to prevent reward hacking.

This module provides TWO interfaces:
  1. `compute_reward()` — env-side scoring (called from TraceEnv.step)
  2. `make_trl_reward_functions()` — returns a list of independent reward
     functions with the TRL-expected signature:
       fn(completions: list[str], prompts: list[str], **kw) -> list[float]

Reward components:
  1. plan_quality        — does the plan decompose the goal correctly?
  2. retrieval_coverage  — did the agent retrieve relevant data?
  3. answer_correctness  — is the final answer close to ground truth?
  4. step_efficiency     — fewer steps is better (up to a point)
  5. verification_bonus  — extra reward for verifying before answering
  6. format_compliance   — answer follows expected schema
  7. process_reward      — intermediate step-level quality signals

Anti-hack penalties:
  8. hack_penalty        — if anti-hack guard triggered
  9. terminal_penalty    — timeout or max-steps exceeded
"""

from __future__ import annotations
from typing import Optional, Any
import json
import re
import requests


# ═══════════════════════════════════════════════════════════════════════════════
# ENV-SIDE REWARD (used inside TraceEnv.step)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reward(
    action,
    state,
    hack_penalty: bool = False,
    terminal_penalty: bool = False,
    config: Optional[dict] = None,
) -> float:
    """
    Compute reward for a single step.
    Returns a float in [-1.0, +2.0].
    All components are logged separately for monitoring.
    """
    if terminal_penalty:
        return -1.0

    if hack_penalty:
        return -0.8

    if action is None:
        return 0.0

    # Difficulty-scaled reward multiplier
    difficulty = state.task.get("difficulty", "easy")
    diff_scale = 1.0
    if config:
        scales = config.get("difficulty_reward_scale", {})
        diff_scale = scales.get(difficulty, 1.0)

    components = {}

    action_type = action.action_type.strip().upper()

    # ── 1. Plan quality (PLAN action) ────────────────────────────────────
    if action_type == "PLAN":
        components["plan_quality"] = _score_plan(action.content, state.task)
    else:
        components["plan_quality"] = 0.0

    # ── 2. Retrieval coverage (RETRIEVE action) ──────────────────────────
    if action_type == "RETRIEVE":
        components["retrieval_coverage"] = _score_retrieval(
            state.retrieved_data, state.task
        )
    else:
        components["retrieval_coverage"] = 0.0

    # ── 3. Verification bonus (VERIFY action) ────────────────────────────
    if action_type == "VERIFY":
        components["verification_bonus"] = 0.15 if state.verified else 0.0
    else:
        components["verification_bonus"] = 0.0

    # ── 4. Answer correctness (ANSWER action) ────────────────────────────
    if action_type == "ANSWER":
        components["answer_correctness"] = _score_answer(
            action.content, state
        )
        components["format_compliance"] = _score_format(action.content)
        components["step_efficiency"] = _score_efficiency(state.steps)
    else:
        components["answer_correctness"] = 0.0
        components["format_compliance"] = 0.0
        components["step_efficiency"] = 0.0

    # ── 5. Process reward: did we plan before retrieving? ────────────────
    components["process_reward"] = _score_process(state)

    total = sum(components.values()) * diff_scale

    # Clip to [-1.0, 2.0]
    total = max(-1.0, min(2.0, total))

    # Track cumulative reward on state
    if hasattr(state, "cumulative_reward"):
        state.cumulative_reward += total

    # Log components (in production: send to W&B / TRL logging)
    _log_reward(components, total, state.episode_id, state.steps)

    return total


# ═══════════════════════════════════════════════════════════════════════════════
# TRL-FACING REWARD FUNCTIONS (passed to GRPOTrainer.reward_funcs)
#
# Per guide: "Use multiple independent reward functions, not just one."
# Each function has signature:
#   fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]
# ═══════════════════════════════════════════════════════════════════════════════

def make_trl_reward_functions(env_url: str) -> list[callable]:
    """
    Build a list of independent reward functions for TRL's GRPOTrainer.

    Each function checks one aspect of quality. Multiple independent
    signals reduce the risk of reward hacking (per guide).

    Args:
        env_url: URL of the running OpenEnv server.

    Returns:
        List of reward functions, each with TRL-expected signature.
    """
    return [
        _make_format_reward_fn(),
        _make_correctness_reward_fn(env_url),
        _make_efficiency_reward_fn(env_url),
        _make_process_reward_fn(env_url),
        _make_anti_hack_reward_fn(),
    ]


def _make_format_reward_fn():
    """
    Reward function 1: Format compliance.
    Checks if the model output is valid JSON with required fields.
    Does NOT call the environment — pure local check.
    """

    def format_reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            score = 0.0
            text = completion.strip()

            # Try to extract JSON from the completion
            try:
                # Handle cases where model wraps JSON in markdown code blocks
                json_text = text
                if "```json" in text:
                    json_text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    json_text = text.split("```")[1].split("```")[0].strip()

                parsed = json.loads(json_text)

                # Valid JSON: base reward
                score += 0.3

                # Has required field: action_type
                if "action_type" in parsed:
                    score += 0.2
                    # Valid action type
                    valid_types = {"PLAN", "RETRIEVE", "MEMORIZE", "VERIFY", "ANSWER"}
                    if parsed["action_type"].upper() in valid_types:
                        score += 0.2

                # Has required field: content
                if "content" in parsed and len(str(parsed["content"]).strip()) > 0:
                    score += 0.2

                # Penalize extra text outside JSON
                if text != json_text:
                    score -= 0.1

            except (json.JSONDecodeError, IndexError, AttributeError):
                # Not valid JSON at all
                score = -0.5

            rewards.append(score)
        return rewards

    format_reward_fn.__name__ = "format_compliance"
    return format_reward_fn


def _make_correctness_reward_fn(env_url: str):
    """
    Reward function 2: Answer correctness via environment verification.
    Sends the action to the environment and checks the returned reward.
    """

    def correctness_reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            try:
                text = completion.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                action_dict = json.loads(text)

                resp = requests.post(
                    f"{env_url}/step",
                    json=action_dict,
                    timeout=15,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    rewards.append(float(data.get("reward", 0.0)))
                else:
                    rewards.append(-0.3)

            except (json.JSONDecodeError, requests.RequestException, KeyError):
                rewards.append(-0.3)

        return rewards

    correctness_reward_fn.__name__ = "answer_correctness"
    return correctness_reward_fn


def _make_efficiency_reward_fn(env_url: str):
    """
    Reward function 3: Step efficiency.
    Checks the environment state for step count after the action.
    """

    def efficiency_reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            try:
                resp = requests.get(f"{env_url}/state", timeout=5)
                if resp.status_code == 200:
                    state = resp.json()
                    steps = state.get("steps", 20)
                    rewards.append(_score_efficiency(steps))
                else:
                    rewards.append(0.0)
            except requests.RequestException:
                rewards.append(0.0)
        return rewards

    efficiency_reward_fn.__name__ = "step_efficiency"
    return efficiency_reward_fn


def _make_process_reward_fn(env_url: str):
    """
    Reward function 4: Process ordering.
    Rewards PLAN → RETRIEVE → VERIFY → ANSWER sequence.
    """

    def process_reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            try:
                text = completion.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                parsed = json.loads(text)
                action_type = parsed.get("action_type", "").upper()

                # Check env state for process compliance
                resp = requests.get(f"{env_url}/state", timeout=5)
                if resp.status_code == 200:
                    state = resp.json()
                    has_plan = len(state.get("plan", [])) > 0
                    has_data = state.get("retrieved_count", 0) > 0
                    has_verify = state.get("verified", False)

                    if action_type == "ANSWER" and has_plan and has_data and has_verify:
                        rewards.append(0.3)
                    elif action_type == "ANSWER" and has_plan and has_data:
                        rewards.append(0.15)
                    elif action_type == "ANSWER" and has_plan:
                        rewards.append(0.05)
                    elif action_type == "ANSWER":
                        rewards.append(-0.2)  # skipped everything
                    else:
                        rewards.append(0.0)  # intermediate steps: neutral
                else:
                    rewards.append(0.0)

            except (json.JSONDecodeError, requests.RequestException):
                rewards.append(0.0)

        return rewards

    process_reward_fn.__name__ = "process_ordering"
    return process_reward_fn


def _make_anti_hack_reward_fn():
    """
    Reward function 5: Anti-hack penalty.
    Pure local check — no env call needed.
    Penalizes outputs that look like reward hacking.
    """

    HACK_PATTERNS = [
        r"ignore",
        r"bypass",
        r"hack",
        r"cheat",
        r"skip verification",
        r"fabricat",
        r"assume.*total",
        r"pretend",
        r"make up",
    ]

    SUSPICIOUS_AMOUNTS = ["1234.56", "3456.78", "999.99", "0.00"]

    def anti_hack_reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            text = completion.lower()
            penalty = 0.0

            # Check for hack patterns
            for pattern in HACK_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    penalty -= 0.3
                    break  # one hit is enough

            # Check for suspicious placeholder amounts in ANSWER
            for amount in SUSPICIOUS_AMOUNTS:
                if amount in completion:
                    penalty -= 0.2
                    break

            # Penalize extremely long outputs (might be prompt stuffing)
            if len(completion) > 2000:
                penalty -= 0.1

            rewards.append(penalty)
        return rewards

    anti_hack_reward_fn.__name__ = "anti_hack"
    return anti_hack_reward_fn


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT SCORERS (shared by env-side and TRL-side)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_plan(plan_content: str, task: dict) -> float:
    """
    Check if the plan mentions the key sub-tasks implied by the instruction.
    Simple keyword heuristic — replace with LLM-as-judge for harder tasks.
    """
    instruction = task.get("instruction", "").lower()
    plan_lower = plan_content.lower()

    score = 0.0

    # Reward for breaking into multiple steps
    if any(marker in plan_lower for marker in ["step 1", "1.", "first,", "1)"]):
        score += 0.1

    # Reward for mentioning data sources
    sources = task.get("available_sources", [])
    for src in sources:
        if src.lower() in plan_lower:
            score += 0.05

    # Reward for mentioning key task keywords
    keywords = _extract_keywords(instruction)
    for kw in keywords:
        if kw in plan_lower:
            score += 0.05

    # Reward for mentioning verification / checking
    if any(w in plan_lower for w in ["verify", "check", "validate", "confirm"]):
        score += 0.05

    return min(score, 0.3)  # cap at 0.3


def _score_retrieval(retrieved_data: list, task: dict) -> float:
    """
    Reward for retrieving relevant data (non-empty, diverse sources).
    """
    if not retrieved_data:
        return -0.05  # penalize empty retrieval

    score = 0.1  # base reward for trying

    # Bonus for volume (up to a point)
    total_items = sum(
        len(batch) if isinstance(batch, list) else 1
        for batch in retrieved_data
    )
    score += min(0.1, total_items * 0.005)

    return min(score, 0.2)


def _score_answer(answer: str, state) -> float:
    """
    Score the final answer against ground truth.
    """
    ground_truth = state.task.get("ground_truth", {})
    gt_answer = ground_truth.get("answer")
    expected_numeric = ground_truth.get("expected_numeric_target")

    score = 0.0

    if expected_numeric is not None:
        # Extract numbers from answer
        nums = re.findall(r"\d+\.?\d*", answer.replace(",", ""))
        nums = [float(n) for n in nums]
        if nums:
            closest = min(nums, key=lambda x: abs(x - expected_numeric))
            error = abs(closest - expected_numeric) / max(expected_numeric, 0.01)
            if error < 0.05:
                score += 0.8
            elif error < 0.15:
                score += 0.4
            elif error < 0.30:
                score += 0.2

    if gt_answer is None and expected_numeric is None:
        # No ground truth — reward for coherent, non-empty answer
        if len(answer.strip()) > 100:
            return 0.4
        elif len(answer.strip()) > 50:
            return 0.3
        else:
            return 0.1

    answer_lower = answer.lower()
    if gt_answer:
        gt_lower = str(gt_answer).lower()
        if gt_lower in answer_lower:
            score += 1.0
        else:
            gt_keywords = gt_lower.split()
            matches = sum(1 for kw in gt_keywords if kw in answer_lower)
            coverage = matches / max(len(gt_keywords), 1)
            score += coverage * 0.7

    return min(score, 1.0)


def _score_format(answer: str) -> float:
    """
    Check that the answer follows the expected output format.
    Expected: contains a summary and optionally a list.
    """
    score = 0.0
    if len(answer) > 100:
        score += 0.05
    if any(marker in answer for marker in ["-", "•", "1.", "\n"]):
        score += 0.05
    if "$" in answer or "₹" in answer or "€" in answer:
        score += 0.05
    return score


def _score_efficiency(steps: int) -> float:
    """
    Reward for solving the task in fewer steps.
    Optimal: ~8 steps for easy, ~15 for hard.
    """
    if steps <= 5:
        return 0.15
    elif steps <= 10:
        return 0.10
    elif steps <= 15:
        return 0.05
    else:
        return 0.0


def _score_process(state) -> float:
    """
    Process reward: did the agent follow the right order?
    PLAN → RETRIEVE → VERIFY → ANSWER is the gold sequence.
    """
    has_plan = bool(state.plan)
    has_data = bool(state.retrieved_data)
    has_verify = state.verified

    if has_plan and has_data and has_verify:
        return 0.1
    if has_plan and has_data:
        return 0.05
    if has_plan:
        return 0.02
    return 0.0


def _extract_keywords(instruction: str) -> list[str]:
    """Extract content keywords from the instruction."""
    stopwords = {"a", "an", "the", "and", "or", "of", "in", "to", "for", "all",
                 "from", "with", "that", "this", "have", "been", "they", "their",
                 "what", "which", "when", "where", "find", "list"}
    words = re.findall(r'\b\w{4,}\b', instruction.lower())
    return [w for w in words if w not in stopwords][:10]


def _log_reward(components: dict, total: float, episode_id: str, step: int):
    """Log reward components. Replace with W&B/TRL logging in production."""
    print(
        f"[REWARD] ep={episode_id[:8]} step={step} total={total:.3f} | "
        + " ".join(f"{k}={v:.3f}" for k, v in components.items() if v != 0)
    )
