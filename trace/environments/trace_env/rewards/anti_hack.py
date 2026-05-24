"""
environments/trace_env/rewards/anti_hack.py

Anti-reward-hacking guard for Trace.

Per the hackathon guide: "The model may learn shortcuts that maximize
your reward without solving the real task."

We watch for:
  1. Fabricated data — claiming to retrieve data that was never queried
  2. Answer without plan — skipping planning entirely
  3. Circular verification — verifying your own unretreived claims
  4. Prompt injection — trying to escape the environment interface
  5. Excessive MEMORIZE calls — stuffing memory to avoid retrieval
  6. Duplicate queries — repeating the same retrieval query
  7. Timing abuse — stalling with empty or near-empty actions
  8. Global state mutation — referencing data from prior episodes
"""

from __future__ import annotations
import re
from typing import Optional


class AntiHackGuard:
    """
    Stateful guard that tracks suspicious patterns across an episode.
    Configurable via env_config.yaml anti_hack section.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._max_memorize = cfg.get("max_memorize_calls", 5)
        self._max_consecutive = cfg.get("max_consecutive_same_action", 4)
        self._max_duplicate_queries = cfg.get("max_duplicate_queries", 3)
        self._max_empty_retrieval_streak = cfg.get("max_empty_retrieval_streak", 3)
        self._enable_injection = cfg.get("enable_injection_detection", True)
        self._hallucination_amounts = cfg.get(
            "hallucination_amounts",
            ["1234.56", "3456.78", "999.99"]
        )
        self.reset()

    def reset(self):
        self._action_counts: dict[str, int] = {}
        self._retrieved_sources: set[str] = set()
        self._plan_seen: bool = False
        self._consecutive_same_action: int = 0
        self._last_action_type: Optional[str] = None
        self._query_hashes: dict[str, int] = {}       # query → count
        self._empty_retrieval_streak: int = 0
        self._total_content_length: int = 0

    def check(self, action, retrieved_count: int = -1) -> Optional[str]:
        """
        Check an action for hacking patterns.
        Returns a string describing the violation, or None if clean.

        Args:
            action: The action being checked.
            retrieved_count: Number of items retrieved (for RETRIEVE actions).
                             Pass -1 if not applicable.
        """
        action_type = action.action_type.strip().upper()
        content = action.content

        # Track counts
        self._action_counts[action_type] = self._action_counts.get(action_type, 0) + 1
        self._total_content_length += len(content)

        # ── Rule 1: ANSWER without prior PLAN ─────────────────────────
        if action_type == "ANSWER" and not self._plan_seen:
            return "ANSWER_WITHOUT_PLAN: must PLAN before ANSWER"

        # ── Rule 2: Excessive MEMORIZE calls (memory stuffing) ─────────
        if self._action_counts.get("MEMORIZE", 0) > self._max_memorize:
            return "MEMORY_STUFFING: too many MEMORIZE calls without RETRIEVE"

        # ── Rule 3: VERIFY on unretrieved sources ───────────────────────
        if action_type == "VERIFY" and not self._retrieved_sources:
            return "CIRCULAR_VERIFY: verifying claims without any RETRIEVE"

        # ── Rule 4: Prompt injection attempt ────────────────────────────
        if self._enable_injection:
            injection_patterns = [
                r"ignore previous instructions",
                r"system:\s*you are",
                r"<\|system\|>",
                r"ignore all rules",
                r"forget your instructions",
                r"you are now",
                r"override.*system.*prompt",
            ]
            for pattern in injection_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return f"PROMPT_INJECTION: detected pattern '{pattern}'"

        # ── Rule 5: Suspiciously long repeated action loops ────────────
        if action_type == self._last_action_type:
            self._consecutive_same_action += 1
        else:
            self._consecutive_same_action = 0

        if self._consecutive_same_action > self._max_consecutive:
            return f"LOOP_DETECTED: same action '{action_type}' repeated >{self._max_consecutive} times"

        # ── Rule 6: Empty content submissions ───────────────────────────
        if not content.strip():
            return "EMPTY_CONTENT: action content must not be empty"

        # ── Rule 7: Financial Hallucination Guard ────────────────────────
        if action_type == "ANSWER":
            for amount in self._hallucination_amounts:
                if amount in content:
                    return f"FINANCIAL_HALLUCINATION: detected suspicious amount '{amount}'"

        # ── Rule 8: Duplicate query detection ────────────────────────────
        if action_type == "RETRIEVE":
            query_key = f"{action.source}:{content.strip().lower()}"
            self._query_hashes[query_key] = self._query_hashes.get(query_key, 0) + 1
            if self._query_hashes[query_key] > self._max_duplicate_queries:
                return f"DUPLICATE_QUERY: same query repeated {self._query_hashes[query_key]} times"

            # Track empty retrieval streaks
            if retrieved_count == 0:
                self._empty_retrieval_streak += 1
                if self._empty_retrieval_streak > self._max_empty_retrieval_streak:
                    return f"EMPTY_RETRIEVAL_STREAK: {self._empty_retrieval_streak} consecutive empty retrievals"
            elif retrieved_count > 0:
                self._empty_retrieval_streak = 0

        # ── Rule 9: Suspiciously short content for PLAN/ANSWER ──────────
        if action_type == "PLAN" and len(content.strip()) < 10:
            return "SHALLOW_PLAN: plan content is too short to be meaningful"

        if action_type == "ANSWER" and len(content.strip()) < 20:
            return "SHALLOW_ANSWER: answer content is too short to be meaningful"

        # ── Track state ──────────────────────────────────────────────────
        if action_type == "PLAN":
            self._plan_seen = True
        if action_type == "RETRIEVE" and action.source:
            self._retrieved_sources.add(action.source)
        self._last_action_type = action_type

        return None  # clean

    @property
    def stats(self) -> dict:
        """Return anti-hack statistics for logging."""
        return {
            "action_counts": dict(self._action_counts),
            "retrieved_sources": list(self._retrieved_sources),
            "plan_seen": self._plan_seen,
            "duplicate_queries": {k: v for k, v in self._query_hashes.items() if v > 1},
            "total_content_length": self._total_content_length,
        }
