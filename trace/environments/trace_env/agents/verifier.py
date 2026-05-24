"""
environments/trace_env/agents/verifier.py

VerifierAgent — cross-checks claims against the world model and memory.

This is the "plan-act-verify" V in Trace's framework.
Verification is a prerequisite for a high-quality ANSWER action.
The reward function gives a bonus for verified episodes.
"""

from __future__ import annotations
from typing import Any


class VerifierAgent:
    def __init__(self, config: dict):
        self.config = config

    def verify(self, claim: str, world_model, memory) -> dict[str, Any]:
        """
        Verify a claim against the world model and episodic memory.

        Returns:
            {
                "passed": bool,
                "confidence": float,
                "reason": str,
            }
        """
        claim_lower = claim.lower()

        # ── Check 1: Is the claim grounded in retrieved data? ────────────
        retrieved_count = len(world_model._retrieved_ids)
        if retrieved_count == 0:
            return {
                "passed": False,
                "confidence": 0.0,
                "reason": "No data has been retrieved yet — cannot verify.",
            }

        # ── Check 2: Does memory contain supporting evidence? ────────────
        relevant_memories = memory.recall(claim, top_k=3)
        memory_support = len(relevant_memories) > 0

        # ── Check 3: Does world model snapshot corroborate the claim? ────
        snapshot = world_model.snapshot()
        visible_items = snapshot.get("visible_items", 0)
        corroborated = visible_items > 0 and any(
            any(str(v).lower() in claim_lower for v in item.values())
            for item in snapshot.get("retrieved_preview", [])
        )

        passed = memory_support or corroborated
        confidence = (0.5 if memory_support else 0.0) + (0.5 if corroborated else 0.0)
        reason = (
            f"Memory support: {memory_support}, "
            f"World model corroboration: {corroborated}, "
            f"Retrieved items visible: {visible_items}"
        )

        return {"passed": passed, "confidence": confidence, "reason": reason}
