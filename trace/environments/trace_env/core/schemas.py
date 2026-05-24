"""
environments/trace_env/core/schemas.py

Pydantic dataclasses for the OpenEnv action/observation interface.
These define the typed contract between the agent and the environment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field

# ── Agent → Environment ─────────────────────────────────────────────────────

class TraceAction(BaseModel):
    """
    An action the agent sends to the environment.

    action_type options:
        PLAN      - decompose the instruction into a sub-task list
        RETRIEVE  - fetch data from a named source
        MEMORIZE  - write a finding into episodic memory
        VERIFY    - cross-check a claim against the world model
        ANSWER    - submit the final synthesized answer (ends episode)
     """
    action_type: str
    content: str                        # The payload (query, answer, claim…)
    source: Optional[str] = None        # Used with RETRIEVE: "gmail" | "sheets" | …
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Environment → Agent ─────────────────────────────────────────────────────

class TraceObservation(BaseModel):
    """
    What the agent sees after each step.
    """
    episode_id: str
    step: int
    instruction: str                    # The original long-horizon goal
    available_sources: list[str]        # Which data sources can be queried
    context: str                        # Result of the last action
    memory_summary: str                 # Compressed episodic memory
    world_state: dict[str, Any]         # Snapshot of the Semantic World Model
    metadata: dict[str, Any] = Field(default_factory=dict)  # Pass-through info dict

    def to_prompt(self) -> str:
        """Convert to a formatted prompt string for the LLM agent."""
        sources = ", ".join(self.available_sources)
        return (
            f"[STEP {self.step}]\n"
            f"GOAL: {self.instruction}\n"
            f"SOURCES: {sources}\n"
            f"LAST RESULT: {self.context}\n"
            f"MEMORY: {self.memory_summary}\n"
            f"WORLD STATE: {self.world_state}\n\n"
            f"Choose your next action: PLAN | RETRIEVE | MEMORIZE | VERIFY | ANSWER\n"
            f"Respond in JSON: {{\"action_type\": \"...\", \"content\": \"...\", "
            f"\"source\": \"...\"}}"
        )

    def to_chat_messages(self, system_prompt: Optional[str] = None) -> list[dict]:
        """Convert to chat-template messages for instruct models."""
        sources = ", ".join(self.available_sources)

        if system_prompt is None:
            system_prompt = (
                "You are an expert financial auditor. Accurately retrieve, verify, "
                "and aggregate transactions from digital sources.\n"
                "You MUST respond ONLY with valid JSON in this exact format:\n"
                '{"action_type": "PLAN|RETRIEVE|MEMORIZE|VERIFY|ANSWER", '
                '"content": "...", "source": "gmail|sheets|null"}\n'
                "Do NOT include any text outside the JSON object."
            )

        user_content = (
            f"[STEP {self.step}]\n"
            f"GOAL: {self.instruction}\n"
            f"SOURCES: {sources}\n"
            f"LAST RESULT: {self.context[:500]}\n"
            f"MEMORY: {self.memory_summary}\n\n"
            f"Choose your next action and respond with valid JSON only."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]


# ── Internal episode state ───────────────────────────────────────────────────

@dataclass
class EpisodeState:
    """
    Full internal episode state — used by the reward function and debugger.
    NOT sent to the agent directly; available via env.state().
    """
    episode_id: str
    task: dict[str, Any]
    plan: list[str]
    retrieved_data: list[Any]
    verified: bool
    steps: int
    done: bool
    final_answer: Optional[str] = None
    hack_flags: list[str] = field(default_factory=list)
    cumulative_reward: float = 0.0            # track total episode reward
    action_history: list[str] = field(default_factory=list)  # sequence of action_types
    query_history: list[str] = field(default_factory=list)   # track retrieval queries
    step_times: list[float] = field(default_factory=list)    # per-step wall-clock times
