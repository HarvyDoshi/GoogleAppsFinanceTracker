"""
Data models for the Trace Environment.

The Trace environment provides Gmail financial transaction intelligence.
It retrieves emails, parses PDF/HTML attachments, and builds summaries.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Any, Optional


class TraceAction(Action):
    """Action for the Trace environment — agent sends a tool command."""

    action_type: str = Field(
        ..., description="PLAN | RETRIEVE | MEMORIZE | VERIFY | ANSWER | EXPORT"
    )
    content: str = Field(..., description="The payload (query, answer, claim…)")
    source: Optional[str] = Field(
        None, description="Used with RETRIEVE: 'gmail' | 'drive' | 'image'"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceObservation(Observation):
    """Observation from the Trace environment — environment returns state."""

    episode_id: str = Field(default="", description="Unique ID for the episode")
    step: int = Field(default=0, description="Current step count")
    instruction: str = Field(default="", description="The original goal")
    available_sources: list[str] = Field(
        default_factory=list, description="Available data sources"
    )
    context: str = Field(default="", description="Result of the last action")
    memory_summary: str = Field(
        default="", description="Compressed episodic memory"
    )
    world_state: dict[str, Any] = Field(
        default_factory=dict, description="Snapshot of the Semantic World Model"
    )
