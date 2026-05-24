"""
Trace Environment Implementation.

An RL environment for Gmail financial transaction intelligence.
Wraps the internal TraceEnv with the OpenEnv Environment interface.
"""

import os
import sys
import yaml
import logging
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so internal imports work in Docker
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from ..models import TraceAction, TraceObservation
except (ImportError, SystemError):
    from models import TraceAction, TraceObservation

# Import the internal environment
from environments.trace_env.core.env import TraceEnv
from environments.trace_env.core.schemas import TraceAction as InternalAction


class TraceEnvironment(Environment):
    """
    OpenEnv-compatible wrapper around the Trace internal environment.

    The environment fetches real financial transaction data from Gmail,
    parses PDF/HTML attachments, and syncs to Google Sheets.

    Example:
        >>> env = TraceEnvironment()
        >>> obs = env.reset()
        >>> print(obs.instruction)
        >>>
        >>> obs = env.step(TraceAction(action_type="RETRIEVE", content="financial emails", source="gmail"))
        >>> print(obs.context)
    """

    # Single-session only — Gmail tokens are per-user
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        """Initialize the Trace environment with config."""
        config_path = os.path.join(_PROJECT_ROOT, "configs", "env_config.yaml")
        try:
            with open(config_path) as f:
                self._config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}, using defaults")
            self._config = {}

        self._env = TraceEnv(self._config)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> TraceObservation:
        """
        Reset the environment and start a new episode.

        Returns:
            TraceObservation with initial state
        """
        task = {
            "instruction": "Audit my financial transactions from Gmail.",
            "difficulty": "medium",
            "available_sources": ["gmail", "image"],
            "ground_truth": {},
        }
        obs = self._env.reset(task)
        self._state = State(episode_id=obs.episode_id, step_count=0)

        return TraceObservation(
            episode_id=obs.episode_id,
            step=obs.step,
            instruction=obs.instruction,
            available_sources=obs.available_sources,
            context=obs.context,
            memory_summary=obs.memory_summary,
            world_state=obs.world_state,
            done=False,
            reward=0.0,
        )

    def step(self, action: TraceAction) -> TraceObservation:  # type: ignore[override]
        """
        Execute one agent action in the environment.

        Args:
            action: TraceAction containing action_type, content, source

        Returns:
            TraceObservation with results
        """
        # Convert OpenEnv action → internal action
        internal_action = InternalAction(
            action_type=action.action_type,
            content=action.content,
            source=action.source,
            metadata=action.metadata,
        )

        obs, reward, done, info = self._env.step(internal_action)
        self._state.step_count += 1

        return TraceObservation(
            episode_id=obs.episode_id,
            step=obs.step,
            instruction=obs.instruction,
            available_sources=obs.available_sources,
            context=obs.context,
            memory_summary=obs.memory_summary,
            world_state=obs.world_state,
            done=done,
            reward=reward,
            metadata=info,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
