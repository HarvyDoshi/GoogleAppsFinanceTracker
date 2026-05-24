"""Trace Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import TraceAction, TraceObservation


class TraceEnvClient(
    EnvClient[TraceAction, TraceObservation, State]
):
    """
    Client for the Trace Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with TraceEnvClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.instruction)
        ...
        ...     result = client.step(TraceAction(action_type="RETRIEVE", content="financial emails", source="gmail"))
        ...     print(result.observation.context)
    """

    def _step_payload(self, action: TraceAction) -> Dict:
        """
        Convert TraceAction to JSON payload for step message.

        Args:
            action: TraceAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "content": action.content,
            "source": action.source,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TraceObservation]:
        """
        Parse server response into StepResult[TraceObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TraceObservation
        """
        obs_data = payload.get("observation", {})
        observation = TraceObservation(
            episode_id=obs_data.get("episode_id", ""),
            step=obs_data.get("step", 0),
            instruction=obs_data.get("instruction", ""),
            available_sources=obs_data.get("available_sources", []),
            context=obs_data.get("context", ""),
            memory_summary=obs_data.get("memory_summary", ""),
            world_state=obs_data.get("world_state", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
