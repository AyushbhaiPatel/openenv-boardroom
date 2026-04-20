# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenBoardroom Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import BoardroomAction, BoardroomObservation


class BoardroomEnv(
    EnvClient[BoardroomAction, BoardroomObservation, State]
):
    """
    Client for the OpenBoardroom Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with BoardroomEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.quarter)
        ...
        ...     action = BoardroomAction(
        ...         action_type="query_data",
        ...         parameters={"metric": "revenue"},
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.data_tables)
    """

    def _step_payload(self, action: BoardroomAction) -> Dict[str, Any]:
        """
        Convert BoardroomAction to JSON payload for step message.

        Args:
            action: BoardroomAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[BoardroomObservation]:
        """
        Parse server response into StepResult[BoardroomObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with BoardroomObservation
        """
        obs_data = dict(payload.get("observation", {}))
        metadata = dict(obs_data.pop("metadata", {}) or {})
        for key in (
            "objective",
            "max_steps",
            "difficulty",
            "seed",
            "brief",
            "step_reward",
            "final_score",
            "oracle_answer",
            "oracle_hit",
            "audit_trail",
            "error",
            "actor_messages",
            "board_vote",
            "vote_result",
        ):
            value = obs_data.get(key)
            if value not in (None, {}, []):
                metadata.setdefault(key, value)

        observation = BoardroomObservation(
            **obs_data,
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=metadata,
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
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
