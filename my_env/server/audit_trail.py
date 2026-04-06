# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Audit trail logger for the OpenBoardroom Environment.

Records every action, observation, and reward during an episode
for replay and debugging purposes.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from my_env.models import AuditEntry, BoardroomAction, BoardroomObservation


class AuditTrail:
    """Chronological log of all episode events.

    Records each step as an AuditEntry with step number, quarter,
    action type, action parameters, reward, and ISO 8601 timestamp.
    Memory is bounded by max_steps (max 30 entries).
    """

    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []

    def record(
        self,
        step: int,
        quarter: int,
        action: BoardroomAction,
        observation: BoardroomObservation,
        reward: float,
    ) -> None:
        """Append an AuditEntry for the given step."""
        entry = AuditEntry(
            step=step,
            quarter=quarter,
            action_type=action.action_type,
            action_params=dict(action.parameters),
            reward=reward,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._entries.append(entry)

    def get_trail(self) -> List[Dict[str, Any]]:
        """Return all entries as a list of dicts in chronological order."""
        return [
            {
                "step": e.step,
                "quarter": e.quarter,
                "action_type": e.action_type,
                "action_params": e.action_params,
                "reward": e.reward,
                "timestamp": e.timestamp,
            }
            for e in self._entries
        ]

    def clear(self) -> None:
        """Reset the trail, removing all entries."""
        self._entries.clear()
