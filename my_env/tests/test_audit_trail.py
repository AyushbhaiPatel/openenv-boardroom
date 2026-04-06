# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the AuditTrail module."""

import re
from datetime import datetime

from my_env.models import BoardroomAction, BoardroomObservation
from my_env.server.audit_trail import AuditTrail

# ISO 8601 pattern (e.g. 2024-01-15T12:30:00+00:00)
ISO_8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _make_action(action_type: str = "query_data", **params) -> BoardroomAction:
    return BoardroomAction(action_type=action_type, parameters=params)


def _make_observation(**kwargs) -> BoardroomObservation:
    defaults = {"done": False, "reward": 0.0, "metadata": {}}
    defaults.update(kwargs)
    return BoardroomObservation(**defaults)


class TestAuditTrailRecord:
    def test_record_appends_entry(self):
        trail = AuditTrail()
        action = _make_action("query_data", metric="revenue")
        obs = _make_observation()

        trail.record(step=1, quarter=1, action=action, observation=obs, reward=0.2)

        entries = trail.get_trail()
        assert len(entries) == 1
        assert entries[0]["step"] == 1
        assert entries[0]["quarter"] == 1
        assert entries[0]["action_type"] == "query_data"
        assert entries[0]["action_params"] == {"metric": "revenue"}
        assert entries[0]["reward"] == 0.2
        assert ISO_8601_RE.match(entries[0]["timestamp"])

    def test_record_multiple_entries_chronological(self):
        trail = AuditTrail()
        for i in range(5):
            action = _make_action("analyze_trend", metric="churn", quarters=i + 1)
            obs = _make_observation()
            trail.record(step=i + 1, quarter=1, action=action, observation=obs, reward=0.25)

        entries = trail.get_trail()
        assert len(entries) == 5
        steps = [e["step"] for e in entries]
        assert steps == [1, 2, 3, 4, 5]

    def test_record_preserves_action_params(self):
        trail = AuditTrail()
        action = _make_action(
            "make_decision",
            decision="increase_ad_spend",
            parameters={"amount": 50000},
            explanation="Based on data analysis",
        )
        obs = _make_observation()
        trail.record(step=1, quarter=2, action=action, observation=obs, reward=0.4)

        entry = trail.get_trail()[0]
        assert entry["action_params"]["decision"] == "increase_ad_spend"
        assert entry["action_params"]["explanation"] == "Based on data analysis"

    def test_timestamps_are_iso_8601(self):
        trail = AuditTrail()
        action = _make_action()
        obs = _make_observation()
        trail.record(step=1, quarter=1, action=action, observation=obs, reward=0.1)

        ts = trail.get_trail()[0]["timestamp"]
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None


class TestAuditTrailGetTrail:
    def test_empty_trail(self):
        trail = AuditTrail()
        assert trail.get_trail() == []

    def test_returns_list_of_dicts(self):
        trail = AuditTrail()
        action = _make_action()
        obs = _make_observation()
        trail.record(step=1, quarter=1, action=action, observation=obs, reward=0.2)

        entries = trail.get_trail()
        assert isinstance(entries, list)
        assert isinstance(entries[0], dict)
        expected_keys = {"step", "quarter", "action_type", "action_params", "reward", "timestamp"}
        assert set(entries[0].keys()) == expected_keys


class TestAuditTrailClear:
    def test_clear_empties_trail(self):
        trail = AuditTrail()
        action = _make_action()
        obs = _make_observation()
        for i in range(3):
            trail.record(step=i + 1, quarter=1, action=action, observation=obs, reward=0.1)

        assert len(trail.get_trail()) == 3
        trail.clear()
        assert trail.get_trail() == []

    def test_clear_allows_new_records(self):
        trail = AuditTrail()
        action = _make_action()
        obs = _make_observation()
        trail.record(step=1, quarter=1, action=action, observation=obs, reward=0.1)
        trail.clear()
        trail.record(step=1, quarter=2, action=action, observation=obs, reward=0.3)

        entries = trail.get_trail()
        assert len(entries) == 1
        assert entries[0]["quarter"] == 2
