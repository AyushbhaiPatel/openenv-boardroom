"""Unit tests for final reward aggregation."""

import pytest

from my_env.server.reward_calculator import RewardCalculator


def test_single_agent_weights_do_not_require_evidence_actions():
    calc = RewardCalculator()
    history = [
        {"action_type": "consult_stakeholder", "reward": 0.15, "difficulty": "medium"},
        {"action_type": "make_decision", "reward": 0.5, "difficulty": "medium"},
    ]

    assert calc.compute_final_score(history) == pytest.approx(0.34)


def test_multi_agent_present_evidence_has_dedicated_score_bucket():
    calc = RewardCalculator()
    without_evidence = [
        {"action_type": "consult_stakeholder", "reward": 0.15, "difficulty": "medium"},
        {"action_type": "negotiate", "reward": 0.25, "difficulty": "medium"},
        {"action_type": "make_decision", "reward": 0.5, "difficulty": "medium"},
    ]
    with_evidence = [
        *without_evidence[:-1],
        {"action_type": "present_evidence", "reward": 0.15, "difficulty": "medium"},
        without_evidence[-1],
    ]

    assert calc.compute_final_score(without_evidence) == pytest.approx(0.28)
    assert calc.compute_final_score(with_evidence) == pytest.approx(0.40)
