"""Unit tests for MultiAgentRewardCalculator."""

import pytest
from my_env.server.multi_agent_reward_calculator import MultiAgentRewardCalculator


@pytest.fixture
def calc():
    return MultiAgentRewardCalculator()


# --- compute_present_evidence_reward ---

class TestComputePresentEvidenceReward:
    def test_zero_delta(self, calc):
        assert calc.compute_present_evidence_reward(None, 0.0) == pytest.approx(0.05)

    def test_half_delta(self, calc):
        assert calc.compute_present_evidence_reward(None, 0.5) == pytest.approx(0.10)

    def test_full_delta(self, calc):
        assert calc.compute_present_evidence_reward(None, 1.0) == pytest.approx(0.15)

    def test_clamp_upper(self, calc):
        # delta > 1.0 should clamp to 0.15
        assert calc.compute_present_evidence_reward(None, 2.0) == pytest.approx(0.15)

    def test_clamp_lower(self, calc):
        # negative delta should clamp to 0.05
        assert calc.compute_present_evidence_reward(None, -5.0) == pytest.approx(0.05)


# --- compute_negotiate_reward ---

class TestComputeNegotiateReward:
    def test_zero_reduction(self, calc):
        assert calc.compute_negotiate_reward(0.0) == pytest.approx(0.0)

    def test_small_positive_reduction(self, calc):
        assert calc.compute_negotiate_reward(0.1) == pytest.approx(0.05)

    def test_large_positive_reduction(self, calc):
        assert calc.compute_negotiate_reward(1.0) == pytest.approx(0.05)

    def test_negative_reduction(self, calc):
        assert calc.compute_negotiate_reward(-0.5) == pytest.approx(0.0)


# --- compute_board_vote_reward ---

class TestComputeBoardVoteReward:
    def test_zero_approvals(self, calc):
        assert calc.compute_board_vote_reward(0) == pytest.approx(-0.2)

    def test_one_approval(self, calc):
        assert calc.compute_board_vote_reward(1) == pytest.approx(-0.2)

    def test_two_approvals(self, calc):
        assert calc.compute_board_vote_reward(2) == pytest.approx(0.2)

    def test_three_approvals(self, calc):
        assert calc.compute_board_vote_reward(3) == pytest.approx(0.2)


# --- compute_risk_alert_reward ---

class TestComputeRiskAlertReward:
    def test_addressed_and_alert_active(self, calc):
        assert calc.compute_risk_alert_reward(True, True) == pytest.approx(0.15)

    def test_not_addressed_alert_active(self, calc):
        assert calc.compute_risk_alert_reward(False, True) == pytest.approx(-0.15)

    def test_not_addressed_no_alert(self, calc):
        assert calc.compute_risk_alert_reward(False, False) == pytest.approx(0.0)

    def test_addressed_no_alert(self, calc):
        # addressed takes priority regardless of alert state
        assert calc.compute_risk_alert_reward(True, False) == pytest.approx(0.15)


# --- compute_hidden_metric_reveal_reward ---

class TestComputeHiddenMetricRevealReward:
    def test_revealed(self, calc):
        assert calc.compute_hidden_metric_reveal_reward(True) == pytest.approx(0.3)

    def test_not_revealed(self, calc):
        assert calc.compute_hidden_metric_reveal_reward(False) == pytest.approx(0.0)
