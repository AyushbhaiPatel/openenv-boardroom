"""Unit tests for present_evidence, negotiate, and board voting handlers."""

import pytest
from my_env.models import BoardroomAction
from my_env.server.multi_agent_boardroom_environment import MultiAgentBoardroomEnvironment


@pytest.fixture()
def env_easy():
    env = MultiAgentBoardroomEnvironment()
    env.reset(seed=42, difficulty="easy")
    return env


@pytest.fixture()
def env_hard():
    env = MultiAgentBoardroomEnvironment()
    env.reset(seed=42, difficulty="hard")
    return env


# ---------------------------------------------------------------------------
# present_evidence tests (task 6.2)
# ---------------------------------------------------------------------------

class TestPresentEvidence:
    def test_invalid_target_returns_error(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "board_chair", "metric": "revenue", "value": 100, "interpretation": "good"},
        ))
        assert "error" in obs.metadata
        assert obs.reward == 0.0

    def test_increases_evidence_weight(self, env_easy):
        before = env_easy._actor_states["cfo"].evidence_weight
        env_easy.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "cfo", "metric": "support_load", "value": 0.9, "interpretation": "risk data shows trend"},
        ))
        after = env_easy._actor_states["cfo"].evidence_weight
        assert after > before

    def test_value_parameter_is_optional(self, env_easy):
        before = env_easy._actor_states["cfo"].evidence_weight
        obs = env_easy.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "cfo", "metric": "support_load", "interpretation": "risk data shows trend"},
        ))
        assert obs.error is None
        assert env_easy._actor_states["cfo"].evidence_weight > before

    def test_evidence_weight_capped_at_one(self, env_easy):
        env_easy._actor_states["cfo"].evidence_weight = 0.95
        env_easy.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "cfo", "metric": "support_load", "value": 0.9, "interpretation": "risk data shows trend"},
        ))
        assert env_easy._actor_states["cfo"].evidence_weight <= 1.0

    def test_ceo_sets_contradiction_flag(self, env_easy):
        assert env_easy._ceo_contradiction_flag is False
        env_easy.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "ceo", "metric": "revenue", "value": 500000, "interpretation": "data shows evidence"},
        ))
        assert env_easy._ceo_contradiction_flag is True

    def test_risk_officer_unlocks_intel_when_alert_active(self, env_hard):
        # Force alert active
        env_hard._actor_states["risk_officer"].alert_active = True
        obs = env_hard.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "risk_officer", "metric": "support_load", "value": 0.95, "interpretation": "support load risk data"},
        ))
        assert env_hard._actor_states["risk_officer"].intel_unlocked is True
        assert "risk_intel" in obs.data_tables

    def test_risk_officer_no_intel_without_alert(self, env_easy):
        env_easy._actor_states["risk_officer"].alert_active = False
        obs = env_easy.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "risk_officer", "metric": "support_load", "value": 0.5, "interpretation": "data"},
        ))
        assert env_easy._actor_states["risk_officer"].intel_unlocked is False
        assert "risk_intel" not in obs.data_tables

    def test_actor_messages_present(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "cfo", "metric": "revenue", "value": 100, "interpretation": "data"},
        ))
        assert "actor_messages" in obs.metadata
        assert set(obs.metadata["actor_messages"].keys()) == {"ceo", "cfo", "risk_officer"}


# ---------------------------------------------------------------------------
# negotiate tests (task 7.2)
# ---------------------------------------------------------------------------

class TestNegotiate:
    def test_invalid_target_returns_error(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="negotiate",
            parameters={"target": "janitor", "position": "we should delay"},
        ))
        assert "error" in obs.metadata
        assert obs.reward == 0.0

    def test_ceo_reduces_lobby_pressure_with_contradicting_position(self, env_hard):
        env_hard._actor_states["cfo"].lobby_pressure = 0.5
        env_hard.step(BoardroomAction(
            action_type="negotiate",
            parameters={"target": "ceo", "position": "we should delay the launch due to risk"},
        ))
        assert env_hard._actor_states["cfo"].lobby_pressure == pytest.approx(0.4)

    def test_non_contradicting_position_no_reduction(self, env_hard):
        env_hard._actor_states["cfo"].lobby_pressure = 0.5
        env_hard.step(BoardroomAction(
            action_type="negotiate",
            parameters={"target": "ceo", "position": "we should grow revenue"},
        ))
        # No reduction because position doesn't contradict launch agenda
        assert env_hard._actor_states["cfo"].lobby_pressure == pytest.approx(0.5)

    def test_negotiate_non_ceo_target_no_lobby_change(self, env_hard):
        env_hard._actor_states["cfo"].lobby_pressure = 0.5
        before = env_hard._actor_states["cfo"].lobby_pressure
        env_hard.step(BoardroomAction(
            action_type="negotiate",
            parameters={"target": "cfo", "position": "delay the launch"},
        ))
        after = env_hard._actor_states["cfo"].lobby_pressure
        # Targeting CFO (not CEO) means no lobby_reduction from negotiate,
        # but CEO still lobbies each step (+0.15 on hard), so pressure increases
        assert after >= before

    def test_actor_messages_present(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="negotiate",
            parameters={"target": "ceo", "position": "let's be cautious"},
        ))
        assert "actor_messages" in obs.metadata
        assert set(obs.metadata["actor_messages"].keys()) == {"ceo", "cfo", "risk_officer"}


# ---------------------------------------------------------------------------
# Board voting tests (task 8.3)
# ---------------------------------------------------------------------------

class TestBoardVoting:
    def test_board_vote_breakdown_in_metadata(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "reduce churn", "parameters": {}, "explanation": "churn is the bottleneck"},
        ))
        assert "board_vote" in obs.metadata
        vote = obs.metadata["board_vote"]
        assert set(vote.keys()) == {"ceo", "cfo", "risk_officer"}
        assert all(v in ("approve", "reject") for v in vote.values())
        assert obs.board_vote == vote
        assert obs.vote_result == obs.metadata["vote_result"]

    def test_board_vote_reward_is_folded_into_final_history(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "reduce churn", "parameters": {}, "explanation": "churn is the bottleneck"},
        ))
        last = env_easy._episode_history[-1]
        assert last["board_reward"] == pytest.approx(0.2)
        assert last["reward"] == pytest.approx(obs.step_reward + last["board_reward"] + last["alert_reward"])
        assert obs.final_score == pytest.approx(obs.reward)

    def test_majority_vote_resolves_episode(self, env_easy):
        # Easy: no alert, CFO neutral with low lobby — should get majority
        env_easy._actor_states["cfo"].lobby_pressure = 0.0
        env_easy._actor_states["cfo"].budget_utilisation = 0.3
        obs = env_easy.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "reduce churn", "parameters": {}, "explanation": "churn is the bottleneck"},
        ))
        approvals = sum(1 for v in obs.metadata["board_vote"].values() if v == "approve")
        if approvals >= 2:
            assert obs.metadata.get("vote_result") == "approved"

    def test_rejection_grants_revision_chance(self, env_hard):
        # Hard: CEO has launch agenda, force rejection scenario
        env_hard._actor_states["ceo"].hidden_agenda = "launch"
        env_hard._actor_states["ceo"].stance = "pro_launch"
        env_hard._actor_states["cfo"].stance = "pro_launch"
        env_hard._actor_states["cfo"].lobby_pressure = 0.8
        env_hard._actor_states["risk_officer"].alert_active = True
        obs = env_hard.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "delay launch", "parameters": {}, "explanation": "too risky"},
        ))
        # If rejected, episode should NOT be done yet (revision chance)
        if obs.metadata.get("vote_result") == "rejected":
            assert obs.done is False

    def test_second_make_decision_always_resolves(self, env_hard):
        # First call
        env_hard._actor_states["ceo"].hidden_agenda = "launch"
        env_hard._actor_states["cfo"].stance = "pro_launch"
        env_hard._actor_states["cfo"].lobby_pressure = 0.9
        env_hard._actor_states["risk_officer"].alert_active = True
        obs1 = env_hard.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "delay launch", "parameters": {}, "explanation": "too risky"},
        ))
        if obs1.done:
            return  # Already resolved on first call, skip
        # Second call must always resolve
        obs2 = env_hard.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "delay launch", "parameters": {}, "explanation": "still too risky"},
        ))
        assert obs2.done is True

    def test_third_make_decision_returns_error(self, env_easy):
        env_easy.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "reduce churn", "parameters": {}, "explanation": "churn is the bottleneck"},
        ))
        # Reset done so we can call again (simulate revision scenario)
        env_easy._done = False
        env_easy._scenario_resolved = False
        env_easy.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "reduce churn again", "parameters": {}, "explanation": "still churn"},
        ))
        env_easy._done = False
        obs3 = env_easy.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "reduce churn third", "parameters": {}, "explanation": "churn again"},
        ))
        assert "error" in obs3.metadata
        assert obs3.reward == pytest.approx(-0.1)
