"""Property-based tests for MultiAgentBoardroomEnvironment (tasks 11.2–11.15)."""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from my_env.models import BoardroomAction
from my_env.server.multi_agent_boardroom_environment import (
    MultiAgentBoardroomEnvironment,
    _LOBBY_DELTA,
    _ALERT_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIFFICULTIES = ["easy", "medium", "hard"]

_SIMPLE_ACTIONS = [
    {"action_type": "query_data", "parameters": {"metric": "revenue"}},
    {"action_type": "query_data", "parameters": {"metric": "churn_rate"}},
    {"action_type": "analyze_trend", "parameters": {"metric": "revenue", "quarters": 3}},
    {"action_type": "consult_stakeholder", "parameters": {"stakeholder": "analyst"}},
]

_PRESENT_EVIDENCE_ACTIONS = [
    {"action_type": "present_evidence", "parameters": {
        "target": "cfo", "metric": "support_load", "value": 0.9,
        "interpretation": "support load risk data shows trend evidence"}},
    {"action_type": "present_evidence", "parameters": {
        "target": "ceo", "metric": "release_risk", "value": 0.8,
        "interpretation": "release risk data analysis indicates"}},
    {"action_type": "present_evidence", "parameters": {
        "target": "risk_officer", "metric": "support_load", "value": 0.85,
        "interpretation": "support load metric shows risk trend"}},
]

_NEGOTIATE_ACTIONS = [
    {"action_type": "negotiate", "parameters": {
        "target": "ceo", "position": "we should delay the launch due to risk"}},
    {"action_type": "negotiate", "parameters": {
        "target": "cfo", "position": "let's be cautious about spending"}},
]

_ALL_VALID_ACTIONS = _SIMPLE_ACTIONS + _PRESENT_EVIDENCE_ACTIONS + _NEGOTIATE_ACTIONS


def make_env(seed: int, difficulty: str) -> MultiAgentBoardroomEnvironment:
    env = MultiAgentBoardroomEnvironment()
    env.reset(seed=seed, difficulty=difficulty)
    return env


# ---------------------------------------------------------------------------
# Property 1: Actor state updates every step
# ---------------------------------------------------------------------------

class TestProperty1ActorStateUpdatesEveryStep:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        difficulty=st.sampled_from(DIFFICULTIES),
        action_idx=st.integers(min_value=0, max_value=len(_SIMPLE_ACTIONS) - 1),
    )
    @settings(max_examples=100)
    def test_actor_messages_present_after_any_step(self, seed, difficulty, action_idx):
        env = make_env(seed, difficulty)
        action_dict = _SIMPLE_ACTIONS[action_idx]
        obs = env.step(BoardroomAction(
            action_type=action_dict["action_type"],
            parameters=action_dict["parameters"],
        ))
        assert "actor_messages" in obs.metadata
        assert set(obs.metadata["actor_messages"].keys()) == {"ceo", "cfo", "risk_officer"}

    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        difficulty=st.sampled_from(DIFFICULTIES),
    )
    @settings(max_examples=100)
    def test_cfo_lobby_pressure_changes_after_non_ceo_action(self, seed, difficulty):
        env = make_env(seed, difficulty)
        before = env._actor_states["cfo"].lobby_pressure
        env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))
        after = env._actor_states["cfo"].lobby_pressure
        # CEO lobbies every step when CDO doesn't target CEO
        assert after >= before


# ---------------------------------------------------------------------------
# Property 2: Reset produces clean actor state
# ---------------------------------------------------------------------------

class TestProperty2ResetProducesCleanActorState:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        difficulty=st.sampled_from(DIFFICULTIES),
        n_steps=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_reset_clears_actor_state(self, seed, difficulty, n_steps):
        env = make_env(seed, difficulty)
        # Take some steps
        for _ in range(n_steps):
            if env._done:
                break
            env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))
        # Reset with same seed/difficulty
        env.reset(seed=seed, difficulty=difficulty)
        assert env._actor_states["cfo"].lobby_pressure == 0.0
        assert env._actor_states["cfo"].evidence_weight == 0.0
        assert env._actor_states["risk_officer"].alert_active is False
        assert env._board_vote_count == 0
        assert env._cfo_flip_rewarded is False


# ---------------------------------------------------------------------------
# Property 3: Lobby pressure accumulates when CEO not targeted
# ---------------------------------------------------------------------------

class TestProperty3LobbyPressureAccumulates:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        difficulty=st.sampled_from(DIFFICULTIES),
        initial_pressure=st.floats(min_value=0.0, max_value=0.9),
    )
    @settings(max_examples=100)
    def test_lobby_pressure_increases_on_non_ceo_action(self, seed, difficulty, initial_pressure):
        env = make_env(seed, difficulty)
        env._actor_states["cfo"].lobby_pressure = initial_pressure
        before = env._actor_states["cfo"].lobby_pressure
        env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))
        after = env._actor_states["cfo"].lobby_pressure
        delta = _LOBBY_DELTA[difficulty]
        expected = min(1.0, before + delta)
        assert after == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Property 4: CFO stance determined by evidence_weight vs lobby_pressure
# ---------------------------------------------------------------------------

class TestProperty4CfoStanceDeterminism:
    @given(
        evidence_weight=st.floats(min_value=0.0, max_value=1.0),
        lobby_pressure=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_cfo_stance_evidence_based_when_ew_gt_lp(self, evidence_weight, lobby_pressure):
        assume(evidence_weight > lobby_pressure)
        env = make_env(42, "medium")
        env._actor_states["cfo"].evidence_weight = evidence_weight
        env._actor_states["cfo"].lobby_pressure = lobby_pressure
        env._update_cfo_stance()
        assert env._actor_states["cfo"].stance == "evidence_based"

    @given(
        evidence_weight=st.floats(min_value=0.0, max_value=0.5),
        lobby_pressure=st.floats(min_value=0.51, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_cfo_stance_pro_launch_when_lp_gt_05_and_ew_lte_lp(self, evidence_weight, lobby_pressure):
        assume(evidence_weight <= lobby_pressure)
        env = make_env(42, "medium")
        env._actor_states["cfo"].evidence_weight = evidence_weight
        env._actor_states["cfo"].lobby_pressure = lobby_pressure
        env._update_cfo_stance()
        assert env._actor_states["cfo"].stance == "pro_launch"


# ---------------------------------------------------------------------------
# Property 5: CFO flip reward awarded at most once
# ---------------------------------------------------------------------------

class TestProperty5CfoFlipRewardAtMostOnce:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        n_presents=st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=100)
    def test_cfo_flip_reward_at_most_once(self, seed, n_presents):
        env = make_env(seed, "medium")
        flip_rewards = 0
        for _ in range(n_presents):
            if env._done:
                break
            reward = env._update_cfo_stance()
            if reward == pytest.approx(0.2):
                flip_rewards += 1
            # Simulate evidence increasing
            env._actor_states["cfo"].evidence_weight = min(1.0, env._actor_states["cfo"].evidence_weight + 0.3)
            env._actor_states["cfo"].lobby_pressure = max(0.0, env._actor_states["cfo"].lobby_pressure - 0.1)
        assert flip_rewards <= 1


# ---------------------------------------------------------------------------
# Property 6: present_evidence increases evidence_weight within bounds
# ---------------------------------------------------------------------------

class TestProperty6PresentEvidenceIncreasesWeight:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        difficulty=st.sampled_from(DIFFICULTIES),
        target=st.sampled_from(["ceo", "cfo", "risk_officer"]),
        metric=st.sampled_from(["support_load", "release_risk", "revenue", "churn_rate"]),
    )
    @settings(max_examples=100)
    def test_evidence_weight_increases_in_bounds(self, seed, difficulty, target, metric):
        env = make_env(seed, difficulty)
        before = env._actor_states[target].evidence_weight
        env.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": target, "metric": metric, "value": 0.5,
                        "interpretation": "data shows risk trend evidence analysis"},
        ))
        after = env._actor_states[target].evidence_weight
        delta = after - before
        assert 0.1 <= delta <= 0.4
        assert after <= 1.0


# ---------------------------------------------------------------------------
# Property 7: negotiate reduces lobby_pressure by 0.1
# ---------------------------------------------------------------------------

class TestProperty7NegotiateReducesLobbyPressure:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        initial_pressure=st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_negotiate_reduces_lobby_pressure_by_01(self, seed, initial_pressure):
        env = make_env(seed, "hard")
        env._actor_states["cfo"].lobby_pressure = initial_pressure
        env.step(BoardroomAction(
            action_type="negotiate",
            parameters={"target": "ceo", "position": "we should delay the launch due to risk"},
        ))
        after = env._actor_states["cfo"].lobby_pressure
        expected = max(0.0, initial_pressure - 0.1)
        assert after == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Property 8: Invalid targets return error with zero reward
# ---------------------------------------------------------------------------

class TestProperty8InvalidTargetsReturnError:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        invalid_target=st.text(min_size=1, max_size=20).filter(
            lambda t: t not in ("ceo", "cfo", "risk_officer")
        ),
        action_type=st.sampled_from(["present_evidence", "negotiate"]),
    )
    @settings(max_examples=100)
    def test_invalid_target_returns_error_zero_reward(self, seed, invalid_target, action_type):
        env = make_env(seed, "medium")
        if action_type == "present_evidence":
            params = {"target": invalid_target, "metric": "revenue", "value": 0.5, "interpretation": "test"}
        else:
            params = {"target": invalid_target, "position": "test position"}
        obs = env.step(BoardroomAction(action_type=action_type, parameters=params))
        assert "error" in obs.metadata
        assert obs.reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Property 9: Risk Officer emits alert when threshold exceeded
# ---------------------------------------------------------------------------

class TestProperty9RiskOfficerAlertWhenThresholdExceeded:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        difficulty=st.sampled_from(DIFFICULTIES),
    )
    @settings(max_examples=100)
    def test_risk_officer_alert_when_support_load_high(self, seed, difficulty):
        env = make_env(seed, difficulty)
        threshold = _ALERT_THRESHOLD[difficulty]
        # Force support_load above threshold
        env._company_state.support_load = threshold + 0.05
        obs = env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))
        risk_msg = obs.metadata["actor_messages"]["risk_officer"]
        assert len(risk_msg) > 0
        assert "ALERT" in risk_msg or "WARNING" in risk_msg

    def test_risk_officer_alert_latches_for_episode(self):
        env = make_env(42, "hard")
        threshold = _ALERT_THRESHOLD["hard"]
        env._company_state.support_load = threshold + 0.05
        env._step_risk_officer()
        assert env._actor_states["risk_officer"].alert_active is True

        env._company_state.support_load = threshold - 0.2
        env._step_risk_officer()
        assert env._actor_states["risk_officer"].alert_active is True
        assert env._actor_states["risk_officer"].stance == "alerting"


# ---------------------------------------------------------------------------
# Property 10: Board vote contains all three actors
# ---------------------------------------------------------------------------

class TestProperty10BoardVoteContainsAllActors:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        difficulty=st.sampled_from(DIFFICULTIES),
    )
    @settings(max_examples=100)
    def test_board_vote_has_all_three_actors(self, seed, difficulty):
        env = make_env(seed, difficulty)
        obs = env.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "reduce churn", "parameters": {}, "explanation": "churn is the bottleneck"},
        ))
        assert "board_vote" in obs.metadata
        vote = obs.metadata["board_vote"]
        assert set(vote.keys()) == {"ceo", "cfo", "risk_officer"}
        assert all(v in ("approve", "reject") for v in vote.values())

    def test_pro_launch_cfo_approves_launch_decision(self):
        env = make_env(42, "hard")
        cfo = env._actor_states["cfo"]
        cfo.stance = "pro_launch"
        cfo.budget_utilisation = 0.9
        cfo.lobby_pressure = 0.9

        assert env._cfo_vote("launch feature x") == "approve"
        assert env._cfo_vote("delay feature x") == "reject"


# ---------------------------------------------------------------------------
# Property 11: Majority resolves; rejection grants revision
# ---------------------------------------------------------------------------

class TestProperty11MajorityResolvesRejectionGrantsRevision:
    def test_majority_resolves_episode(self):
        env = make_env(42, "easy")
        # Easy: no alert, low lobby — CFO and CEO should approve
        env._actor_states["cfo"].lobby_pressure = 0.0
        env._actor_states["cfo"].budget_utilisation = 0.3
        env._actor_states["risk_officer"].alert_active = False
        obs = env.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "reduce churn", "parameters": {}, "explanation": "churn is the bottleneck"},
        ))
        approvals = sum(1 for v in obs.metadata["board_vote"].values() if v == "approve")
        if approvals >= 2:
            assert obs.metadata["vote_result"] == "approved"
            assert obs.done is True

    def test_rejection_does_not_resolve(self):
        env = make_env(42, "hard")
        env._actor_states["ceo"].hidden_agenda = "launch"
        env._actor_states["cfo"].stance = "pro_launch"
        env._actor_states["cfo"].lobby_pressure = 0.9
        env._actor_states["risk_officer"].alert_active = True
        obs = env.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "delay launch", "parameters": {}, "explanation": "too risky"},
        ))
        if obs.metadata.get("vote_result") == "rejected":
            assert obs.done is False


# ---------------------------------------------------------------------------
# Property 12: Third make_decision returns error with -0.1 penalty
# ---------------------------------------------------------------------------

class TestProperty12ThirdMakeDecisionReturnsError:
    @given(seed=st.integers(min_value=0, max_value=2**16))
    @settings(max_examples=100)
    def test_third_make_decision_error(self, seed):
        env = make_env(seed, "easy")
        decision_params = {"decision": "reduce churn", "parameters": {}, "explanation": "churn is the bottleneck"}
        env.step(BoardroomAction(action_type="make_decision", parameters=decision_params))
        env._done = False
        env._scenario_resolved = False
        env.step(BoardroomAction(action_type="make_decision", parameters=decision_params))
        env._done = False
        obs3 = env.step(BoardroomAction(action_type="make_decision", parameters=decision_params))
        assert "error" in obs3.metadata
        assert obs3.reward == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Property 13: Final score strictly in (0.0, 1.0)
# ---------------------------------------------------------------------------

class TestProperty13FinalScoreInBounds:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        difficulty=st.sampled_from(DIFFICULTIES),
    )
    @settings(max_examples=100)
    def test_final_score_strictly_between_0_and_1(self, seed, difficulty):
        env = make_env(seed, difficulty)
        obs = env.step(BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": "reduce churn rate",
                "parameters": {"priority": "retention"},
                "explanation": "churn rate is the primary bottleneck based on data analysis",
            },
        ))
        assert obs.done is True
        final_score = obs.metadata.get("final_score", obs.reward)
        assert 0.0 < final_score < 1.0


# ---------------------------------------------------------------------------
# Property 14: Difficulty configuration applied correctly
# ---------------------------------------------------------------------------

class TestProperty14DifficultyConfigApplied:
    @given(
        seed=st.integers(min_value=0, max_value=2**16),
        difficulty=st.sampled_from(DIFFICULTIES),
    )
    @settings(max_examples=100)
    def test_lobby_delta_matches_difficulty(self, seed, difficulty):
        env = make_env(seed, difficulty)
        before = env._actor_states["cfo"].lobby_pressure
        env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))
        after = env._actor_states["cfo"].lobby_pressure
        expected_delta = _LOBBY_DELTA[difficulty]
        assert after == pytest.approx(min(1.0, before + expected_delta), abs=1e-9)

    @given(seed=st.integers(min_value=0, max_value=2**16))
    @settings(max_examples=100)
    def test_hard_has_hidden_metrics(self, seed):
        env = make_env(seed, "hard")
        assert len(env._actor_states["ceo"].hidden_metrics) > 0

    @given(seed=st.integers(min_value=0, max_value=2**16))
    @settings(max_examples=100)
    def test_easy_has_no_hidden_metrics(self, seed):
        env = make_env(seed, "easy")
        assert len(env._actor_states["ceo"].hidden_metrics) == 0

    @given(seed=st.integers(min_value=0, max_value=2**16))
    @settings(max_examples=100)
    def test_hard_hidden_agenda_is_launch(self, seed):
        env = make_env(seed, "hard")
        assert env._actor_states["ceo"].hidden_agenda == "launch"
