"""Tests for the BoardroomEnvironment orchestrator."""

import math

from my_env.models import BoardroomAction
from my_env.server.boardroom_environment import BoardroomEnvironment


class TestReset:
    def test_default_difficulty_is_medium(self):
        env = BoardroomEnvironment()
        obs = env.reset(seed=42)
        assert obs.metadata["difficulty"] == "medium"
        assert obs.metadata["max_steps"] == 20

    def test_seed_in_metadata(self):
        env = BoardroomEnvironment()
        obs = env.reset(seed=123)
        assert obs.metadata["seed"] == 123

    def test_auto_seed_when_none(self):
        env = BoardroomEnvironment()
        obs = env.reset()
        assert "seed" in obs.metadata
        assert isinstance(obs.metadata["seed"], int)

    def test_initial_observation_fields(self):
        env = BoardroomEnvironment()
        obs = env.reset(seed=0, difficulty="easy")
        assert obs.quarter == 1
        assert obs.step_count == 0
        assert obs.done is False
        assert obs.data_tables  # non-empty initial snapshot

    def test_each_difficulty_tier(self):
        env = BoardroomEnvironment()
        for diff, expected_steps in [("easy", 10), ("medium", 20), ("hard", 30)]:
            obs = env.reset(seed=0, difficulty=diff)
            assert obs.metadata["max_steps"] == expected_steps
            assert obs.metadata["difficulty"] == diff


class TestStep:
    def test_step_counter_increments(self):
        env = BoardroomEnvironment()
        env.reset(seed=42)
        obs = env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))
        assert obs.step_count == 1
        obs = env.step(BoardroomAction(action_type="query_data", parameters={"metric": "churn_rate"}))
        assert obs.step_count == 2

    def test_query_data_returns_metric(self):
        env = BoardroomEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs = env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))
        assert "revenue" in obs.data_tables

    def test_consult_stakeholder_returns_feedback(self):
        env = BoardroomEnvironment()
        env.reset(seed=42)
        obs = env.step(BoardroomAction(action_type="consult_stakeholder", parameters={"stakeholder": "analyst"}))
        assert obs.stakeholder_feedback is not None
        assert len(obs.stakeholder_feedback) > 0

    def test_simulate_counterfactual_returns_results(self):
        env = BoardroomEnvironment()
        env.reset(seed=42)
        obs = env.step(BoardroomAction(
            action_type="simulate_counterfactual",
            parameters={"decision": "increase_marketing", "parameters": {"budget": 50000}},
        ))
        assert obs.simulation_results is not None
        assert "projected_revenue_delta" in obs.simulation_results

    def test_make_decision_ends_episode(self):
        env = BoardroomEnvironment()
        env.reset(seed=42)
        obs = env.step(BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": "test",
                "parameters": {},
                "explanation": "Revenue data shows growth. Analyst agrees. Some uncertainty.",
            },
        ))
        assert obs.done is True
        assert "final_score" in obs.metadata
        assert 0.0 <= obs.metadata["final_score"] <= 1.0


class TestErrorHandling:
    def test_invalid_params_no_step_advance(self):
        env = BoardroomEnvironment()
        env.reset(seed=42)
        obs = env.step(BoardroomAction(action_type="query_data", parameters={}))
        assert obs.step_count == 0
        assert obs.reward == 0.0
        assert "error" in obs.metadata

    def test_step_after_done(self):
        env = BoardroomEnvironment()
        env.reset(seed=42)
        env.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "x", "parameters": {}, "explanation": "test"},
        ))
        obs = env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))
        assert "error" in obs.metadata
        assert "already done" in obs.metadata["error"].lower()

    def test_step_before_reset(self):
        env = BoardroomEnvironment()
        obs = env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))
        assert "error" in obs.metadata


class TestSeedDeterminism:
    def test_same_seed_same_output(self):
        env = BoardroomEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs1 = env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))

        env.reset(seed=42, difficulty="easy")
        obs2 = env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))

        assert obs1.data_tables == obs2.data_tables
        assert obs1.reward == obs2.reward


class TestAuditTrail:
    def test_audit_trail_in_final_observation(self):
        env = BoardroomEnvironment()
        env.reset(seed=42)
        env.step(BoardroomAction(action_type="query_data", parameters={"metric": "revenue"}))
        obs = env.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "x", "parameters": {}, "explanation": "test"},
        ))
        assert obs.done is True
        assert "audit_trail" in obs.metadata
        # Trail snapshot is taken before the final step is recorded,
        # so it contains entries from prior steps only.
        assert len(obs.metadata["audit_trail"]) >= 1


class TestMaxStepsTermination:
    def test_easy_terminates_at_10(self):
        env = BoardroomEnvironment()
        env.reset(seed=42, difficulty="easy")
        for i in range(10):
            obs = env.step(BoardroomAction(
                action_type="query_data", parameters={"metric": "revenue"},
            ))
        assert obs.done is True
        assert obs.step_count == 10
