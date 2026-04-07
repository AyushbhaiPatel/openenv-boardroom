"""Tests for the BoardroomEnvironment orchestrator."""

import math

from my_env.models import BoardroomAction
from my_env.server.boardroom_environment import BoardroomEnvironment
from my_env.server.data_generator import SyntheticDataGenerator


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
        assert "support_load" in obs.data_tables
        assert "release_risk" in obs.data_tables

    def test_each_difficulty_tier(self):
        env = BoardroomEnvironment()
        for diff, expected_steps in [("easy", 10), ("medium", 20), ("hard", 30)]:
            obs = env.reset(seed=0, difficulty=diff)
            assert obs.metadata["max_steps"] == expected_steps
            assert obs.metadata["difficulty"] == diff

    def test_reset_includes_scenario_brief(self):
        env = BoardroomEnvironment()
        obs = env.reset(seed=0, difficulty="hard")
        assert "brief" in obs.metadata
        assert "launch" in obs.metadata["brief"].lower()


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

    def test_trend_analysis_has_history_from_reset(self):
        env = BoardroomEnvironment()
        env.reset(seed=42, difficulty="medium")
        obs = env.step(BoardroomAction(
            action_type="analyze_trend",
            parameters={"metric": "revenue", "quarters": 4},
        ))
        assert len(obs.data_tables["trend"]) >= 4

    def test_reset_history_has_monotonic_quarters(self):
        env = BoardroomEnvironment()
        env.reset(seed=42, difficulty="medium")
        quarters = [entry["quarter"] for entry in env._company_state.history]
        assert quarters == sorted(quarters)


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


class TestDecisionDifficulty:
    def test_hard_task_penalizes_shallow_immediate_decision(self):
        env = BoardroomEnvironment()
        env.reset(seed=42, difficulty="hard")
        obs = env.step(BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": "launch feature x",
                "parameters": {},
                "explanation": "Launch now because growth matters.",
            },
        ))
        assert obs.done is True
        stronger_env = BoardroomEnvironment()
        stronger_env.reset(seed=42, difficulty="hard")
        for metric in ["churn_rate", "support_load", "release_risk"]:
            stronger_env.step(BoardroomAction(action_type="query_data", parameters={"metric": metric}))
        stronger = stronger_env.step(BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": "delay feature x launch",
                "parameters": {
                    "rollout_percentage": 10,
                    "support_headcount_delta": 4,
                    "rollback_plan": "Hold launch behind a feature flag and rollback within one hour if churn spikes.",
                },
                "explanation": (
                    "Support capacity and release risk are elevated, while churn is already rising. "
                    "The analyst and risk officer feedback suggest delaying broad launch until support capacity improves."
                ),
            },
        ))
        assert obs.metadata["final_score"] < stronger.metadata["final_score"]

    def test_hard_task_rewards_structured_launch_plan_fields(self):
        env = BoardroomEnvironment()
        env.reset(seed=1, difficulty="hard")
        for metric in ["churn_rate", "support_load", "release_risk"]:
            env.step(BoardroomAction(action_type="query_data", parameters={"metric": metric}))
        obs = env.step(BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": "delay feature x launch",
                "parameters": {
                    "rollout_percentage": 10,
                    "support_headcount_delta": 4,
                    "rollback_plan": "Hold launch behind a feature flag and rollback within one hour if churn spikes.",
                },
                "explanation": (
                    "Release risk and support capacity are both elevated, while churn is already rising. "
                    "The risk officer and analyst feedback suggest delaying broad launch until support capacity improves."
                ),
            },
        ))
        assert obs.done is True
        assert obs.metadata["final_score"] > 0.20

    def test_structured_launch_plan_scores_above_generic_launch_prose(self):
        generic_env = BoardroomEnvironment()
        generic_env.reset(seed=42, difficulty="hard")
        for metric in ["churn_rate", "support_load", "release_risk"]:
            generic_env.step(BoardroomAction(action_type="query_data", parameters={"metric": metric}))
        generic = generic_env.step(BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": "launch feature x",
                "parameters": {},
                "explanation": "We should launch because growth matters and the market is moving quickly.",
            },
        ))

        structured_env = BoardroomEnvironment()
        structured_env.reset(seed=1, difficulty="hard")
        for metric in ["churn_rate", "support_load", "release_risk"]:
            structured_env.step(BoardroomAction(action_type="query_data", parameters={"metric": metric}))
        structured = structured_env.step(BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": "delay feature x launch",
                "parameters": {
                    "rollout_percentage": 10,
                    "support_headcount_delta": 4,
                    "rollback_plan": "Hold launch behind a feature flag and rollback within one hour if churn spikes.",
                },
                "explanation": (
                    "Support capacity and release risk are elevated, while churn is already rising. "
                    "The analyst and risk officer feedback suggest delaying broad launch until support capacity improves."
                ),
            },
        ))

        assert structured.metadata["final_score"] > generic.metadata["final_score"]

    def test_easy_decision_alignment_accepts_oracle_aliases(self):
        env = BoardroomEnvironment()
        env.reset(seed=2, difficulty="easy")
        env._oracle_answer = "monthly_active_users"
        alias_score = env._score_decision_alignment(
            "focus on onboarding",
            "Monthly active users and MAU are the clearest bottlenecks.",
        )
        plain_score = env._score_decision_alignment(
            "focus on onboarding",
            "Revenue feels soft, but the story is unclear.",
        )
        assert alias_score > plain_score

    def test_hard_launch_alignment_penalizes_negative_launch_intent(self):
        env = BoardroomEnvironment()
        env.reset(seed=42, difficulty="hard")
        env._oracle_answer = "launch"
        negative_score = env._score_decision_alignment(
            "delay feature x launch",
            "Delay launch until release risk and support capacity improve.",
        )
        positive_score = env._score_decision_alignment(
            "launch feature x",
            "Launch now with support safeguards because capacity is stable.",
        )
        assert positive_score > negative_score

    def test_launch_readiness_does_not_reward_generic_support_language(self):
        env = BoardroomEnvironment()
        env.reset(seed=42, difficulty="hard")
        vague = env._score_launch_readiness(
            "launch feature x",
            {},
            "I support the plan and think the release should happen soon.",
        )
        specific = env._score_launch_readiness(
            "launch feature x",
            {},
            "Support capacity is stable, ticket backlog is low, and release risk is manageable.",
        )
        assert specific > vague

    def test_oracle_hit_uses_alias_and_negative_launch_logic(self):
        env = BoardroomEnvironment()
        env.reset(seed=42, difficulty="hard")
        env._oracle_answer = "do not launch"
        assert env._oracle_alignment_hit(
            "delay feature x launch",
            "Support capacity is overloaded, so postpone the launch.",
        ) is True

        env._oracle_answer = "launch"
        assert env._oracle_alignment_hit(
            "delay feature x launch",
            "Support capacity is overloaded, so postpone the launch.",
        ) is False

    def test_evidence_quality_counts_only_relevant_metrics(self):
        env = BoardroomEnvironment()
        env.reset(seed=42, difficulty="easy")
        env._queried_metrics = {"quarter", "support_load"}
        assert env._compute_evidence_quality() == 0.0


class TestDataEvolution:
    def test_evolve_state_updates_hard_mode_risk_signals(self):
        generator = SyntheticDataGenerator(seed=42)
        state = generator.generate_initial_state("hard", oracle_answer="do not launch")
        before_support = state.support_load
        before_risk = state.release_risk
        generator.evolve_state(state, decision_quality=0.9)
        assert state.support_load != before_support
        assert state.release_risk != before_risk
