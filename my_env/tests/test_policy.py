"""Tests for the shared scenario-aware policy."""

from my_env.policy import ScenarioAwarePolicy
from my_env.server.boardroom_environment import BoardroomEnvironment


class TestScenarioAwarePolicy:
    def test_policy_predicts_oracle_across_seed_samples(self):
        env = BoardroomEnvironment()

        for difficulty in ("easy", "medium", "hard"):
            for seed in range(15):
                obs = env.reset(seed=seed, difficulty=difficulty)
                policy = ScenarioAwarePolicy(difficulty=difficulty, snapshot=obs.data_tables)
                assert policy.plan.oracle_prediction == env._oracle_answer

    def test_easy_plan_finishes_with_simulation_then_decision(self):
        env = BoardroomEnvironment()
        obs = env.reset(seed=0, difficulty="easy")
        policy = ScenarioAwarePolicy(difficulty="easy", snapshot=obs.data_tables)

        # Easy plan: 4 query_data + 1 trend + 1 present_evidence + 3 stakeholders + 1 simulation + 1 decision
        # Find simulate_counterfactual and make_decision by scanning
        actions = [policy.next_action(s) for s in range(1, 15)]
        action_types = [a["action_type"] for a in actions]
        assert "simulate_counterfactual" in action_types
        sim_idx = action_types.index("simulate_counterfactual")
        assert action_types[sim_idx + 1] == "make_decision"
        final_action = actions[sim_idx + 1]
        assert "explanation" in final_action["parameters"]

    def test_hard_plan_emits_structured_launch_fields(self):
        env = BoardroomEnvironment()
        obs = env.reset(seed=1, difficulty="hard")
        policy = ScenarioAwarePolicy(difficulty="hard", snapshot=obs.data_tables)

        # Find make_decision by scanning steps
        for step in range(1, 20):
            action = policy.next_action(step)
            if action["action_type"] == "make_decision":
                params = action["parameters"]["parameters"]
                assert "rollout_percentage" in params
                assert "support_headcount_delta" in params
                assert "rollback_plan" in params
                return
        raise AssertionError("make_decision not found in first 20 steps")
