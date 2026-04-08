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

        assert policy.next_action(9)["action_type"] == "simulate_counterfactual"
        final_action = policy.next_action(10)
        assert final_action["action_type"] == "make_decision"
        assert "explanation" in final_action["parameters"]

    def test_hard_plan_emits_structured_launch_fields(self):
        env = BoardroomEnvironment()
        obs = env.reset(seed=1, difficulty="hard")
        policy = ScenarioAwarePolicy(difficulty="hard", snapshot=obs.data_tables)

        final_action = policy.next_action(11)
        params = final_action["parameters"]["parameters"]
        assert final_action["action_type"] == "make_decision"
        assert "rollout_percentage" in params
        assert "support_headcount_delta" in params
        assert "rollback_plan" in params
