"""Integration tests for MultiAgentBoardroomEnvironment (tasks 15.1–15.4)."""

import pytest
from my_env.models import BoardroomAction, BoardroomObservation
from my_env.server.multi_agent_boardroom_environment import MultiAgentBoardroomEnvironment
from my_env.policy import ScenarioAwarePolicy


# ---------------------------------------------------------------------------
# Task 15.1: Full easy episode with multi-agent environment
# ---------------------------------------------------------------------------

class TestFullEasyEpisodeWithMultiAgent:
    def test_full_easy_episode_final_score_positive(self):
        env = MultiAgentBoardroomEnvironment()
        obs = env.reset(seed=42, difficulty="easy")
        policy = ScenarioAwarePolicy(difficulty="easy", snapshot=obs.data_tables, multi_agent=True)
        max_steps = int(obs.metadata.get("max_steps", 10))

        for step in range(1, max_steps + 1):
            if obs.done:
                break
            action_dict = policy.next_action(step)
            action = BoardroomAction(
                action_type=action_dict["action_type"],
                parameters=action_dict["parameters"],
            )
            obs = env.step(action)

        final_score = obs.metadata.get("final_score", obs.reward or 0.0)
        assert final_score > 0
        assert obs.done is True

    def test_actor_messages_present_throughout_episode(self):
        env = MultiAgentBoardroomEnvironment()
        obs = env.reset(seed=1, difficulty="easy")
        policy = ScenarioAwarePolicy(difficulty="easy", snapshot=obs.data_tables, multi_agent=True)
        max_steps = int(obs.metadata.get("max_steps", 10))

        for step in range(1, max_steps + 1):
            if obs.done:
                break
            action_dict = policy.next_action(step)
            action = BoardroomAction(
                action_type=action_dict["action_type"],
                parameters=action_dict["parameters"],
            )
            obs = env.step(action)
            assert "actor_messages" in obs.metadata
            assert set(obs.metadata["actor_messages"].keys()) == {"ceo", "cfo", "risk_officer"}


# ---------------------------------------------------------------------------
# Task 15.2: OpenEnv compatibility
# ---------------------------------------------------------------------------

class TestOpenEnvCompatibility:
    def test_importable_from_my_env(self):
        from my_env import MultiAgentBoardroomEnvironment as MAE
        assert MAE is MultiAgentBoardroomEnvironment

    def test_reset_returns_boardroom_observation(self):
        env = MultiAgentBoardroomEnvironment()
        obs = env.reset(seed=42, difficulty="medium")
        assert isinstance(obs, BoardroomObservation)

    def test_step_returns_boardroom_observation(self):
        env = MultiAgentBoardroomEnvironment()
        env.reset(seed=42, difficulty="medium")
        obs = env.step(BoardroomAction(
            action_type="query_data",
            parameters={"metric": "revenue"},
        ))
        assert isinstance(obs, BoardroomObservation)

    def test_observation_has_required_fields(self):
        env = MultiAgentBoardroomEnvironment()
        obs = env.reset(seed=42, difficulty="easy")
        assert hasattr(obs, "data_tables")
        assert hasattr(obs, "done")
        assert hasattr(obs, "reward")
        assert hasattr(obs, "metadata")
        assert hasattr(obs, "step_count")
        assert hasattr(obs, "quarter")


# ---------------------------------------------------------------------------
# Task 15.3: Inference runner compatibility
# ---------------------------------------------------------------------------

class TestInferenceRunnerCompatibility:
    def test_reset_and_step_no_exceptions(self):
        env = MultiAgentBoardroomEnvironment()
        obs = env.reset(seed=0, difficulty="medium")
        assert obs is not None
        assert not obs.done

        obs2 = env.step(BoardroomAction(
            action_type="query_data",
            parameters={"metric": "revenue"},
        ))
        assert obs2 is not None
        assert obs2.step_count == 1

    def test_observation_schema_matches_existing(self):
        env = MultiAgentBoardroomEnvironment()
        env.reset(seed=0, difficulty="easy")
        obs = env.step(BoardroomAction(
            action_type="query_data",
            parameters={"metric": "churn_rate"},
        ))
        # Must have all fields from the existing BoardroomObservation schema
        assert isinstance(obs.data_tables, dict)
        assert isinstance(obs.done, bool)
        assert isinstance(obs.reward, float)
        assert isinstance(obs.metadata, dict)
        assert isinstance(obs.step_count, int)
        assert isinstance(obs.quarter, int)
        # Plus the new actor_messages field
        assert "actor_messages" in obs.metadata

    def test_openenv_serialization_keeps_multi_agent_fields(self):
        from openenv.core.env_server.http_server import serialize_observation

        env = MultiAgentBoardroomEnvironment()
        env.reset(seed=0, difficulty="easy")
        obs = env.step(BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": "reduce churn",
                "parameters": {},
                "explanation": "churn is the bottleneck based on data",
            },
        ))
        payload = serialize_observation(obs)
        serialized_obs = payload["observation"]

        assert serialized_obs["objective"] == "Find the growth bottleneck"
        assert serialized_obs["max_steps"] == 10
        assert serialized_obs["actor_messages"]
        assert serialized_obs["board_vote"]
        assert serialized_obs["vote_result"] in {"approved", "rejected"}

    def test_all_difficulties_work(self):
        for difficulty in ("easy", "medium", "hard"):
            env = MultiAgentBoardroomEnvironment()
            obs = env.reset(seed=42, difficulty=difficulty)
            assert obs.metadata["difficulty"] == difficulty
            obs2 = env.step(BoardroomAction(
                action_type="query_data",
                parameters={"metric": "revenue"},
            ))
            assert obs2.step_count == 1


# ---------------------------------------------------------------------------
# Task 15.4: GRPO training script smoke test
# ---------------------------------------------------------------------------

class TestGrpoTrainingScriptSmokeTest:
    def test_train_grpo_imports_without_error(self):
        import train_grpo
        assert hasattr(train_grpo, "MultiAgentBoardroomEnvironment")
        assert hasattr(train_grpo, "CurriculumScheduler")
        assert hasattr(train_grpo, "make_reward_fn")

    def test_grpo_trainer_initialises_with_reward_fn(self):
        import train_grpo
        env = MultiAgentBoardroomEnvironment()
        env.reset(seed=0, difficulty="easy")
        scheduler = train_grpo.CurriculumScheduler()
        reward_fn = train_grpo.make_reward_fn(env, scheduler)
        assert callable(reward_fn)

    def test_reward_fn_returns_floats_for_valid_actions(self):
        import train_grpo
        import json
        env = MultiAgentBoardroomEnvironment()
        env.reset(seed=0, difficulty="easy")
        scheduler = train_grpo.CurriculumScheduler()
        reward_fn = train_grpo.make_reward_fn(env, scheduler)

        completions = [
            json.dumps({"action_type": "query_data", "parameters": {"metric": "revenue"}}),
            json.dumps({"action_type": "query_data", "parameters": {"metric": "churn_rate"}}),
        ]
        rewards = reward_fn(completions)
        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)

    def test_reward_fn_does_not_reset_mid_batch(self):
        import train_grpo
        import json
        env = MultiAgentBoardroomEnvironment()
        env.reset(seed=0, difficulty="easy")
        scheduler = train_grpo.CurriculumScheduler()
        reward_fn = train_grpo.make_reward_fn(env, scheduler)

        completions = [
            json.dumps({
                "action_type": "make_decision",
                "parameters": {
                    "decision": "reduce churn",
                    "parameters": {},
                    "explanation": "churn is the bottleneck based on data",
                },
            }),
            json.dumps({"action_type": "query_data", "parameters": {"metric": "revenue"}}),
        ]
        rewards = reward_fn(completions)

        assert len(rewards) == 2
        assert env._step_count == 0
        assert env._done is False

    def test_reward_fn_returns_zero_for_malformed_json(self):
        import train_grpo
        env = MultiAgentBoardroomEnvironment()
        env.reset(seed=0, difficulty="easy")
        scheduler = train_grpo.CurriculumScheduler()
        reward_fn = train_grpo.make_reward_fn(env, scheduler)

        rewards = reward_fn(["not valid json at all !!!"])
        assert rewards == [0.0]

    def test_curriculum_scheduler_advances_difficulty(self):
        import train_grpo
        scheduler = train_grpo.CurriculumScheduler()
        assert scheduler.difficulty == "easy"
        # Fill rolling window with high scores
        for _ in range(train_grpo.CURRICULUM_WINDOW):
            scheduler.record(0.8)
        advanced = scheduler.maybe_advance()
        assert advanced is True
        assert scheduler.difficulty == "medium"
