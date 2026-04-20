# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic benchmark agent powered by a scenario-aware playbook."""

import numpy as np

from my_env.models import BoardroomAction
from my_env.policy import ScenarioAwarePolicy
from my_env.server.boardroom_environment import BoardroomEnvironment
from my_env.server.multi_agent_boardroom_environment import MultiAgentBoardroomEnvironment

# Difficulty tiers to benchmark.
DIFFICULTY_TIERS = ["easy", "medium", "hard"]

# Number of episodes per tier.
EPISODES_PER_TIER = 100


def run_episode(env: BoardroomEnvironment, seed: int, difficulty: str) -> float:
    """Run a single episode with the baseline policy. Returns the final score."""
    obs = env.reset(seed=seed, difficulty=difficulty)
    policy = ScenarioAwarePolicy(difficulty=difficulty, snapshot=obs.data_tables)
    max_steps = int(obs.metadata.get("max_steps", 20))

    for step in range(1, max_steps + 1):
        if obs.done:
            break
        action_dict = policy.next_action(step)
        action = BoardroomAction(
            action_type=action_dict["action_type"],
            parameters=action_dict["parameters"],
        )
        obs = env.step(action)

    # Extract final score from the last observation.
    return obs.metadata.get("final_score", obs.reward or 0.0)


def run_multi_agent_episode(env: MultiAgentBoardroomEnvironment, seed: int, difficulty: str) -> float:
    """Run a single episode with the multi-agent environment. Returns the final score."""
    obs = env.reset(seed=seed, difficulty=difficulty)
    policy = ScenarioAwarePolicy(difficulty=difficulty, snapshot=obs.data_tables, multi_agent=True)
    max_steps = int(obs.metadata.get("max_steps", 20))

    for step in range(1, max_steps + 1):
        if obs.done:
            break
        action_dict = policy.next_action(step)
        action = BoardroomAction(
            action_type=action_dict["action_type"],
            parameters=action_dict["parameters"],
        )
        obs = env.step(action)

    return obs.metadata.get("final_score", obs.reward or 0.0)


def main() -> None:
    """Run baseline agent across all difficulty tiers and print results."""
    env = BoardroomEnvironment()

    for tier in DIFFICULTY_TIERS:
        scores: list[float] = []
        for i in range(EPISODES_PER_TIER):
            score = run_episode(env, seed=i, difficulty=tier)
            scores.append(score)

        arr = np.array(scores)
        print(f"{tier:>8s}: mean={arr.mean():.4f} ± std={arr.std():.4f}")

    print("\n--- Multi-Agent Benchmark ---")
    ma_env = MultiAgentBoardroomEnvironment()

    for tier in DIFFICULTY_TIERS:
        scores = []
        for i in range(EPISODES_PER_TIER):
            score = run_multi_agent_episode(ma_env, seed=i, difficulty=tier)
            scores.append(score)

        arr = np.array(scores)
        print(f"{tier:>8s}: mean={arr.mean():.4f} ± std={arr.std():.4f}")


if __name__ == "__main__":
    main()
