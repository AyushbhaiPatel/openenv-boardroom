# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline Agent for the OpenBoardroom Environment.

A deterministic rule-based policy that validates environment correctness
and produces benchmark scores across all difficulty tiers.

Policy per quarter:
  1. Query each key metric (revenue, churn_rate, monthly_active_users) — 3 steps
  2. Analyze trend on the worst-performing metric — 1 step
  3. Consult each stakeholder (analyst, ceo, risk_officer) — 3 steps
  4. Run counterfactual on the most obvious decision — 1 step
  5. Make decision with a template explanation — 1 step
  Total: 9 steps per quarter cycle

Executable with: python -m my_env.baseline_agent
"""

import numpy as np

from my_env.models import BoardroomAction
from my_env.server.boardroom_environment import BoardroomEnvironment

# Metrics to query each quarter cycle.
KEY_METRICS = ["revenue", "churn_rate", "monthly_active_users"]

# Stakeholders to consult each quarter cycle.
STAKEHOLDERS = ["analyst", "ceo", "risk_officer"]

# Difficulty tiers to benchmark.
DIFFICULTY_TIERS = ["easy", "medium", "hard"]

# Number of episodes per tier.
EPISODES_PER_TIER = 100


def _identify_worst_metric(queried: dict[str, float]) -> str:
    """Pick the metric that looks worst relative to a healthy baseline."""
    # Lower is worse for revenue/MAU; higher is worse for churn.
    scores: dict[str, float] = {}
    if "revenue" in queried:
        # Normalise: higher revenue is better → invert for "badness".
        scores["revenue"] = -queried["revenue"]
    if "monthly_active_users" in queried:
        scores["monthly_active_users"] = -queried["monthly_active_users"]
    if "churn_rate" in queried:
        # Higher churn is worse → positive = bad.
        scores["churn_rate"] = queried["churn_rate"] * 1e6  # scale up
    if not scores:
        return "revenue"
    return max(scores, key=lambda k: scores[k])


def _pick_decision(worst_metric: str) -> tuple[str, dict]:
    """Choose a decision and parameters based on the worst metric."""
    if worst_metric == "churn_rate":
        return "reduce_churn", {"strategy": "improve_retention", "budget": 50000}
    elif worst_metric == "monthly_active_users":
        return "increase_marketing", {"strategy": "expand_acquisition", "budget": 100000}
    else:
        return "optimize_revenue", {"strategy": "pricing_adjustment", "target": "premium_tier"}


def _build_explanation(
    worst_metric: str,
    queried: dict[str, float],
    stakeholder_feedback: list[str],
) -> str:
    """Build a template explanation referencing data, uncertainty, and stakeholders."""
    metric_summary = ", ".join(f"{k}={v:.4g}" for k, v in queried.items())
    return (
        f"After reviewing the data ({metric_summary}), "
        f"the worst-performing metric is {worst_metric}. "
        f"There is some uncertainty in the data due to potential noise, "
        f"but the trend is concerning. "
        f"The analyst emphasised data-driven caution, "
        f"the CEO pushed for growth, "
        f"and the risk officer urged stability. "
        f"Balancing these stakeholder perspectives, "
        f"I recommend addressing {worst_metric} as the priority."
    )


def run_episode(env: BoardroomEnvironment, seed: int, difficulty: str) -> float:
    """Run a single episode with the baseline policy. Returns the final score."""
    obs = env.reset(seed=seed, difficulty=difficulty)
    max_steps = obs.metadata.get("max_steps", 20)

    while not obs.done:
        # Phase 1: Query key metrics (3 steps)
        queried: dict[str, float] = {}
        for metric in KEY_METRICS:
            if obs.done:
                break
            action = BoardroomAction(
                action_type="query_data",
                parameters={"metric": metric},
            )
            obs = env.step(action)
            if metric in obs.data_tables:
                val = obs.data_tables[metric]
                if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
                    queried[metric] = float(val)

        if obs.done:
            break

        # Phase 2: Analyze trend on worst metric (1 step)
        worst = _identify_worst_metric(queried) if queried else "revenue"
        action = BoardroomAction(
            action_type="analyze_trend",
            parameters={"metric": worst, "quarters": 4},
        )
        obs = env.step(action)
        if obs.done:
            break

        # Phase 3: Consult each stakeholder (3 steps)
        feedback_list: list[str] = []
        for stakeholder in STAKEHOLDERS:
            if obs.done:
                break
            action = BoardroomAction(
                action_type="consult_stakeholder",
                parameters={"stakeholder": stakeholder},
            )
            obs = env.step(action)
            if obs.stakeholder_feedback:
                feedback_list.append(obs.stakeholder_feedback)

        if obs.done:
            break

        # Phase 4: Simulate counterfactual (1 step)
        decision, params = _pick_decision(worst)
        action = BoardroomAction(
            action_type="simulate_counterfactual",
            parameters={"decision": decision, "parameters": params},
        )
        obs = env.step(action)
        if obs.done:
            break

        # Phase 5: Make decision (1 step)
        explanation = _build_explanation(worst, queried, feedback_list)
        action = BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": decision,
                "parameters": params,
                "explanation": explanation,
            },
        )
        obs = env.step(action)

    # Extract final score from the last observation.
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


if __name__ == "__main__":
    main()
