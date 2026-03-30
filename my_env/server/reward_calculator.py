# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RewardCalculator for the OpenBoardroom Environment.

Computes dense, multi-component step rewards for each action type and
a normalized final episode score with difficulty-tier-specific weights.
"""

from typing import Any, Dict, List

try:
    from my_env.models import BoardroomAction
except ImportError:
    from models import BoardroomAction


# Difficulty-tier weight profiles for final score aggregation.
# Keys: data_gathering, analysis (trends only), counterfactual, stakeholder, decision
_TIER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "easy": {
        "data_gathering": 0.36,
        "analysis": 0.22,
        "counterfactual": 0.22,
        "stakeholder": 0.10,
        "decision": 0.10,
    },
    "medium": {
        "data_gathering": 0.22,
        "analysis": 0.22,
        "counterfactual": 0.22,
        "stakeholder": 0.22,
        "decision": 0.12,
    },
    "hard": {
        "data_gathering": 0.18,
        "analysis": 0.22,
        "counterfactual": 0.28,
        "stakeholder": 0.20,
        "decision": 0.12,
    },
}


class RewardCalculator:
    """Dense, multi-component reward computation.

    Provides per-step rewards based on action type and quality context,
    and a final normalized episode score aggregated with difficulty-tier
    weights.
    """

    def compute_step_reward(
        self, action: BoardroomAction, context: Dict[str, Any]
    ) -> float:
        """Compute the reward for a single step.

        Args:
            action: The BoardroomAction taken by the agent.
            context: Dictionary with contextual information used to evaluate
                the quality of the action.  Expected keys vary by action type:
                - query_data: ``relevant`` (bool) — whether the queried metric
                  is relevant to the scenario.
                - analyze_trend: ``noise_handled`` (bool) — whether the agent
                  correctly identified or handled noise.
                - simulate_counterfactual: ``insightful`` (bool) — whether the
                  simulation was insightful (relevant decision + params).
                - consult_stakeholder: ``navigation_score`` (float 0-1) — how
                  effectively the agent navigated stakeholder pressure.
                - make_decision: ``decision_quality`` (float 0-1) — overall
                  quality of the decision; ``explanation_score`` (float 0-1).

        Returns:
            A float reward value.
        """
        action_type = action.action_type
        if action_type == "query_data":
            return self._reward_query_data(context)
        elif action_type == "analyze_trend":
            return self._reward_analyze_trend(context)
        elif action_type == "simulate_counterfactual":
            return self._reward_simulate_counterfactual(context)
        elif action_type == "consult_stakeholder":
            return self._reward_consult_stakeholder(context)
        elif action_type == "make_decision":
            return self._reward_make_decision(context)
        # Fallback for unknown action types (should not happen with Pydantic
        # validation, but be defensive).
        return 0.0

    # ------------------------------------------------------------------
    # Per-action reward helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reward_query_data(context: Dict[str, Any]) -> float:
        """query_data: +0.2 when the queried metric is relevant."""
        if context.get("relevant", False):
            return 0.2
        return 0.0

    @staticmethod
    def _reward_analyze_trend(context: Dict[str, Any]) -> float:
        """analyze_trend: +0.25 when noise is correctly handled."""
        if context.get("noise_handled", False):
            return 0.25
        return 0.0

    @staticmethod
    def _reward_simulate_counterfactual(context: Dict[str, Any]) -> float:
        """simulate_counterfactual: +0.3 when the simulation is insightful."""
        if context.get("insightful", False):
            return 0.3
        return 0.0

    @staticmethod
    def _reward_consult_stakeholder(context: Dict[str, Any]) -> float:
        """consult_stakeholder: +0.15 when stakeholder pressure is navigated."""
        nav = context.get("navigation_score", 0.0)
        # Scale the base reward by navigation effectiveness.
        return 0.15 * float(nav)

    @staticmethod
    def _reward_make_decision(context: Dict[str, Any]) -> float:
        """make_decision: -0.1 to +0.5 depending on quality.

        Combines decision quality and explanation score.  A very low quality
        decision receives a penalty (-0.1), while a perfect decision with a
        perfect explanation can reach +0.5.
        """
        quality = float(context.get("decision_quality", 0.0))
        explanation = float(context.get("explanation_score", 0.0))

        # Map quality 0-1 to the range -0.1 .. +0.4, then add up to +0.1
        # from explanation score, giving a total range of -0.1 .. +0.5.
        base = -0.1 + 0.5 * quality  # -0.1 at q=0, +0.4 at q=1
        explanation_bonus = 0.1 * explanation  # 0.0 .. +0.1
        return base + explanation_bonus

    # ------------------------------------------------------------------
    # Final episode score
    # ------------------------------------------------------------------

    def compute_final_score(
        self, episode_history: List[Dict[str, Any]]
    ) -> float:
        """Compute the final normalized episode score in [0.0, 1.0].

        Aggregates cumulative step rewards across four component buckets
        (data_gathering, analysis, counterfactual, stakeholder) and applies
        difficulty-tier-specific weights.

        Args:
            episode_history: List of step records.  Each record is a dict
                with at least:
                - ``action_type`` (str)
                - ``reward`` (float)
                - ``difficulty`` (str) — the episode difficulty tier
                Optionally:
                - ``decision_quality`` (float)
                - ``explanation_score`` (float)
                - ``navigation_score`` (float)

        Returns:
            A float in [0.0, 1.0].
        """
        if not episode_history:
            return 0.0

        # Determine difficulty from the first entry (consistent across episode).
        difficulty = episode_history[0].get("difficulty", "medium")
        weights = _TIER_WEIGHTS.get(difficulty, _TIER_WEIGHTS["medium"])

        buckets: Dict[str, float] = {
            "data_gathering": 0.0,
            "analysis": 0.0,
            "counterfactual": 0.0,
            "stakeholder": 0.0,
            "decision": 0.0,
        }
        counts: Dict[str, int] = {
            "data_gathering": 0,
            "analysis": 0,
            "counterfactual": 0,
            "stakeholder": 0,
            "decision": 0,
        }
        max_rewards: Dict[str, float] = {
            "query_data": 0.2,
            "analyze_trend": 0.25,
            "simulate_counterfactual": 0.3,
            "consult_stakeholder": 0.15,
            "make_decision": 0.5,
        }
        min_rewards: Dict[str, float] = {
            "make_decision": -0.1,
        }
        bucket_map: Dict[str, str] = {
            "query_data": "data_gathering",
            "analyze_trend": "analysis",
            "simulate_counterfactual": "counterfactual",
            "consult_stakeholder": "stakeholder",
            "make_decision": "decision",
        }

        for entry in episode_history:
            action_type = entry.get("action_type", "")
            reward = float(entry.get("reward", 0.0))
            bucket = bucket_map.get(action_type)
            if bucket is None:
                continue
            if bucket == "decision":
                buckets[bucket] += reward  # signed: penalties count
            else:
                buckets[bucket] += max(reward, 0.0)
            counts[bucket] += 1

        normalized: Dict[str, float] = {}
        for key in buckets:
            if counts[key] == 0:
                normalized[key] = 0.0
                continue
            if key == "decision":
                n = counts[key]
                lo = n * min_rewards.get("make_decision", -0.1)
                hi = n * max_rewards.get("make_decision", 0.5)
                span = hi - lo
                if span > 0:
                    normalized[key] = max(
                        0.0, min(1.0, (buckets[key] - lo) / span)
                    )
                else:
                    normalized[key] = 0.0
            else:
                max_per_step = max(
                    max_rewards.get(at, 0.5)
                    for at, b in bucket_map.items()
                    if b == key
                )
                max_possible = counts[key] * max_per_step
                if max_possible > 0:
                    normalized[key] = min(buckets[key] / max_possible, 1.0)
                else:
                    normalized[key] = 0.0

        score = sum(weights[k] * normalized.get(k, 0.0) for k in weights)

        # Clamp to [0.0, 1.0] for safety (Property 11).
        return max(0.0, min(1.0, score))
