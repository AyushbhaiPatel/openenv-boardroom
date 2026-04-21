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

# Multi-agent episodes split evidence presentation out from general stakeholder
# handling so repeated present_evidence steps do not dilute negotiation/consulting.
_MULTI_AGENT_TIER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "easy": {
        "data_gathering": 0.32,
        "analysis": 0.20,
        "counterfactual": 0.20,
        "stakeholder": 0.08,
        "evidence": 0.10,
        "decision": 0.10,
    },
    "medium": {
        "data_gathering": 0.20,
        "analysis": 0.20,
        "counterfactual": 0.20,
        "stakeholder": 0.16,
        "evidence": 0.12,
        "decision": 0.12,
    },
    "hard": {
        "data_gathering": 0.16,
        "analysis": 0.20,
        "counterfactual": 0.26,
        "stakeholder": 0.14,
        "evidence": 0.12,
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
        """query_data: reward relevant discovery, taper repeats."""
        if not context.get("relevant", False):
            if not context.get("novel", True):
                return -0.02
            return 0.0
        if context.get("novel", True):
            return 0.2
        return 0.05

    @staticmethod
    def _reward_analyze_trend(context: Dict[str, Any]) -> float:
        """analyze_trend: reward sufficiently deep and targeted trend work."""
        if not context.get("relevant", True) and not context.get("novel", True):
            return -0.03
        if context.get("noise_handled", False) and context.get("relevant", True):
            return 0.25
        if context.get("noise_handled", False):
            return 0.12
        return 0.0

    @staticmethod
    def _reward_simulate_counterfactual(context: Dict[str, Any]) -> float:
        """simulate_counterfactual: reward insightful and non-redundant simulations."""
        if context.get("insightful", False) and context.get("novel", True):
            return 0.3
        if context.get("insightful", False):
            return 0.12
        return 0.0

    @staticmethod
    def _reward_consult_stakeholder(context: Dict[str, Any]) -> float:
        """consult_stakeholder: +0.15 when stakeholder pressure is navigated."""
        nav = context.get("navigation_score", 0.0)
        novelty = 1.0 if context.get("novel", True) else 0.4
        return 0.15 * float(nav) * novelty

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

        Aggregates cumulative step rewards across component buckets and applies
        difficulty-tier-specific weights. Multi-agent episodes use a dedicated
        evidence bucket for present_evidence actions.

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
        is_multi_agent_episode = any(
            entry.get("action_type") in {"present_evidence", "negotiate"}
            for entry in episode_history
        )
        weight_profiles = (
            _MULTI_AGENT_TIER_WEIGHTS if is_multi_agent_episode else _TIER_WEIGHTS
        )
        weights = weight_profiles.get(difficulty, weight_profiles["medium"])

        buckets: Dict[str, float] = {
            "data_gathering": 0.0,
            "analysis": 0.0,
            "counterfactual": 0.0,
            "stakeholder": 0.0,
            "evidence": 0.0,
            "decision": 0.0,
        }
        bucket_max_possible: Dict[str, float] = {
            "data_gathering": 0,
            "analysis": 0,
            "counterfactual": 0,
            "stakeholder": 0,
            "evidence": 0,
            "decision": 0,
        }
        bucket_min_possible: Dict[str, float] = dict(bucket_max_possible)
        max_rewards: Dict[str, float] = {
            "query_data": 0.2,
            "analyze_trend": 0.25,
            "simulate_counterfactual": 0.3,
            "consult_stakeholder": 0.15,
            "present_evidence": 0.15,
            "negotiate": 0.25,
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
            "present_evidence": "evidence",
            "negotiate": "stakeholder",
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
            bucket_max_possible[bucket] += max_rewards.get(action_type, 0.0)
            bucket_min_possible[bucket] += min_rewards.get(action_type, 0.0)

        normalized: Dict[str, float] = {}
        for key in buckets:
            hi = bucket_max_possible[key]
            lo = bucket_min_possible[key]
            if hi == 0.0 and lo == 0.0:
                normalized[key] = 0.0
                continue
            span = hi - lo
            if span > 0:
                normalized[key] = max(
                    0.0, min(1.0, (buckets[key] - lo) / span)
                )
            else:
                normalized[key] = 0.0

        score = sum(weights[k] * normalized.get(k, 0.0) for k in weights)

        # Clamp to (0.0, 1.0) for safety (Property 11).
        # Scores must be strictly between 0 and 1 (not inclusive).
        return max(0.01, min(0.99, score))
