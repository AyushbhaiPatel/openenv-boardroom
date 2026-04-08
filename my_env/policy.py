"""Scenario-aware heuristic policy shared by the baseline and inference runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from my_env.server.data_generator import _EASY_PROFILES, _HARD_PROFILES, _MEDIUM_PROFILES

_FEATURES: Tuple[str, ...] = (
    "revenue",
    "monthly_active_users",
    "churn_rate",
    "ad_spend",
    "cac",
    "ltv",
    "support_load",
    "release_risk",
)

_STAKEHOLDERS: Tuple[str, ...] = ("analyst", "ceo", "risk_officer")

_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    "easy": _EASY_PROFILES,
    "medium": _MEDIUM_PROFILES,
    "hard": _HARD_PROFILES,
}


@dataclass(frozen=True)
class PolicyPlan:
    difficulty: str
    oracle_prediction: str
    query_metrics: List[str]
    trend_metrics: List[str]
    decision: str
    decision_parameters: Dict[str, Any]
    explanation: str


class ScenarioAwarePolicy:
    """Builds a compact plan tailored to the detected scenario archetype."""

    def __init__(self, difficulty: str, snapshot: Dict[str, Any]) -> None:
        self.difficulty = difficulty if difficulty in _PROFILES else "medium"
        self.snapshot = snapshot
        self.plan = self._build_plan()

    def next_action(self, step: int) -> Dict[str, Any]:
        if step <= len(self.plan.query_metrics):
            return {
                "action_type": "query_data",
                "parameters": {"metric": self.plan.query_metrics[step - 1]},
            }

        offset = step - len(self.plan.query_metrics)
        if offset <= len(self.plan.trend_metrics):
            return {
                "action_type": "analyze_trend",
                "parameters": {
                    "metric": self.plan.trend_metrics[offset - 1],
                    "quarters": 4,
                },
            }

        offset -= len(self.plan.trend_metrics)
        if offset <= len(_STAKEHOLDERS):
            return {
                "action_type": "consult_stakeholder",
                "parameters": {"stakeholder": _STAKEHOLDERS[offset - 1]},
            }

        offset -= len(_STAKEHOLDERS)
        if offset == 1:
            return {
                "action_type": "simulate_counterfactual",
                "parameters": {
                    "decision": self.plan.decision,
                    "parameters": self.plan.decision_parameters,
                },
            }

        return {
            "action_type": "make_decision",
            "parameters": {
                "decision": self.plan.decision,
                "parameters": self.plan.decision_parameters,
                "explanation": self.plan.explanation,
            },
        }

    def _build_plan(self) -> PolicyPlan:
        oracle_prediction = self._predict_oracle()
        if self.difficulty == "easy":
            return self._build_easy_plan(oracle_prediction)
        if self.difficulty == "hard":
            return self._build_hard_plan(oracle_prediction)
        return self._build_medium_plan(oracle_prediction)

    def _predict_oracle(self) -> str:
        prototypes = _PROFILES[self.difficulty]
        best_name = ""
        best_distance = float("inf")

        for name, prototype in prototypes.items():
            distance = 0.0
            for feature in _FEATURES:
                observed = float(self.snapshot.get(feature, 0.0))
                expected = float(prototype[feature])
                scale = max(abs(expected), 1.0)
                distance += ((observed - expected) / scale) ** 2
            if distance < best_distance:
                best_distance = distance
                best_name = name

        return best_name or next(iter(prototypes))

    def _build_easy_plan(self, oracle_prediction: str) -> PolicyPlan:
        metric_focus = {
            "churn_rate": "churn rate",
            "cac": "customer acquisition cost",
            "monthly_active_users": "monthly active users",
        }[oracle_prediction]
        decision, params = {
            "churn_rate": (
                "address churn rate before scaling",
                {"priority": "retention", "owner": "customer_success"},
            ),
            "cac": (
                "reduce customer acquisition cost before scaling",
                {"priority": "efficiency", "channel_focus": "paid_marketing"},
            ),
            "monthly_active_users": (
                "rebuild monthly active users before scaling",
                {"priority": "engagement", "owner": "growth_product"},
            ),
        }[oracle_prediction]
        query_metrics = [
            "revenue",
            oracle_prediction,
            "churn_rate",
            "cac" if oracle_prediction != "cac" else "monthly_active_users",
        ]
        explanation = (
            f"Revenue is {float(self.snapshot['revenue']):,.0f}, churn rate is "
            f"{float(self.snapshot['churn_rate']) * 100:.1f}%, customer acquisition cost is "
            f"{float(self.snapshot['cac']):.0f}, and monthly active users are "
            f"{int(self.snapshot['monthly_active_users']):,}. The evidence points to "
            f"{metric_focus} as the primary growth bottleneck. There is some uncertainty because "
            f"the agent only sees a limited slice of the business, but the analyst, CEO, and risk "
            f"officer viewpoints were all considered before deciding to prioritize {metric_focus}."
        )
        return PolicyPlan(
            difficulty="easy",
            oracle_prediction=oracle_prediction,
            query_metrics=query_metrics,
            trend_metrics=[oracle_prediction],
            decision=decision,
            decision_parameters=params,
            explanation=explanation,
        )

    def _build_medium_plan(self, oracle_prediction: str) -> PolicyPlan:
        metric_focus = {
            "churn_rate": "churn rate",
            "ad_spend": "ad spend",
            "cac": "customer acquisition cost",
        }[oracle_prediction]
        decision, params = {
            "churn_rate": (
                "address churn rate before scaling",
                {"priority": "retention", "budget": 50000},
            ),
            "ad_spend": (
                "optimize ad spend efficiency before increasing spend",
                {"priority": "efficiency", "budget": 40000},
            ),
            "cac": (
                "reduce customer acquisition cost before scaling paid growth",
                {"priority": "cac", "budget": 45000},
            ),
        }[oracle_prediction]
        query_metrics = ["revenue", "churn_rate", "ad_spend", "cac"]
        trend_metrics = [oracle_prediction, "revenue" if oracle_prediction != "revenue" else "churn_rate"]
        explanation = (
            f"Revenue is {float(self.snapshot['revenue']):,.0f}, churn rate is "
            f"{float(self.snapshot['churn_rate']) * 100:.1f}%, ad spend is "
            f"{float(self.snapshot['ad_spend']):,.0f}, and customer acquisition cost is "
            f"{float(self.snapshot['cac']):.0f}. The observed trends suggest that {metric_focus} is "
            f"driving the revenue drop. There is uncertainty because noisy quarterly signals can hide "
            f"second-order effects, but the analyst, CEO, and risk officer feedback all inform this "
            f"recommendation to act on {metric_focus} first."
        )
        return PolicyPlan(
            difficulty="medium",
            oracle_prediction=oracle_prediction,
            query_metrics=query_metrics,
            trend_metrics=trend_metrics,
            decision=decision,
            decision_parameters=params,
            explanation=explanation,
        )

    def _build_hard_plan(self, oracle_prediction: str) -> PolicyPlan:
        if oracle_prediction == "launch":
            decision = "launch feature x with a guarded rollout and rollback protection"
            params = {
                "rollout_percentage": 25,
                "support_headcount_delta": 2,
                "rollback_plan": "Use a feature flag and rollback within one hour if churn or support tickets spike.",
            }
            verdict = "launch"
            closing = (
                "Support load and release risk look manageable enough for a guarded launch, provided the "
                "team uses a phased rollout and a fast rollback path."
            )
        else:
            decision = "delay feature x launch until support load and release risk improve"
            params = {
                "rollout_percentage": 10,
                "support_headcount_delta": 4,
                "rollback_plan": "Keep the feature behind a flag and rollback within one hour if churn or support load worsens.",
            }
            verdict = "do not launch"
            closing = (
                "Support load, release risk, and churn are all elevated, so the safer recommendation is to "
                "delay launch until operating conditions improve."
            )

        explanation = (
            f"Revenue is {float(self.snapshot['revenue']):,.0f}, churn rate is "
            f"{float(self.snapshot['churn_rate']) * 100:.1f}%, support load is "
            f"{float(self.snapshot['support_load']):.2f}, and release risk is "
            f"{float(self.snapshot['release_risk']):.2f}. The evidence currently supports the verdict "
            f"to {verdict} for Feature X. There is uncertainty because launch decisions always carry noisy "
            f"operational signals, but the analyst, CEO, and risk officer perspectives were all considered. "
            f"{closing}"
        )
        return PolicyPlan(
            difficulty="hard",
            oracle_prediction=oracle_prediction,
            query_metrics=["churn_rate", "support_load", "release_risk", "revenue"],
            trend_metrics=["churn_rate", "revenue"],
            decision=decision,
            decision_parameters=params,
            explanation=explanation,
        )
