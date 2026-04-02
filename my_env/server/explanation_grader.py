# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ExplanationGrader for the OpenBoardroom Environment.

Heuristic-based scoring of agent explanations on make_decision actions.
Evaluates presence of data evidence references, uncertainty acknowledgment,
and stakeholder perspective consideration.  Returns a deterministic score
in [0.0, 1.0] for the same input and context.
"""

import re
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Keyword / pattern sets used for heuristic grading
# ---------------------------------------------------------------------------

# Data-evidence indicators: metric names and numeric references.
_DATA_EVIDENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brevenue\b", re.IGNORECASE),
    re.compile(r"\bchurn\b", re.IGNORECASE),
    re.compile(r"\bmau\b", re.IGNORECASE),
    re.compile(r"\bmonthly.active.users?\b", re.IGNORECASE),
    re.compile(r"\bcac\b", re.IGNORECASE),
    re.compile(r"\bltv\b", re.IGNORECASE),
    re.compile(r"\bad.spend\b", re.IGNORECASE),
    re.compile(r"\bgrowth\b", re.IGNORECASE),
    re.compile(r"\bmetric\b", re.IGNORECASE),
    re.compile(r"\bdata\b", re.IGNORECASE),
    # Numeric references (e.g. "$1.2M", "15%", "50000")
    re.compile(r"\d+\.?\d*\s*%"),
    re.compile(r"\$\s*\d"),
    # Business-scale numbers (avoid matching years like 2024 as sole "evidence")
    re.compile(r"\b\d{1,3}(?:,\d{3})+\b"),
    re.compile(r"\b\d+\.\d+\s*(?:%|k|m|b)\b", re.IGNORECASE),
]

# Scenario-specific bonus keywords (objective substring -> patterns)
_SCENARIO_KEYWORD_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        "bottleneck",
        [
            re.compile(r"\bbottleneck\b", re.IGNORECASE),
            re.compile(r"\blimiting\b", re.IGNORECASE),
            re.compile(r"\bconstraint\b", re.IGNORECASE),
            re.compile(r"\bgrowth\b", re.IGNORECASE),
        ],
    ),
    (
        "revenue drop",
        [
            re.compile(r"\brevenue\b", re.IGNORECASE),
            re.compile(r"\bdeclin", re.IGNORECASE),
            re.compile(r"\bdrop\b", re.IGNORECASE),
            re.compile(r"\bcause\b", re.IGNORECASE),
        ],
    ),
    (
        "launch",
        [
            re.compile(r"\blaunch\b", re.IGNORECASE),
            re.compile(r"\bfeature\b", re.IGNORECASE),
            re.compile(r"\brisk\b", re.IGNORECASE),
            re.compile(r"\btrade-?off\b", re.IGNORECASE),
        ],
    ),
]

# Uncertainty / hedging language.
_UNCERTAINTY_KEYWORDS: list[str] = [
    "might",
    "could",
    "may",
    "uncertain",
    "uncertainty",
    "risk",
    "risky",
    "possible",
    "possibly",
    "likely",
    "unlikely",
    "perhaps",
    "unclear",
    "approximate",
    "estimated",
    "potential",
    "potentially",
]

# Stakeholder perspective references.
_STAKEHOLDER_KEYWORDS: list[str] = [
    "analyst",
    "ceo",
    "risk officer",
    "risk_officer",
    "stakeholder",
    "stakeholders",
    "perspective",
    "viewpoint",
    "viewpoints",
    "feedback",
]

_ORACLE_ALIASES: dict[str, tuple[str, ...]] = {
    "ad_spend": ("ad spend", "ad-spend", "marketing spend"),
    "monthly_active_users": ("monthly active users", "active users", "mau"),
    "churn_rate": ("churn rate", "churn", "retention"),
    "do not launch": ("do not launch", "delay", "hold", "postpone"),
}

_NEGATIVE_LAUNCH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdo not launch\b", re.IGNORECASE),
    re.compile(r"\bdon't launch\b", re.IGNORECASE),
    re.compile(r"\bdelay(?:ed|ing)?\s+launch\b", re.IGNORECASE),
    re.compile(r"\bhold(?:ing)?\s+launch\b", re.IGNORECASE),
    re.compile(r"\bpostpone(?:d|ment)?\s+launch\b", re.IGNORECASE),
)

# ---------------------------------------------------------------------------
# Component weights (must sum to 1.0)
# ---------------------------------------------------------------------------
_WEIGHT_DATA_EVIDENCE = 0.40
_WEIGHT_UNCERTAINTY = 0.30
_WEIGHT_STAKEHOLDER = 0.30


class ExplanationGrader:
    """Heuristic grader for agent explanations on ``make_decision`` actions.

    The grader evaluates three dimensions of explanation quality:

    1. **Data evidence** — references to metric names, numbers, or
       quantitative evidence.
    2. **Uncertainty acknowledgment** — hedging language that shows the
       agent recognises limits of its knowledge.
    3. **Stakeholder perspective** — mentions of different stakeholder
       viewpoints or personas.

    Each dimension produces a sub-score in [0, 1].  The final score is a
    weighted combination clamped to [0.0, 1.0].  The grading is fully
    deterministic for the same ``explanation`` and ``scenario_context``.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grade(self, explanation: str, scenario_context: Dict[str, Any]) -> float:
        """Grade an explanation string.

        Args:
            explanation: Free-text explanation provided by the agent in a
                ``make_decision`` action.
            scenario_context: Must include ``objective`` (and optionally
                ``difficulty``) to weight explanation toward scenario-relevant
                vocabulary (bottleneck, revenue drop, launch / feature, etc.).

        Returns:
            A float score in [0.0, 1.0].
        """
        if not explanation or not explanation.strip():
            return 0.0

        data_score = self._score_data_evidence(explanation)
        uncertainty_score = self._score_uncertainty(explanation)
        stakeholder_score = self._score_stakeholder(explanation)
        scenario_bonus = self._score_scenario_alignment(
            explanation, scenario_context
        )
        oracle_bonus = self._score_oracle_alignment(explanation, scenario_context)

        combined = (
            _WEIGHT_DATA_EVIDENCE * data_score
            + _WEIGHT_UNCERTAINTY * uncertainty_score
            + _WEIGHT_STAKEHOLDER * stakeholder_score
        )
        # Up to +0.15 from objective-aligned phrasing (then re-normalize into [0,1])
        combined = combined * 0.85 + 0.15 * scenario_bonus
        combined = combined * 0.9 + 0.1 * oracle_bonus

        return max(0.0, min(1.0, combined))

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_data_evidence(text: str) -> float:
        """Score presence of data / metric references in *text*.

        Returns a value in [0, 1] based on how many distinct evidence
        patterns are matched.  Hitting 3+ distinct patterns yields 1.0.
        """
        matches = sum(1 for p in _DATA_EVIDENCE_PATTERNS if p.search(text))
        # 1 match → 0.33, 2 → 0.67, 3+ → 1.0
        return min(matches / 3.0, 1.0)

    @staticmethod
    def _score_uncertainty(text: str) -> float:
        """Score presence of uncertainty / hedging language.

        Returns a value in [0, 1].  Hitting 2+ keywords yields 1.0.
        """
        lower = text.lower()
        matches = sum(1 for kw in _UNCERTAINTY_KEYWORDS if kw in lower)
        # 1 match → 0.5, 2+ → 1.0
        return min(matches / 2.0, 1.0)

    @staticmethod
    def _score_stakeholder(text: str) -> float:
        """Score consideration of stakeholder perspectives.

        Returns a value in [0, 1].  Hitting 2+ keywords yields 1.0.
        """
        lower = text.lower()
        matches = sum(1 for kw in _STAKEHOLDER_KEYWORDS if kw in lower)
        # 1 match → 0.5, 2+ → 1.0
        return min(matches / 2.0, 1.0)

    @staticmethod
    def _score_scenario_alignment(text: str, scenario_context: Dict[str, Any]) -> float:
        """Reward phrases that match the episode objective (0..1)."""
        objective = (scenario_context or {}).get("objective") or ""
        if not objective:
            return 0.5
        obj_lower = objective.lower()
        best = 0.0
        for needle, patterns in _SCENARIO_KEYWORD_PATTERNS:
            if needle not in obj_lower:
                continue
            hits = sum(1 for p in patterns if p.search(text))
            best = max(best, min(hits / float(len(patterns)), 1.0))
        return best if best > 0 else 0.3

    @staticmethod
    def _score_oracle_alignment(text: str, scenario_context: Dict[str, Any]) -> float:
        oracle = ((scenario_context or {}).get("oracle_answer") or "").lower()
        if not oracle:
            return 0.5
        lower = text.lower()
        negative_launch = any(pattern.search(lower) for pattern in _NEGATIVE_LAUNCH_PATTERNS)
        aliases = _ORACLE_ALIASES.get(oracle, ())
        normalized_candidates = {
            oracle,
            oracle.replace("_", " "),
            oracle.replace("_", "-"),
            *aliases,
        }
        if oracle == "launch" and negative_launch:
            return 0.25
        if any(candidate and candidate in lower for candidate in normalized_candidates):
            return 1.0
        if oracle == "do not launch" and any(token in lower for token in ("delay", "hold", "postpone")):
            return 0.9
        if oracle == "launch" and "launch" in lower:
            return 0.9
        return 0.25
