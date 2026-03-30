# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Stakeholder Simulator for the OpenBoardroom Environment.

Simulates three stakeholder personas (analyst, ceo, risk_officer) with hidden
bias weight vectors that colour their feedback. All randomness is seeded via
numpy for full determinism (Req 4.3).
"""

import hashlib
from typing import Dict, List

import numpy as np

try:
    from ..models import CompanyState
except ImportError:
    from models import CompanyState


# ---------------------------------------------------------------------------
# Persona feedback templates
# ---------------------------------------------------------------------------
# Each persona has a pool of sentence fragments keyed by "dimension".
# The bias weights determine how strongly each dimension is emphasised.
# Dimensions: 0 = data/metrics, 1 = growth/revenue, 2 = risk/stability

_ANALYST_TEMPLATES: Dict[str, List[str]] = {
    "data": [
        "The numbers tell a clear story here.",
        "I'd want to see statistical significance before acting.",
        "Let's look at the data objectively — the metrics suggest",
        "Based on the quantitative evidence,",
        "The trend data indicates",
    ],
    "growth": [
        "Revenue growth of {revenue_fmt} is noteworthy, but we need more data points.",
        "MAU at {mau} warrants deeper cohort analysis.",
        "The growth figures need to be adjusted for seasonality.",
    ],
    "risk": [
        "Churn at {churn_pct} is a data point worth monitoring.",
        "CAC/LTV ratio of {cac_ltv_ratio:.2f} needs statistical context.",
        "I'd recommend a controlled experiment before drawing conclusions.",
    ],
}

_CEO_TEMPLATES: Dict[str, List[str]] = {
    "data": [
        "Forget the spreadsheets — what's our market position?",
        "Data is fine, but we need to move fast.",
        "Analysis paralysis won't capture market share.",
    ],
    "growth": [
        "Revenue at {revenue_fmt} — we should be doubling down on growth.",
        "With {mau} users, we need aggressive expansion.",
        "This is our moment to accelerate. Revenue growth is everything.",
        "Market share is the priority. Let's push harder.",
        "We need to 10x our user base. {mau} is just the start.",
    ],
    "risk": [
        "Churn at {churn_pct}? That's the cost of moving fast.",
        "Don't let risk aversion hold us back.",
        "Some churn is acceptable if we're growing the top line.",
    ],
}

_RISK_OFFICER_TEMPLATES: Dict[str, List[str]] = {
    "data": [
        "We need to be cautious about what the data isn't showing us.",
        "There may be hidden risks in these numbers.",
        "I'd want a thorough risk assessment before proceeding.",
    ],
    "growth": [
        "Revenue of {revenue_fmt} looks good, but at what cost?",
        "Growth is meaningless if we can't retain users.",
        "Sustainable growth matters more than top-line numbers.",
    ],
    "risk": [
        "Churn at {churn_pct} is alarming — we must prioritise retention.",
        "With CAC/LTV at {cac_ltv_ratio:.2f}, our unit economics are fragile.",
        "Downside protection should be our primary concern right now.",
        "We need to stabilise before expanding.",
        "The risk exposure here is significant. Let's be conservative.",
    ],
}

_PERSONA_TEMPLATES = {
    "analyst": _ANALYST_TEMPLATES,
    "ceo": _CEO_TEMPLATES,
    "risk_officer": _RISK_OFFICER_TEMPLATES,
}

# Default dimension ordering: [data, growth, risk]
_DIMENSION_KEYS = ["data", "growth", "risk"]


class StakeholderSimulator:
    """Simulates three stakeholder personas with hidden bias vectors.

    Each persona has a hidden bias weight vector (3 floats) initialised from
    the episode seed. The bias determines how strongly each dimension
    (data, growth, risk) is emphasised in their feedback.

    Personas:
        - **analyst**: Data-focused, skeptical of intuition.
        - **ceo**: Growth-obsessed, dismisses risk.
        - **risk_officer**: Conservative, prioritises stability.
    """

    VALID_STAKEHOLDERS = ("analyst", "ceo", "risk_officer")

    def __init__(self, seed: int) -> None:
        """Initialise bias weight vectors for all three personas.

        Args:
            seed: Episode seed for deterministic bias generation.
        """
        self._seed = seed
        rng = np.random.default_rng(seed)

        # Each persona gets a 3-element bias vector over [data, growth, risk].
        # We start with a persona-specific "base" bias and add seeded noise
        # so that episodes with different seeds produce different feedback.
        self._biases: Dict[str, np.ndarray] = {}

        # Analyst: heavy on data dimension
        base_analyst = np.array([0.6, 0.2, 0.2])
        self._biases["analyst"] = self._normalise(
            base_analyst + rng.uniform(-0.1, 0.1, size=3)
        )

        # CEO: heavy on growth dimension
        base_ceo = np.array([0.15, 0.65, 0.2])
        self._biases["ceo"] = self._normalise(
            base_ceo + rng.uniform(-0.1, 0.1, size=3)
        )

        # Risk Officer: heavy on risk dimension
        base_risk = np.array([0.2, 0.15, 0.65])
        self._biases["risk_officer"] = self._normalise(
            base_risk + rng.uniform(-0.1, 0.1, size=3)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consult(
        self,
        stakeholder: str,
        company_state: CompanyState,
        context: Dict,
    ) -> str:
        """Return feedback from *stakeholder* coloured by their hidden bias.

        Args:
            stakeholder: One of ``"analyst"``, ``"ceo"``, ``"risk_officer"``.
            company_state: Current company metrics.
            context: Additional episode context (unused currently but
                available for future extensions).

        Returns:
            A deterministic feedback string for the given stakeholder, state,
            and seed combination.

        Raises:
            ValueError: If *stakeholder* is not a recognised persona.
        """
        if stakeholder not in self.VALID_STAKEHOLDERS:
            raise ValueError(
                f"Unknown stakeholder '{stakeholder}'. "
                f"Must be one of {self.VALID_STAKEHOLDERS}."
            )

        bias = self._biases[stakeholder]
        templates = _PERSONA_TEMPLATES[stakeholder]

        # Build format kwargs from company state.
        fmt = self._format_kwargs(company_state)

        # Use a per-consultation RNG seeded from the episode seed, the
        # stakeholder name, and a hash of the company state snapshot so that
        # feedback is deterministic for the same inputs.
        state_hash = self._state_hash(company_state)
        consult_seed = (
            self._seed
            ^ _stable_u64(stakeholder)
            ^ state_hash
        ) % (2**63)
        rng = np.random.default_rng(consult_seed)

        # Select sentences weighted by bias (max ``total_picks`` sentences).
        sentences: List[str] = []
        total_picks = 3
        for dim_idx, dim_key in enumerate(_DIMENSION_KEYS):
            pool = templates[dim_key]
            n_picks = max(1, round(bias[dim_idx] * total_picks))
            if dim_idx == len(_DIMENSION_KEYS) - 1:
                n_picks = max(1, total_picks - len(sentences))
            n_picks = min(n_picks, total_picks - len(sentences))
            if n_picks <= 0:
                break
            indices = rng.choice(len(pool), size=min(n_picks, len(pool)), replace=False)
            for idx in indices:
                if len(sentences) >= total_picks:
                    break
                raw = pool[idx]
                try:
                    sentences.append(raw.format(**fmt))
                except (KeyError, IndexError):
                    sentences.append(raw)

        return " ".join(sentences[:total_picks])

    def compute_navigation_score(
        self, consultation_history: List[Dict]
    ) -> float:
        """Score how well the agent navigated stakeholder pressure.

        A high score means the agent consulted multiple stakeholders and
        didn't blindly follow any single one.  The score feeds into the
        reward calculator with weight 0.15 (Req 4.4).

        Args:
            consultation_history: List of dicts, each with at least a
                ``"stakeholder"`` key indicating who was consulted.

        Returns:
            A float in [0.0, 1.0].
        """
        if not consultation_history:
            return 0.0

        consulted = set()
        counts: Dict[str, int] = {}
        for entry in consultation_history:
            name = entry.get("stakeholder", "")
            if name in self.VALID_STAKEHOLDERS:
                consulted.add(name)
                counts[name] = counts.get(name, 0) + 1

        # Diversity component: fraction of personas consulted.
        diversity = len(consulted) / len(self.VALID_STAKEHOLDERS)

        # Balance component: penalise over-reliance on one persona.
        if counts:
            total = sum(counts.values())
            proportions = [c / total for c in counts.values()]
            # Ideal is uniform (1/n each).  Use 1 - max deviation.
            max_prop = max(proportions)
            balance = 1.0 - max(0.0, max_prop - 1.0 / len(self.VALID_STAKEHOLDERS))
        else:
            balance = 0.0

        # Weighted combination: diversity matters most.
        score = 0.6 * diversity + 0.4 * balance
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """Normalise *vec* so its elements sum to 1, all non-negative."""
        vec = np.maximum(vec, 0.0)
        total = vec.sum()
        if total == 0:
            return np.ones_like(vec) / len(vec)
        return vec / total

    @staticmethod
    def _format_kwargs(state: CompanyState) -> Dict[str, str]:
        """Build template format kwargs from a CompanyState."""
        cac_ltv = state.cac / state.ltv if state.ltv > 0 else float("inf")
        return {
            "revenue_fmt": f"${state.revenue:,.0f}",
            "mau": f"{state.monthly_active_users:,}",
            "churn_pct": f"{state.churn_rate * 100:.1f}%",
            "cac_ltv_ratio": cac_ltv,
        }

    @staticmethod
    def _state_hash(state: CompanyState) -> int:
        """Stable 63-bit fingerprint from CompanyState (process-independent)."""
        snap = state.snapshot()
        acc = 0
        for key in sorted(snap.keys()):
            val = snap[key]
            piece = f"{key}={round(float(val), 6)}".encode("utf-8")
            acc ^= _stable_u64_bytes(piece)
        return acc % (2**63)


def _stable_u64(s: str) -> int:
    return _stable_u64_bytes(s.encode("utf-8"))


def _stable_u64_bytes(b: bytes) -> int:
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "big")
