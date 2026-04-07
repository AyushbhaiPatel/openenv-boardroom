# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Synthetic Data Generator for the OpenBoardroom Environment.

Generates scenario-shaped SaaS company states and evolves them across
quarters based on agent decision quality. All randomness is seeded via
numpy for full determinism.
"""

import numpy as np
from typing import Dict

try:
    from ..models import CompanyState
except ImportError:
    from models import CompanyState


# Realistic SaaS metric ranges per difficulty tier.
# Each tier maps to (min, max) tuples for the six core metrics.
_DIFFICULTY_RANGES = {
    "easy": {
        "revenue": (500_000, 5_000_000),
        "monthly_active_users": (50_000, 500_000),
        "churn_rate": (0.02, 0.06),
        "ad_spend": (10_000, 200_000),
        "cac": (10, 150),
        "ltv": (1_000, 5_000),
        "support_load": (0.15, 0.65),
        "release_risk": (0.10, 0.55),
    },
    "medium": {
        "revenue": (200_000, 3_000_000),
        "monthly_active_users": (10_000, 300_000),
        "churn_rate": (0.05, 0.10),
        "ad_spend": (20_000, 300_000),
        "cac": (50, 300),
        "ltv": (500, 3_000),
        "support_load": (0.25, 0.80),
        "release_risk": (0.20, 0.75),
    },
    "hard": {
        "revenue": (50_000, 1_500_000),
        "monthly_active_users": (1_000, 100_000),
        "churn_rate": (0.08, 0.15),
        "ad_spend": (5_000, 150_000),
        "cac": (100, 500),
        "ltv": (100, 1_500),
        "support_load": (0.35, 1.00),
        "release_risk": (0.25, 1.00),
    },
}


_EASY_PROFILES: Dict[str, Dict[str, float]] = {
    "churn_rate": {
        "revenue": 1_450_000,
        "monthly_active_users": 182_000,
        "churn_rate": 0.108,
        "ad_spend": 86_000,
        "cac": 78,
        "ltv": 2_350,
        "support_load": 0.41,
        "release_risk": 0.28,
    },
    "cac": {
        "revenue": 1_220_000,
        "monthly_active_users": 138_000,
        "churn_rate": 0.051,
        "ad_spend": 172_000,
        "cac": 218,
        "ltv": 1_150,
        "support_load": 0.47,
        "release_risk": 0.30,
    },
    "monthly_active_users": {
        "revenue": 910_000,
        "monthly_active_users": 56_000,
        "churn_rate": 0.041,
        "ad_spend": 34_000,
        "cac": 48,
        "ltv": 2_840,
        "support_load": 0.32,
        "release_risk": 0.22,
    },
}

_MEDIUM_PROFILES: Dict[str, Dict[str, float]] = {
    "churn_rate": {
        "revenue": 820_000,
        "monthly_active_users": 118_000,
        "churn_rate": 0.142,
        "ad_spend": 91_000,
        "cac": 104,
        "ltv": 1_160,
        "support_load": 0.63,
        "release_risk": 0.42,
    },
    "ad_spend": {
        "revenue": 905_000,
        "monthly_active_users": 126_000,
        "churn_rate": 0.074,
        "ad_spend": 246_000,
        "cac": 181,
        "ltv": 1_420,
        "support_load": 0.58,
        "release_risk": 0.40,
    },
    "cac": {
        "revenue": 760_000,
        "monthly_active_users": 92_000,
        "churn_rate": 0.082,
        "ad_spend": 164_000,
        "cac": 262,
        "ltv": 980,
        "support_load": 0.52,
        "release_risk": 0.39,
    },
}

_HARD_PROFILES: Dict[str, Dict[str, float]] = {
    "launch": {
        "revenue": 1_180_000,
        "monthly_active_users": 96_000,
        "churn_rate": 0.089,
        "ad_spend": 122_000,
        "cac": 156,
        "ltv": 1_860,
        "support_load": 0.48,
        "release_risk": 0.33,
    },
    "do not launch": {
        "revenue": 685_000,
        "monthly_active_users": 58_000,
        "churn_rate": 0.148,
        "ad_spend": 109_000,
        "cac": 284,
        "ltv": 870,
        "support_load": 0.91,
        "release_risk": 0.88,
    },
}


class SyntheticDataGenerator:
    """Generates and evolves CompanyState using seeded numpy randomness.

    All outputs are fully deterministic for a given seed, satisfying the
    seed-determinism requirement (Req 2.2).
    """

    def __init__(self, seed: int) -> None:
        """Initialise with a numpy random Generator seeded by *seed*."""
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_initial_state(self, difficulty: str, oracle_answer: str = "") -> CompanyState:
        """Create a fresh CompanyState with realistic SaaS metrics.

        Args:
            difficulty: One of ``"easy"``, ``"medium"``, ``"hard"``.

        Returns:
            A new :class:`CompanyState` with quarter=1 and an empty history.

        Ranges (global bounds across all tiers):
            - revenue:  $50 000 – $5 000 000
            - MAU:      1 000 – 500 000
            - churn:    2 % – 15 %
            - CAC:      $10 – $500
            - LTV:      $100 – $5 000
        """
        profile = self._build_profile(difficulty, oracle_answer)

        state = CompanyState(
            revenue=profile["revenue"],
            monthly_active_users=int(profile["monthly_active_users"]),
            churn_rate=profile["churn_rate"],
            ad_spend=profile["ad_spend"],
            cac=profile["cac"],
            ltv=profile["ltv"],
            support_load=profile["support_load"],
            release_risk=profile["release_risk"],
            quarter=1,
            history=[],
        )
        state.history = self._build_history(difficulty, oracle_answer, state)
        return state

    def evolve_state(
        self, state: CompanyState, decision_quality: float
    ) -> CompanyState:
        """Evolve *state* into the next quarter based on *decision_quality*.

        ``decision_quality`` is expected in [0, 1] where 1 is a perfect
        decision and 0 is the worst possible decision.

        Good decisions (high quality) decrease churn and increase revenue.
        Poor decisions (low quality) increase churn and decrease revenue.

        The state is mutated **in-place** and also returned for convenience.
        """
        # Centre quality around 0 so positive = good, negative = bad.
        quality_signal = decision_quality - 0.5  # range [-0.5, 0.5]

        # Base growth/decay rates with small random perturbation.
        base_revenue_growth = 0.03 + quality_signal * 0.10
        base_churn_delta = -quality_signal * 0.02
        base_mau_growth = 0.02 + quality_signal * 0.08
        base_cac_delta = -quality_signal * 0.05
        base_ltv_growth = 0.01 + quality_signal * 0.06

        # Ad spend moves with growth intent (good decisions → measured increase).
        base_ad_growth = quality_signal * 0.08

        # Add small stochastic noise (seeded).
        noise = self._rng.normal(0, 0.005, size=6)

        # Apply changes.
        state.revenue *= 1.0 + base_revenue_growth + noise[0]
        state.monthly_active_users = max(
            1, int(state.monthly_active_users * (1.0 + base_mau_growth + noise[1]))
        )
        state.churn_rate += base_churn_delta + noise[2]
        state.cac *= 1.0 + base_cac_delta + noise[3]
        state.ltv *= 1.0 + base_ltv_growth + noise[4]
        state.ad_spend *= 1.0 + base_ad_growth + noise[5]
        state.ad_spend = max(0.0, state.ad_spend)

        # Clamp to valid ranges.
        state.revenue = max(0.0, state.revenue)
        state.churn_rate = float(np.clip(state.churn_rate, 0.01, 0.30))
        state.cac = max(1.0, state.cac)
        state.ltv = max(1.0, state.ltv)
        state.monthly_active_users = max(1, state.monthly_active_users)

        # Advance quarter and record snapshot.
        state.quarter += 1
        state.history.append(state.snapshot())

        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _uniform(self, low: float, high: float) -> float:
        """Draw a single uniform random value in [low, high)."""
        return float(self._rng.uniform(low, high))

    def _build_profile(self, difficulty: str, oracle_answer: str) -> Dict[str, float]:
        ranges = _DIFFICULTY_RANGES.get(difficulty, _DIFFICULTY_RANGES["medium"])
        if difficulty == "easy":
            base = dict(_EASY_PROFILES.get(oracle_answer, _EASY_PROFILES["churn_rate"]))
        elif difficulty == "medium":
            base = dict(_MEDIUM_PROFILES.get(oracle_answer, _MEDIUM_PROFILES["churn_rate"]))
        else:
            base = dict(_HARD_PROFILES.get(oracle_answer, _HARD_PROFILES["do not launch"]))

        for key, value in list(base.items()):
            jitter = self._rng.normal(0.0, 0.03)
            noisy = value * (1.0 + jitter)
            low, high = ranges[key]
            if key == "monthly_active_users":
                noisy = int(np.clip(noisy, low, high))
            else:
                noisy = float(np.clip(noisy, low, high))
            base[key] = noisy
        return base

    def _build_history(
        self,
        difficulty: str,
        oracle_answer: str,
        state: CompanyState,
    ) -> list[Dict]:
        """Create a short backstory so trend analysis is meaningful from reset()."""
        history: list[Dict] = []
        for lookback in range(3, 0, -1):
            factor = 1.0 - 0.05 * lookback
            churn_bump = 0.0
            revenue_bump = 0.0
            mau_bump = 0.0
            cac_bump = 0.0
            spend_bump = 0.0

            if oracle_answer == "churn_rate":
                churn_bump = -0.010 * lookback
                revenue_bump = 0.06 * lookback
                mau_bump = 0.03 * lookback
            elif oracle_answer == "cac":
                cac_bump = -12.0 * lookback
                spend_bump = -0.08 * lookback
                revenue_bump = 0.035 * lookback
            elif oracle_answer == "ad_spend":
                spend_bump = -0.10 * lookback
                revenue_bump = 0.045 * lookback
            elif oracle_answer == "monthly_active_users":
                mau_bump = -0.08 * lookback
                revenue_bump = 0.05 * lookback
            elif oracle_answer == "launch":
                revenue_bump = -0.02 * lookback
                mau_bump = -0.015 * lookback
                churn_bump = -0.004 * lookback
            elif oracle_answer == "do not launch":
                churn_bump = -0.006 * lookback
                cac_bump = -8.0 * lookback
                revenue_bump = 0.025 * lookback

            history.append(
                {
                    "revenue": max(0.0, state.revenue * (1.0 + revenue_bump)),
                    "monthly_active_users": max(1, int(state.monthly_active_users * (1.0 + mau_bump))),
                    "churn_rate": float(np.clip(state.churn_rate + churn_bump, 0.01, 0.30)),
                    "ad_spend": max(0.0, state.ad_spend * (1.0 + spend_bump + 0.01 * lookback)),
                    "cac": max(1.0, state.cac + cac_bump),
                    "ltv": max(1.0, state.ltv * factor),
                    "support_load": float(np.clip(state.support_load - 0.04 * lookback, 0.05, 1.0)),
                    "release_risk": float(np.clip(state.release_risk - 0.03 * lookback, 0.05, 1.0)),
                    "quarter": max(1, state.quarter - lookback),
                }
            )
        history.append(state.snapshot())
        return history
