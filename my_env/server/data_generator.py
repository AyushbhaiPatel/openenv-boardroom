# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Synthetic Data Generator for the OpenBoardroom Environment.

Generates realistic SaaS company states and evolves them across quarters
based on agent decision quality. All randomness is seeded via numpy for
full determinism.
"""

import numpy as np

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
    },
    "medium": {
        "revenue": (200_000, 3_000_000),
        "monthly_active_users": (10_000, 300_000),
        "churn_rate": (0.05, 0.10),
        "ad_spend": (20_000, 300_000),
        "cac": (50, 300),
        "ltv": (500, 3_000),
    },
    "hard": {
        "revenue": (50_000, 1_500_000),
        "monthly_active_users": (1_000, 100_000),
        "churn_rate": (0.08, 0.15),
        "ad_spend": (5_000, 150_000),
        "cac": (100, 500),
        "ltv": (100, 1_500),
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

    def generate_initial_state(self, difficulty: str) -> CompanyState:
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
        ranges = _DIFFICULTY_RANGES.get(difficulty, _DIFFICULTY_RANGES["medium"])

        revenue = self._uniform(*ranges["revenue"])
        mau = int(self._uniform(*ranges["monthly_active_users"]))
        churn = self._uniform(*ranges["churn_rate"])
        ad_spend = self._uniform(*ranges["ad_spend"])
        cac = self._uniform(*ranges["cac"])
        ltv = self._uniform(*ranges["ltv"])

        state = CompanyState(
            revenue=revenue,
            monthly_active_users=mau,
            churn_rate=churn,
            ad_spend=ad_spend,
            cac=cac,
            ltv=ltv,
            quarter=1,
            history=[],
        )
        # Record the initial snapshot in history.
        state.history.append(state.snapshot())
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
