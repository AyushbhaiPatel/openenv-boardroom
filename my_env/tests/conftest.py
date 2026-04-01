# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared pytest configuration and fixtures for OpenBoardroom tests."""

import pytest
from hypothesis import settings

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------

settings.register_profile("ci", max_examples=200)
settings.register_profile("dev", max_examples=100)
settings.load_profile("dev")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_environment():
    """Return a BoardroomEnvironment reset with seed=42 and difficulty='medium'."""
    from my_env.server.boardroom_environment import BoardroomEnvironment

    env = BoardroomEnvironment()
    env.reset(seed=42, difficulty="medium")
    return env


@pytest.fixture()
def sample_actions():
    """Return a list of BoardroomAction instances, one per action type."""
    from my_env.models import BoardroomAction

    return [
        BoardroomAction(
            action_type="query_data",
            parameters={"metric": "revenue"},
        ),
        BoardroomAction(
            action_type="analyze_trend",
            parameters={"metric": "churn_rate", "quarters": 2},
        ),
        BoardroomAction(
            action_type="simulate_counterfactual",
            parameters={
                "decision": "increase_ad_spend",
                "parameters": {"amount": 50000},
            },
        ),
        BoardroomAction(
            action_type="consult_stakeholder",
            parameters={"stakeholder": "analyst"},
        ),
        BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": "increase_ad_spend",
                "parameters": {"amount": 50000},
                "explanation": "Revenue data shows 15% growth opportunity. "
                "The analyst recommends this approach, though there is "
                "some risk the CEO might disagree.",
            },
        ),
    ]


@pytest.fixture()
def sample_company_state():
    """Return a CompanyState with realistic values."""
    from my_env.models import CompanyState

    return CompanyState(
        revenue=500_000.0,
        monthly_active_users=50_000,
        churn_rate=0.07,
        ad_spend=25_000.0,
        cac=120.0,
        ltv=1_500.0,
        support_load=0.55,
        release_risk=0.44,
        quarter=1,
        history=[],
    )
