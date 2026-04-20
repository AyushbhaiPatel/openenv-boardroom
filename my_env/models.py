# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the OpenBoardroom Environment.

Defines the action/observation Pydantic models for the CDO decision-making
environment, plus internal dataclasses for company state, scenario config,
and audit trail entries.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


class BoardroomAction(Action):
    """Action for the OpenBoardroom environment.

    The agent selects an action type with action-specific parameters.
    """

    action_type: Literal[
        "query_data",
        "analyze_trend",
        "simulate_counterfactual",
        "consult_stakeholder",
        "make_decision",
        "present_evidence",
        "negotiate",
    ] = Field(..., description="Type of action to take")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters",
    )

    @field_validator("parameters", mode="before")
    @classmethod
    def _parse_parameters(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {}
        return v


class BoardroomObservation(Observation):
    """Observation from the OpenBoardroom environment.

    Contains queried data, stakeholder feedback, simulation results,
    and episode progress. Inherits done, reward, and metadata from Observation.
    """

    data_tables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Queried data tables and metrics",
    )
    stakeholder_feedback: Optional[str] = Field(
        default=None,
        description="Feedback from consulted stakeholder",
    )
    simulation_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Counterfactual simulation output",
    )
    quarter: int = Field(default=1, description="Current quarter number")
    step_count: int = Field(default=0, description="Steps taken this episode")
    objective: Optional[str] = Field(default=None, description="Episode objective")
    max_steps: Optional[int] = Field(default=None, description="Maximum episode length")
    difficulty: Optional[str] = Field(default=None, description="Difficulty tier")
    seed: Optional[int] = Field(default=None, description="Episode seed")
    brief: Optional[str] = Field(default=None, description="Scenario brief")
    step_reward: Optional[float] = Field(default=None, description="Raw reward for the final action")
    final_score: Optional[float] = Field(default=None, description="Final episode score")
    oracle_answer: Optional[str] = Field(default=None, description="Hidden oracle answer revealed at episode end")
    oracle_hit: Optional[bool] = Field(default=None, description="Whether the final decision matched the oracle")
    audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Episode audit trail revealed at episode end",
    )
    error: Optional[str] = Field(default=None, description="Environment error message")
    actor_messages: Dict[str, str] = Field(
        default_factory=dict,
        description="Latest messages from multi-agent board actors",
    )
    board_vote: Dict[str, str] = Field(
        default_factory=dict,
        description="Board approval breakdown for a final decision",
    )
    vote_result: Optional[str] = Field(
        default=None,
        description="Board vote result: approved or rejected",
    )


@dataclass
class CompanyState:
    """Internal mutable state tracking the simulated company across quarters."""

    revenue: float = 0.0
    monthly_active_users: int = 0
    churn_rate: float = 0.05
    ad_spend: float = 0.0
    cac: float = 0.0
    ltv: float = 0.0
    support_load: float = 0.0
    release_risk: float = 0.0
    quarter: int = 1
    history: List[Dict] = field(default_factory=list)

    def snapshot(self) -> Dict:
        """Return a dictionary snapshot of current metrics."""
        return {
            "revenue": self.revenue,
            "monthly_active_users": self.monthly_active_users,
            "churn_rate": self.churn_rate,
            "ad_spend": self.ad_spend,
            "cac": self.cac,
            "ltv": self.ltv,
            "support_load": self.support_load,
            "release_risk": self.release_risk,
            "quarter": self.quarter,
        }

    def to_tensor_input(self) -> List[float]:
        """Flatten to feature vector for the counterfactual MLP."""
        return [
            self.revenue / 1e6,
            self.monthly_active_users / 1e5,
            self.churn_rate,
            self.ad_spend / 1e5,
            self.cac / 100,
            self.ltv / 1000,
            self.support_load,
            self.release_risk,
        ]


@dataclass
class ScenarioConfig:
    """Configuration for a generated scenario episode."""

    difficulty: str  # "easy", "medium", "hard"
    objective: str  # Description of the scenario goal
    max_steps: int  # 10, 20, or 30
    noise_level: float  # 0.0, 0.1, 0.2
    has_misleading_signal: bool  # True only for hard


@dataclass
class AuditEntry:
    """Single entry in the episode audit trail."""

    step: int
    quarter: int
    action_type: str
    action_params: Dict[str, Any]
    reward: float
    timestamp: str  # ISO 8601


@dataclass
class ActorState:
    """State for a multi-agent boardroom participant."""

    name: str
    stance: str
    lobby_pressure: float = 0.0
    evidence_weight: float = 0.0
    alert_active: bool = False
    alert_history: list = field(default_factory=list)
    hidden_metrics: set = field(default_factory=set)
    hidden_agenda: str = ""
    budget_utilisation: float = 0.0
    intel_unlocked: bool = False
