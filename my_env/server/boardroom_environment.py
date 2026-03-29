# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
BoardroomEnvironment — main orchestrator for the OpenBoardroom RL environment.

Extends the OpenEnv ``Environment`` base class and wires together all
subsystems: data generation, stakeholder simulation, counterfactual engine,
reward calculation, explanation grading, noise injection, and audit trail.
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from my_env.models import (
    BoardroomAction,
    BoardroomObservation,
    CompanyState,
    ScenarioConfig,
)
from my_env.server.audit_trail import AuditTrail
from my_env.server.counterfactual_engine import CounterfactualEngine
from my_env.server.data_generator import SyntheticDataGenerator
from my_env.server.explanation_grader import ExplanationGrader
from my_env.server.noise_injector import NoiseInjector
from my_env.server.reward_calculator import RewardCalculator
from my_env.server.stakeholder_simulator import StakeholderSimulator

# ---------------------------------------------------------------------------
# Scenario definitions per difficulty tier
# ---------------------------------------------------------------------------

_SCENARIO_DEFS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "objective": "Find the growth bottleneck",
        "max_steps": 10,
        "noise_level": 0.0,
        "has_misleading_signal": False,
        # Candidate oracle answers — one is chosen deterministically from seed
        "oracle_candidates": ["churn_rate", "cac", "monthly_active_users"],
    },
    "medium": {
        "objective": "Diagnose the revenue drop",
        "max_steps": 20,
        "noise_level": 0.1,
        "has_misleading_signal": False,
        "oracle_candidates": ["churn_rate", "ad_spend", "cac"],
    },
    "hard": {
        "objective": "Should we launch Feature X?",
        "max_steps": 30,
        "noise_level": 0.2,
        "has_misleading_signal": True,
        # Two possible verdicts for the launch decision
        "oracle_candidates": ["launch", "do not launch"],
    },
}

# Metrics that are considered "relevant" for each scenario objective.
_RELEVANT_METRICS: Dict[str, set] = {
    "Find the growth bottleneck": {"revenue", "monthly_active_users", "churn_rate", "cac", "ltv"},
    "Diagnose the revenue drop": {"revenue", "churn_rate", "ad_spend", "cac"},
    "Should we launch Feature X?": {"revenue", "monthly_active_users", "churn_rate", "ad_spend", "cac", "ltv"},
}

# Valid action types
_VALID_ACTION_TYPES = {
    "query_data",
    "analyze_trend",
    "simulate_counterfactual",
    "consult_stakeholder",
    "make_decision",
}

# Required parameter keys per action type
_REQUIRED_PARAMS: Dict[str, set] = {
    "query_data": {"metric"},
    "analyze_trend": {"metric", "quarters"},
    "simulate_counterfactual": {"decision", "parameters"},
    "consult_stakeholder": {"stakeholder"},
    "make_decision": {"decision", "parameters", "explanation"},
}


class BoardroomEnvironment(Environment[BoardroomAction, BoardroomObservation, State]):
    """OpenBoardroom RL environment orchestrator.

    Manages a multi-step episode where an AI agent acts as CDO of a
    simulated SaaS company. Wires together data generation, stakeholder
    simulation, counterfactual reasoning, reward computation, explanation
    grading, noise injection, and audit logging.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Episode state — populated on reset()
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._done: bool = False
        self._seed: Optional[int] = None
        self._difficulty: str = "medium"
        self._scenario: Optional[ScenarioConfig] = None
        self._company_state: Optional[CompanyState] = None

        # Subsystems — initialised on reset()
        self._data_gen: Optional[SyntheticDataGenerator] = None
        self._stakeholders: Optional[StakeholderSimulator] = None
        self._cf_engine: Optional[CounterfactualEngine] = None
        self._noise: Optional[NoiseInjector] = None
        self._reward_calc: RewardCalculator = RewardCalculator()
        self._grader: ExplanationGrader = ExplanationGrader()
        self._audit: AuditTrail = AuditTrail()

        # Per-episode tracking
        self._consultation_history: List[Dict] = []
        self._episode_history: List[Dict[str, Any]] = []
        self._scenario_resolved: bool = False

    # ------------------------------------------------------------------
    # OpenEnv API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> BoardroomObservation:
        """Reset the environment and start a new episode.

        Args:
            seed: Optional integer seed for deterministic episode generation.
                If not provided, a random seed is generated.
            episode_id: Optional episode identifier. Auto-generated if absent.
            **kwargs: May include ``difficulty`` (``"easy"``/``"medium"``/``"hard"``).

        Returns:
            Initial :class:`BoardroomObservation` describing the scenario.
        """
        # Determine seed
        if seed is not None:
            self._seed = seed
        else:
            self._seed = random.randint(0, 2**32 - 1)

        self._episode_id = episode_id or str(uuid4())
        self._difficulty = kwargs.get("difficulty", "medium")
        if self._difficulty not in _SCENARIO_DEFS:
            self._difficulty = "medium"

        # Build scenario config
        sdef = _SCENARIO_DEFS[self._difficulty]
        self._scenario = ScenarioConfig(
            difficulty=self._difficulty,
            objective=sdef["objective"],
            max_steps=sdef["max_steps"],
            noise_level=sdef["noise_level"],
            has_misleading_signal=sdef["has_misleading_signal"],
        )

        # Choose oracle answer deterministically from seed (hidden from agent)
        candidates = sdef.get("oracle_candidates", [])
        self._oracle_answer: str = (
            candidates[self._seed % len(candidates)] if candidates else ""
        )

        # Initialise subsystems
        self._data_gen = SyntheticDataGenerator(self._seed)
        self._company_state = self._data_gen.generate_initial_state(self._difficulty)
        self._stakeholders = StakeholderSimulator(self._seed)
        self._cf_engine = CounterfactualEngine(self._seed)
        self._noise = NoiseInjector(self._seed, self._difficulty)

        # Reset episode tracking
        self._step_count = 0
        self._done = False
        self._scenario_resolved = False
        self._consultation_history = []
        self._episode_history = []
        self._audit.clear()

        # Build initial observation
        metadata: Dict[str, Any] = {
            "seed": self._seed,
            "difficulty": self._difficulty,
            "objective": self._scenario.objective,
            "max_steps": self._scenario.max_steps,
        }

        return BoardroomObservation(
            data_tables=self._company_state.snapshot(),
            stakeholder_feedback=None,
            simulation_results=None,
            quarter=self._company_state.quarter,
            step_count=self._step_count,
            done=False,
            reward=0.0,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # OpenEnv API: step
    # ------------------------------------------------------------------

    def step(
        self,
        action: BoardroomAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> BoardroomObservation:
        """Execute one step in the environment.

        Validates the action, dispatches to the appropriate handler, applies
        noise, computes reward, records audit, checks termination, and
        returns an observation.
        """
        # Guard: episode not started
        if self._company_state is None or self._scenario is None:
            return self._error_observation("No active episode. Call reset() first.")

        # Guard: episode already done
        if self._done:
            return self._error_observation("Episode is already done. Call reset() to start a new episode.")

        # Validate action type
        if action.action_type not in _VALID_ACTION_TYPES:
            return self._error_observation(f"Invalid action_type: {action.action_type}")

        # Validate required parameters
        required = _REQUIRED_PARAMS.get(action.action_type, set())
        missing = required - set(action.parameters.keys())
        if missing:
            return self._error_observation(
                f"Missing required parameters for {action.action_type}: {sorted(missing)}"
            )

        # Dispatch to handler
        handler = {
            "query_data": self._handle_query_data,
            "analyze_trend": self._handle_analyze_trend,
            "simulate_counterfactual": self._handle_simulate_counterfactual,
            "consult_stakeholder": self._handle_consult_stakeholder,
            "make_decision": self._handle_make_decision,
        }[action.action_type]

        data_tables, stakeholder_feedback, simulation_results, reward_context = handler(action)

        # Apply noise to data_tables (medium/hard)
        if data_tables and self._noise is not None:
            data_tables = self._noise.inject(data_tables)

        # Advance step counter
        self._step_count += 1

        # Compute reward
        step_reward = self._reward_calc.compute_step_reward(action, reward_context)
        reward = step_reward

        # Record episode history entry
        history_entry = {
            "action_type": action.action_type,
            "reward": step_reward,
            "difficulty": self._difficulty,
            **reward_context,
        }
        self._episode_history.append(history_entry)

        # Check termination
        done = False
        if self._step_count >= self._scenario.max_steps:
            done = True
        if self._scenario_resolved:
            done = True

        self._done = done

        # Episode context on every step (agents / inference need this each turn)
        metadata: Dict[str, Any] = {
            "objective": self._scenario.objective,
            "max_steps": self._scenario.max_steps,
            "difficulty": self._difficulty,
            "seed": self._seed,
        }
        if done:
            final_score = self._reward_calc.compute_final_score(self._episode_history)

            # Oracle bonus: +0.05 if agent's final decision mentions the ground truth
            oracle_hit = False
            if self._oracle_answer and action.action_type == "make_decision":
                decision_text = (action.parameters.get("decision") or "").lower()
                explanation_text = (action.parameters.get("explanation") or "").lower()
                oracle_hit = self._oracle_answer.lower() in decision_text or \
                             self._oracle_answer.lower() in explanation_text
            if oracle_hit:
                final_score = min(1.0, final_score + 0.05)

            metadata["step_reward"] = step_reward
            metadata["final_score"] = final_score
            # Reveal oracle answer at episode end so judges/evaluators can verify
            metadata["oracle_answer"] = self._oracle_answer
            metadata["oracle_hit"] = oracle_hit
            metadata["audit_trail"] = self._audit.get_trail()
            reward = final_score

        obs = BoardroomObservation(
            data_tables=data_tables,
            stakeholder_feedback=stakeholder_feedback,
            simulation_results=simulation_results,
            quarter=self._company_state.quarter,
            step_count=self._step_count,
            done=done,
            reward=reward,
            metadata=metadata,
        )

        # Record to audit trail
        self._audit.record(
            step=self._step_count,
            quarter=self._company_state.quarter,
            action=action,
            observation=obs,
            reward=reward,
        )

        return obs

    # ------------------------------------------------------------------
    # OpenEnv API: state
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return the current environment state."""
        return State(
            episode_id=self._episode_id or "",
            step_count=self._step_count,
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_query_data(
        self, action: BoardroomAction
    ) -> tuple:
        """Handle query_data: return metric value from CompanyState snapshot."""
        metric = action.parameters["metric"]
        snapshot = self._company_state.snapshot()

        data_tables: Dict[str, Any] = {}
        relevant = False

        if metric in snapshot:
            data_tables[metric] = snapshot[metric]
            # Check if metric is relevant to the scenario objective
            relevant_set = _RELEVANT_METRICS.get(self._scenario.objective, set())
            relevant = metric in relevant_set
        else:
            data_tables["error"] = f"Unknown metric: {metric}"

        reward_context = {"relevant": relevant}
        return data_tables, None, None, reward_context

    def _handle_analyze_trend(
        self, action: BoardroomAction
    ) -> tuple:
        """Handle analyze_trend: return trend data from history."""
        metric = action.parameters["metric"]
        quarters = int(action.parameters["quarters"])

        history = self._company_state.history
        # Get the last N quarters of data for the metric
        trend_data: List[Any] = []
        for entry in history[-quarters:]:
            if metric in entry:
                trend_data.append(entry[metric])

        data_tables: Dict[str, Any] = {
            "metric": metric,
            "quarters_requested": quarters,
            "trend": trend_data,
        }

        # Easy: shallow trend is enough. Medium/hard: require more quarters
        # so the agent looks past single-point noise.
        if self._difficulty == "easy":
            noise_handled = quarters >= 2 and len(trend_data) >= 2
        else:
            noise_handled = quarters >= 3 and len(trend_data) >= 3

        reward_context = {"noise_handled": noise_handled}
        return data_tables, None, None, reward_context

    def _handle_simulate_counterfactual(
        self, action: BoardroomAction
    ) -> tuple:
        """Handle simulate_counterfactual: use CounterfactualEngine."""
        decision = action.parameters["decision"]
        params = action.parameters["parameters"]

        params_d = params if isinstance(params, dict) else {}
        results = self._cf_engine.simulate(
            company_state=self._company_state,
            decision=decision,
            params=params_d,
        )

        d = decision.strip().lower()
        biz_keywords = (
            "revenue", "churn", "growth", "spend", "cac", "ltv", "mau", "user",
            "retention", "acquisition", "launch", "cut", "increase", "reduce",
            "invest", "marketing", "onboarding", "feature",
        )
        has_keyword = any(k in d for k in biz_keywords)
        non_empty_params = any(
            v not in (None, "", [], {})
            for v in params_d.values()
        )
        insightful = (len(decision.strip()) >= 16 and has_keyword) or (
            len(decision.strip()) >= 10 and non_empty_params
        )

        reward_context = {"insightful": insightful}
        return {}, None, results, reward_context

    def _handle_consult_stakeholder(
        self, action: BoardroomAction
    ) -> tuple:
        """Handle consult_stakeholder: use StakeholderSimulator."""
        stakeholder = action.parameters["stakeholder"]

        valid_stakeholders = StakeholderSimulator.VALID_STAKEHOLDERS
        if stakeholder not in valid_stakeholders:
            return (
                {},
                f"Unknown stakeholder: {stakeholder}. Valid: {valid_stakeholders}",
                None,
                {"navigation_score": 0.0},
            )

        feedback = self._stakeholders.consult(
            stakeholder=stakeholder,
            company_state=self._company_state,
            context={"objective": self._scenario.objective},
        )

        # Track consultation history
        self._consultation_history.append({"stakeholder": stakeholder})

        # Compute navigation score based on consultation diversity
        nav_score = self._stakeholders.compute_navigation_score(
            self._consultation_history
        )

        reward_context = {"navigation_score": nav_score}
        return {}, feedback, None, reward_context

    def _handle_make_decision(
        self, action: BoardroomAction
    ) -> tuple:
        """Handle make_decision: grade explanation, evolve state, advance quarter."""
        decision = action.parameters["decision"]
        explanation = action.parameters["explanation"]

        # Grade the explanation
        scenario_context = {
            "objective": self._scenario.objective,
            "difficulty": self._difficulty,
        }
        explanation_score = self._grader.grade(explanation, scenario_context)

        decision_lower = decision.lower()
        obj_lower = self._scenario.objective.lower()
        obj_tokens = [w for w in obj_lower.replace("?", "").split() if len(w) > 3]
        overlap_hits = sum(1 for w in obj_tokens if w in decision_lower)
        overlap = min(
            1.0,
            overlap_hits / max(3, len(obj_tokens)) * 1.4,
        ) if obj_tokens else 0.5
        decision_quality = 0.72 * explanation_score + 0.28 * overlap

        # Evolve company state
        self._data_gen.evolve_state(self._company_state, decision_quality)

        # Mark scenario as resolved (make_decision resolves the objective)
        self._scenario_resolved = True

        data_tables = self._company_state.snapshot()

        reward_context = {
            "decision_quality": decision_quality,
            "explanation_score": explanation_score,
        }
        return data_tables, None, None, reward_context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _error_observation(self, message: str) -> BoardroomObservation:
        """Return an error observation without advancing the step counter."""
        return BoardroomObservation(
            data_tables={},
            stakeholder_feedback=None,
            simulation_results=None,
            quarter=self._company_state.quarter if self._company_state else 1,
            step_count=self._step_count,
            done=self._done,
            reward=0.0,
            metadata={"error": message},
        )
