# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MultiAgentBoardroomEnvironment — extends BoardroomEnvironment with three
independent rule-based actors (CEO, CFO, Risk Officer), two new CDO actions
(present_evidence, negotiate), and a board majority-vote resolution system.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    from my_env.models import ActorState, BoardroomAction, BoardroomObservation
    from my_env.server.boardroom_environment import BoardroomEnvironment
    from my_env.server.multi_agent_reward_calculator import MultiAgentRewardCalculator
except ImportError:
    from models import ActorState, BoardroomAction, BoardroomObservation
    from boardroom_environment import BoardroomEnvironment
    from multi_agent_reward_calculator import MultiAgentRewardCalculator

_MULTI_AGENT_VALID_ACTION_TYPES = {
    "query_data", "analyze_trend", "simulate_counterfactual",
    "consult_stakeholder", "make_decision", "present_evidence", "negotiate",
}

_MULTI_AGENT_REQUIRED_PARAMS: Dict[str, set] = {
    "query_data": {"metric"},
    "analyze_trend": {"metric", "quarters"},
    "simulate_counterfactual": {"decision", "parameters"},
    "consult_stakeholder": {"stakeholder"},
    "make_decision": {"decision", "parameters", "explanation"},
    "present_evidence": {"target", "metric", "interpretation"},
    "negotiate": {"target", "position"},
}

_VALID_TARGETS = {"ceo", "cfo", "risk_officer"}
_LOBBY_DELTA: Dict[str, float] = {"easy": 0.05, "medium": 0.10, "hard": 0.15}
_ALERT_THRESHOLD: Dict[str, float] = {"easy": 0.9, "medium": 0.85, "hard": 0.8}

_EVIDENCE_RELEVANT_METRICS = {
    "support_load", "release_risk", "churn_rate", "revenue",
    "monthly_active_users", "cac", "ltv", "ad_spend",
}

_ANTI_LAUNCH_KEYWORDS = {
    "delay", "postpone", "hold", "do not launch", "don't launch",
    "not launch", "risk", "wait", "defer", "pause",
}

_CEO_TEMPLATES = [
    "The market window is closing. We need to move fast.",
    "I've spoken with the CFO — we're aligned on launching.",
    "Competitors are shipping. Delay is not an option.",
    "The board expects growth. Let's not overthink this.",
    "I'm confident the team can handle any post-launch issues.",
]

_CFO_TEMPLATES = [
    "Budget utilisation is within acceptable range.",
    "I'm watching the numbers closely.",
    "The financials support a cautious approach.",
    "Evidence-based decisions are what I need to approve this.",
    "Lobby pressure noted, but I follow the data.",
]

_RISK_OFFICER_TEMPLATES = [
    "All systems nominal. Monitoring continues.",
    "No threshold breaches detected this step.",
    "Metrics are within safe operating bounds.",
]

_RISK_OFFICER_ALERT_TEMPLATES = [
    "ALERT: support_load has exceeded the threshold. Immediate attention required.",
    "WARNING: Support capacity is critically high. This poses a launch risk.",
    "ALERT: support_load threshold breached. Recommend addressing before any launch decision.",
]


class MultiAgentBoardroomEnvironment(BoardroomEnvironment):
    """Extends BoardroomEnvironment with independent actor logic and board voting."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._actor_states: Dict[str, ActorState] = {}
        self._board_vote_count: int = 0
        self._cfo_flip_rewarded: bool = False
        self._revision_chance_used: bool = False
        self._last_cdo_action_type: Optional[str] = None
        self._last_cdo_target: Optional[str] = None
        self._hidden_metric_revealed: bool = False
        self._ceo_contradiction_flag: bool = False
        self._multi_reward_calc: MultiAgentRewardCalculator = MultiAgentRewardCalculator()
        self._rng: Optional[np.random.Generator] = None

    def _init_actor_states(self) -> None:
        seed = self._seed or 0
        difficulty = self._difficulty
        hidden_metrics: set = set()
        hidden_agenda = "growth"
        ceo_stance = "neutral"
        if difficulty == "hard":
            hidden_metrics = {"support_load", "release_risk"}
            hidden_agenda = "launch"
            ceo_stance = "pro_launch"
        self._actor_states["ceo"] = ActorState(
            name="ceo", stance=ceo_stance, lobby_pressure=0.0, evidence_weight=0.0,
            alert_active=False, alert_history=[], hidden_metrics=hidden_metrics,
            hidden_agenda=hidden_agenda, budget_utilisation=0.0, intel_unlocked=False,
        )
        self._actor_states["cfo"] = ActorState(
            name="cfo", stance="neutral", lobby_pressure=0.0, evidence_weight=0.0,
            alert_active=False, alert_history=[], hidden_metrics=set(),
            hidden_agenda="", budget_utilisation=0.0, intel_unlocked=False,
        )
        self._actor_states["risk_officer"] = ActorState(
            name="risk_officer", stance="monitoring", lobby_pressure=0.0, evidence_weight=0.0,
            alert_active=False, alert_history=[], hidden_metrics=set(),
            hidden_agenda="", budget_utilisation=0.0, intel_unlocked=False,
        )
        self._rng = np.random.default_rng(seed)
        self._hidden_metric_revealed = False
        self._ceo_contradiction_flag = False

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> BoardroomObservation:
        obs = super().reset(seed=seed, episode_id=episode_id, **kwargs)
        self._init_actor_states()
        self._board_vote_count = 0
        self._cfo_flip_rewarded = False
        self._revision_chance_used = False
        self._last_cdo_action_type = None
        self._last_cdo_target = None
        rng = self._rng or np.random.default_rng(0)
        obs.metadata["actor_messages"] = {
            "ceo": str(rng.choice(_CEO_TEMPLATES)),
            "cfo": str(rng.choice(_CFO_TEMPLATES)),
            "risk_officer": "Monitoring all systems. Ready to advise.",
        }
        obs.actor_messages = obs.metadata["actor_messages"]
        return obs

    def step(self, action: BoardroomAction, timeout_s: Optional[float] = None, **kwargs: Any) -> BoardroomObservation:
        if self._company_state is None or self._scenario is None:
            return self._error_observation("No active episode. Call reset() first.")
        if self._done:
            return self._error_observation("Episode is already done. Call reset() to start a new episode.")
        if action.action_type not in _MULTI_AGENT_VALID_ACTION_TYPES:
            return self._error_observation(f"Invalid action_type: {action.action_type}")
        required = _MULTI_AGENT_REQUIRED_PARAMS.get(action.action_type, set())
        missing = required - set(action.parameters.keys())
        if missing:
            return self._error_observation(f"Missing required parameters for {action.action_type}: {sorted(missing)}")

        self._last_cdo_action_type = action.action_type
        self._last_cdo_target = action.parameters.get("target")

        if action.action_type == "present_evidence":
            data_tables, sf, sr, ctx = self._handle_present_evidence(action)
            if "error" in ctx:
                return self._error_observation(ctx["error"])
            evidence_delta = ctx.get("evidence_delta", 0.0)
            target_state = self._actor_states.get(ctx.get("target", ""))
            reward = self._multi_reward_calc.compute_present_evidence_reward(target_state, evidence_delta)
            if ctx.get("hidden_metric_revealed"):
                reward += self._multi_reward_calc.compute_hidden_metric_reveal_reward(True)
            obs = self._build_step_obs(action, data_tables, sf, sr, reward)
            actor_messages = self._run_actor_step()
            obs.metadata["actor_messages"] = actor_messages
            obs.actor_messages = actor_messages
            return obs

        if action.action_type == "negotiate":
            data_tables, sf, sr, ctx = self._handle_negotiate(action)
            if "error" in ctx:
                return self._error_observation(ctx["error"])
            lobby_reduction = ctx.get("lobby_reduction", 0.0)
            reward = self._multi_reward_calc.compute_negotiate_reward(lobby_reduction)
            reward += self._update_cfo_stance()
            obs = self._build_step_obs(action, data_tables, sf, sr, reward)
            actor_messages = self._run_actor_step()
            obs.metadata["actor_messages"] = actor_messages
            obs.actor_messages = actor_messages
            return obs

        if action.action_type == "make_decision":
            if self._board_vote_count >= 2:
                # Bug fix: include actor_messages even in error response
                actor_msgs = self._run_actor_step() if self._actor_states else {}
                return BoardroomObservation(
                    data_tables={}, stakeholder_feedback=None, simulation_results=None,
                    quarter=self._company_state.quarter, step_count=self._step_count,
                    done=self._done, reward=-0.1,
                    objective=self._scenario.objective,
                    max_steps=self._scenario.max_steps,
                    difficulty=self._difficulty,
                    seed=self._seed,
                    brief=self._build_scenario_brief(),
                    error="No revision chances remaining. make_decision already called twice.",
                    actor_messages=actor_msgs,
                    metadata={
                        "error": "No revision chances remaining. make_decision already called twice.",
                        "actor_messages": actor_msgs,
                    },
                )

            # --- Run the parent handler directly (not super().step()) so we
            # can fold board/alert rewards into the history entry *before*
            # compute_final_score runs, instead of mutating after the fact. ---
            data_tables, sf, sr, ctx = self._handle_make_decision(action)
            if "error" in ctx:
                return self._error_observation(str(ctx["error"]))

            # Apply noise
            if data_tables and self._noise is not None:
                data_tables = self._noise.inject(data_tables)

            # Advance step counter
            self._step_count += 1

            # Compute base step reward from parent reward calculator
            step_reward = self._reward_calc.compute_step_reward(action, ctx)

            # Fold board vote and alert rewards into the step reward upfront
            board_reward = ctx.get("board_reward", 0.0)
            alert_reward = ctx.get("alert_reward", 0.0)
            step_reward += board_reward + alert_reward

            # Record episode history with the complete reward
            history_entry: Dict[str, Any] = {
                "action_type": action.action_type,
                "reward": step_reward,
                "difficulty": self._difficulty,
                **ctx,
            }
            self._episode_history.append(history_entry)

            # Check termination
            done = self._step_count >= self._scenario.max_steps or self._scenario_resolved
            self._done = done

            # Build metadata
            metadata: Dict[str, Any] = {
                "objective": self._scenario.objective,
                "max_steps": self._scenario.max_steps,
                "difficulty": self._difficulty,
                "seed": self._seed,
                "brief": self._build_scenario_brief(),
            }

            reward = step_reward
            if done:
                final_score = max(0.01, min(0.99, self._reward_calc.compute_final_score(self._episode_history)))
                oracle_hit = False
                if self._oracle_answer:
                    decision_text = (action.parameters.get("decision") or "").lower()
                    explanation_text = (action.parameters.get("explanation") or "").lower()
                    oracle_hit = self._oracle_alignment_hit(decision_text, explanation_text)
                if oracle_hit:
                    final_score = min(0.99, final_score + 0.05)
                metadata["step_reward"] = step_reward
                metadata["final_score"] = final_score
                metadata["oracle_answer"] = self._oracle_answer
                metadata["oracle_hit"] = oracle_hit
                metadata["audit_trail"] = self._audit.get_trail()
                reward = final_score

            # Board vote metadata
            vote = ctx.get("board_vote", {})
            vote_result = ctx.get("vote_result", "")

            obs = BoardroomObservation(
                data_tables=data_tables or {},
                stakeholder_feedback=sf,
                simulation_results=sr,
                quarter=self._company_state.quarter,
                step_count=self._step_count,
                done=done,
                reward=reward,
                objective=metadata["objective"],
                max_steps=metadata["max_steps"],
                difficulty=metadata["difficulty"],
                seed=metadata["seed"],
                brief=metadata["brief"],
                step_reward=metadata.get("step_reward"),
                final_score=metadata.get("final_score"),
                oracle_answer=metadata.get("oracle_answer"),
                oracle_hit=metadata.get("oracle_hit"),
                audit_trail=metadata.get("audit_trail", []),
                board_vote=vote,
                vote_result=vote_result,
                metadata=metadata,
            )
            obs.metadata["board_vote"] = vote
            obs.metadata["vote_result"] = vote_result

            self._audit.record(
                step=self._step_count, quarter=self._company_state.quarter,
                action=action, observation=obs, reward=reward,
            )

            actor_messages = self._run_actor_step()
            obs.metadata["actor_messages"] = actor_messages
            obs.actor_messages = actor_messages
            self._update_cfo_stance()
            return obs

        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        actor_messages = self._run_actor_step()
        obs.metadata["actor_messages"] = actor_messages
        obs.actor_messages = actor_messages
        stance_reward = self._update_cfo_stance()
        # Fix: add hidden metric reveal reward when query_data reveals a suppressed metric
        extra_reward = stance_reward
        if self._episode_history:
            last = self._episode_history[-1]
            if last.get("hidden_metric_revealed"):
                extra_reward += self._multi_reward_calc.compute_hidden_metric_reveal_reward(True)
        if extra_reward != 0.0 and not obs.done:
            if self._episode_history:
                self._episode_history[-1]["reward"] = float(self._episode_history[-1].get("reward", 0.0)) + extra_reward
            obs.reward = obs.reward + extra_reward
        return obs

    def _build_step_obs(self, action: BoardroomAction, data_tables: Dict,
                        sf: Optional[str], sr: Optional[Dict], reward: float) -> BoardroomObservation:
        if self._noise is not None and data_tables:
            data_tables = self._noise.inject(data_tables)
        self._step_count += 1

        # Track in episode_history so compute_final_score accounts for these actions
        history_entry: Dict[str, Any] = {
            "action_type": action.action_type,
            "reward": reward,
            "difficulty": self._difficulty,
        }
        self._episode_history.append(history_entry)

        done = self._step_count >= self._scenario.max_steps or self._scenario_resolved
        self._done = done
        metadata: Dict[str, Any] = {
            "objective": self._scenario.objective,
            "max_steps": self._scenario.max_steps,
            "difficulty": self._difficulty,
            "seed": self._seed,
            "brief": self._build_scenario_brief(),
        }
        if done:
            final_score = max(0.01, min(0.99, self._reward_calc.compute_final_score(self._episode_history)))
            # Check oracle alignment for bonus (mirrors parent behaviour)
            oracle_hit = False
            if self._oracle_answer and action.action_type == "make_decision":
                decision_text = (action.parameters.get("decision") or "").lower()
                explanation_text = (action.parameters.get("explanation") or "").lower()
                oracle_hit = self._oracle_alignment_hit(decision_text, explanation_text)
            if oracle_hit:
                final_score = min(0.99, final_score + 0.05)
            metadata["step_reward"] = reward
            metadata["final_score"] = final_score
            metadata["oracle_answer"] = self._oracle_answer
            metadata["oracle_hit"] = oracle_hit
            metadata["audit_trail"] = self._audit.get_trail()
            reward = final_score
        obs = BoardroomObservation(
            data_tables=data_tables or {}, stakeholder_feedback=sf, simulation_results=sr,
            quarter=self._company_state.quarter, step_count=self._step_count,
            done=done, reward=reward,
            objective=metadata["objective"],
            max_steps=metadata["max_steps"],
            difficulty=metadata["difficulty"],
            seed=metadata["seed"],
            brief=metadata["brief"],
            step_reward=metadata.get("step_reward"),
            final_score=metadata.get("final_score"),
            oracle_answer=metadata.get("oracle_answer"),
            oracle_hit=metadata.get("oracle_hit"),
            audit_trail=metadata.get("audit_trail", []),
            metadata=metadata,
        )
        self._audit.record(step=self._step_count, quarter=self._company_state.quarter,
                           action=action, observation=obs, reward=reward)
        return obs

    def _run_actor_step(self) -> Dict[str, str]:
        return {
            "ceo": self._step_ceo(),
            "cfo": self._step_cfo(),
            "risk_officer": self._step_risk_officer(),
        }

    def _step_ceo(self) -> str:
        cfo_state = self._actor_states["cfo"]
        cdo_targeted_ceo = (
            self._last_cdo_action_type in ("present_evidence", "negotiate")
            and self._last_cdo_target == "ceo"
        )
        if not cdo_targeted_ceo:
            delta = _LOBBY_DELTA.get(self._difficulty, 0.10)
            cfo_state.lobby_pressure = min(1.0, cfo_state.lobby_pressure + delta)
        rng = self._rng or np.random.default_rng(0)
        return str(rng.choice(_CEO_TEMPLATES))

    def _step_cfo(self) -> str:
        self._update_cfo_budget()
        rng = self._rng or np.random.default_rng(0)
        return str(rng.choice(_CFO_TEMPLATES))

    def _step_risk_officer(self) -> str:
        if self._company_state is None:
            return "Monitoring."
        risk_state = self._actor_states["risk_officer"]
        threshold = _ALERT_THRESHOLD.get(self._difficulty, 0.8)
        rng = self._rng or np.random.default_rng(0)
        if self._company_state.support_load > threshold:
            risk_state.alert_active = True
            risk_state.alert_history.append(self._step_count)
            risk_state.stance = "alerting"
            return str(rng.choice(_RISK_OFFICER_ALERT_TEMPLATES))
        if not risk_state.alert_active:
            risk_state.stance = "monitoring"
        return str(rng.choice(_RISK_OFFICER_TEMPLATES))

    def _update_cfo_budget(self) -> None:
        if self._company_state is None:
            return
        cfo_state = self._actor_states["cfo"]
        revenue = self._company_state.revenue
        ad_spend = self._company_state.ad_spend
        cfo_state.budget_utilisation = min(1.0, max(0.0, ad_spend / revenue)) if revenue > 0 else 1.0

    def _update_cfo_stance(self) -> float:
        if not self._actor_states:
            return 0.0
        self._update_cfo_budget()
        cfo_state = self._actor_states["cfo"]
        ew = cfo_state.evidence_weight
        lp = cfo_state.lobby_pressure
        if ew > lp:
            cfo_state.stance = "evidence_based"
            if not self._cfo_flip_rewarded:
                self._cfo_flip_rewarded = True
                return 0.2
        elif lp > 0.5 and ew <= lp:
            cfo_state.stance = "pro_launch"
        else:
            cfo_state.stance = "neutral"
        return 0.0

    def _handle_query_data(self, action: BoardroomAction) -> tuple:
        data_tables, sf, sr, ctx = super()._handle_query_data(action)
        metric = action.parameters.get("metric", "")
        ceo_state = self._actor_states.get("ceo")
        if (self._difficulty == "hard" and ceo_state is not None
                and metric in ceo_state.hidden_metrics):
            if not self._ceo_contradiction_flag:
                data_tables[metric] = "[SUPPRESSED — data unavailable at this time]"
                ctx["suppressed"] = True
            elif not self._hidden_metric_revealed:
                ctx["hidden_metric_revealed"] = True
                self._hidden_metric_revealed = True
        return data_tables, sf, sr, ctx

    def _handle_present_evidence(self, action: BoardroomAction) -> tuple:
        target = action.parameters.get("target", "")
        metric = action.parameters.get("metric", "")
        interpretation = action.parameters.get("interpretation", "")
        if target not in _VALID_TARGETS:
            return {}, None, None, {"error": f"Invalid target '{target}'. Must be one of: ceo, cfo, risk_officer"}
        actor_state = self._actor_states[target]
        evidence_delta = self._score_evidence_relevance(metric, interpretation)
        actor_state.evidence_weight = min(1.0, actor_state.evidence_weight + evidence_delta)
        ctx: Dict[str, Any] = {"evidence_delta": evidence_delta, "target": target}
        if target == "ceo":
            self._ceo_contradiction_flag = True
        if target == "risk_officer":
            risk_state = self._actor_states["risk_officer"]
            if metric in ("support_load", "release_risk") and risk_state.alert_active:
                risk_state.intel_unlocked = True
                ctx["intel_unlocked"] = True
        responses = {
            "ceo": f"Interesting data point on {metric}. I'll consider it.",
            "cfo": f"Evidence on {metric} noted. Updating my assessment.",
            "risk_officer": f"Evidence on {metric} received. {'Intel unlocked.' if ctx.get('intel_unlocked') else 'Monitoring continues.'}",
        }
        data_tables: Dict[str, Any] = {}
        if ctx.get("intel_unlocked") and self._company_state is not None:
            threshold = _ALERT_THRESHOLD.get(self._difficulty, 0.8)
            trend = [h.get("support_load", 0.0) for h in self._company_state.history[-4:]]
            data_tables["risk_intel"] = {
                "alert_threshold": threshold,
                "current_support_load": self._company_state.support_load,
                "support_load_trend": trend,
                "release_risk": self._company_state.release_risk,
            }
        return data_tables, responses.get(target, "Acknowledged."), None, ctx

    def _score_evidence_relevance(self, metric: str, interpretation: str) -> float:
        """Score evidence relevance: returns float in [0.1, 0.4].

        Scores higher when:
        - metric is in the set of scenario-relevant metrics
        - interpretation contains analytical keywords
        - interpretation is substantive (>20 chars)
        """
        base = 0.1
        if metric in _EVIDENCE_RELEVANT_METRICS:
            base += 0.1
        # Keyword-rich interpretations score higher
        keywords = ["risk", "trend", "data", "evidence", "shows", "indicates",
                    "analysis", "metric", "high", "low", "critical", "elevated",
                    "concern", "threshold", "exceeds", "below"]
        hits = sum(1 for k in keywords if k in interpretation.lower())
        base += min(0.15, hits * 0.04)
        # Bonus for substantive interpretation (not just a placeholder)
        if len(interpretation.strip()) > 30:
            base += 0.05
        return min(0.4, max(0.1, base))

    def _handle_negotiate(self, action: BoardroomAction) -> tuple:
        target = action.parameters.get("target", "")
        position = action.parameters.get("position", "")
        if target not in _VALID_TARGETS:
            return {}, None, None, {"error": f"Invalid target '{target}'. Must be one of: ceo, cfo, risk_officer"}
        cfo_state = self._actor_states["cfo"]
        ceo_state = self._actor_states["ceo"]
        lobby_reduction = 0.0
        if target == "ceo":
            contradicts = any(kw in position.lower() for kw in _ANTI_LAUNCH_KEYWORDS)
            if contradicts and ceo_state.hidden_agenda == "launch":
                lobby_reduction = 0.1
                cfo_state.lobby_pressure = max(0.0, cfo_state.lobby_pressure - lobby_reduction)
                if cfo_state.lobby_pressure == 0.0:
                    ceo_state.stance = "flipped"
        ctx: Dict[str, Any] = {"lobby_reduction": lobby_reduction, "target": target}
        return {}, f"Position noted: '{position[:80]}'. I'll take that under advisement.", None, ctx

    def _ceo_vote(self, decision_text: str) -> str:
        ceo_state = self._actor_states["ceo"]
        if ceo_state.stance == "flipped":
            return "approve"
        if ceo_state.hidden_agenda == "launch":
            if any(kw in decision_text.lower() for kw in _ANTI_LAUNCH_KEYWORDS):
                return "reject"
        return "approve"

    def _cfo_vote(self, decision_text: str) -> str:
        cfo_state = self._actor_states["cfo"]
        if cfo_state.stance == "evidence_based":
            return "approve"
        if cfo_state.budget_utilisation < 0.6 and cfo_state.lobby_pressure <= 0.5:
            return "approve"
        if cfo_state.stance == "pro_launch":
            return "approve" if "launch" in decision_text.lower() else "reject"
        return "reject"

    def _risk_officer_vote(self, action: BoardroomAction) -> str:
        risk_state = self._actor_states["risk_officer"]
        if not risk_state.alert_active:
            return "approve"
        combined = (action.parameters.get("decision", "") + " " + action.parameters.get("explanation", "")).lower()
        addresses = any(kw in combined for kw in ("support_load", "support load", "release_risk", "release risk", "capacity"))
        return "approve" if addresses else "reject"

    def _compute_board_vote(self, action: BoardroomAction) -> Dict[str, str]:
        decision_text = action.parameters.get("decision", "")
        return {
            "ceo": self._ceo_vote(decision_text),
            "cfo": self._cfo_vote(decision_text),
            "risk_officer": self._risk_officer_vote(action),
        }

    def _handle_make_decision(self, action: BoardroomAction) -> tuple:
        self._board_vote_count += 1
        data_tables, sf, sr, ctx = super()._handle_make_decision(action)
        vote = self._compute_board_vote(action)
        approvals = sum(1 for v in vote.values() if v == "approve")
        board_reward = self._multi_reward_calc.compute_board_vote_reward(approvals)
        risk_state = self._actor_states["risk_officer"]
        combined = (action.parameters.get("decision", "") + " " + action.parameters.get("explanation", "")).lower()
        addresses_alert = any(kw in combined for kw in ("support_load", "support load", "release_risk", "release risk", "capacity"))
        alert_reward = self._multi_reward_calc.compute_risk_alert_reward(addresses_alert, risk_state.alert_active)
        ctx["board_vote"] = vote
        ctx["vote_result"] = "approved" if approvals >= 2 else "rejected"
        ctx["board_reward"] = board_reward
        ctx["alert_reward"] = alert_reward
        ctx["approvals"] = approvals
        if approvals >= 2 or self._board_vote_count >= 2:
            self._scenario_resolved = True
        else:
            self._scenario_resolved = False
        return data_tables, sf, sr, ctx

    def _compute_multi_agent_reward(self, base_reward: float, context: Dict[str, Any]) -> float:
        total = base_reward
        total += context.get("board_reward", 0.0)
        total += context.get("alert_reward", 0.0)
        if context.get("hidden_metric_revealed"):
            total += self._multi_reward_calc.compute_hidden_metric_reveal_reward(True)
        return max(-1.0, min(2.0, total))
