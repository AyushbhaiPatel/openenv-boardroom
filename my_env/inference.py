"""
LLM-based inference agent for the OpenBoardroom Environment.

Connects to the deployed OpenBoardroom environment on HuggingFace Spaces
and uses an LLM via the OpenAI client to act as CDO.

Required environment variables:
    API_BASE_URL  — The API endpoint for the LLM
    MODEL_NAME    — The model identifier to use
    HF_TOKEN      — Your Hugging Face / API key

Usage:
    python inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from my_env.client import BoardroomEnv
from my_env.models import BoardroomAction

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:novita")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# The deployed HF Space URL for the environment
ENV_SPACE_URL = os.getenv("ENV_SPACE_URL", "http://localhost:8000")

DIFFICULTIES = ["easy", "medium", "hard"]
EPISODES_PER_TIER = 5
SEEDS = list(range(EPISODES_PER_TIER))

VALID_METRICS = ["revenue", "monthly_active_users", "churn_rate",
                 "ad_spend", "cac", "ltv"]
VALID_STAKEHOLDERS = ["analyst", "ceo", "risk_officer"]

# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Chief Data Officer (CDO) at a SaaS company. You are playing \
a strategic decision-making game. Each turn you must choose ONE action.

Available actions (respond with valid JSON only):

1. Query a metric:
   {"action_type": "query_data", "parameters": {"metric": "<metric_name>"}}
   Valid metrics: revenue, monthly_active_users, churn_rate, ad_spend, cac, ltv

2. Analyze a trend:
   {"action_type": "analyze_trend", "parameters": {"metric": "<metric_name>", "quarters": 2}}

3. Consult a stakeholder:
   {"action_type": "consult_stakeholder", "parameters": {"stakeholder": "<name>"}}
   Valid stakeholders: analyst, ceo, risk_officer

4. Run a counterfactual simulation:
   {"action_type": "simulate_counterfactual", "parameters": {"decision": "<description>", "parameters": {}}}

5. Make a final decision (ends the quarter):
   {"action_type": "make_decision", "parameters": {"decision": "<description>", "parameters": {}, "explanation": "<your reasoning>"}}

Strategy: Query key metrics first, consult stakeholders, run simulations, \
then make a well-reasoned decision. Your explanation should reference data, \
acknowledge uncertainty, and consider stakeholder perspectives.

IMPORTANT: Respond with ONLY a single JSON object. No markdown, no commentary.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from LLM output, handling markdown fences."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def obs_to_text(obs) -> str:
    """Convert a BoardroomObservation to a human-readable summary."""
    parts = [f"Step {obs.step_count} | Quarter {obs.quarter}"]
    if obs.data_tables:
        parts.append(f"Data: {json.dumps(obs.data_tables, default=str)}")
    if obs.stakeholder_feedback:
        parts.append(f"Stakeholder feedback: {obs.stakeholder_feedback}")
    if obs.simulation_results:
        parts.append(f"Simulation results: {json.dumps(obs.simulation_results, default=str)}")
    if obs.metadata.get("error"):
        parts.append(f"Error: {obs.metadata['error']}")
    parts.append(f"Reward: {obs.reward}")
    return "\n".join(parts)


def fallback_action(step: int) -> Dict[str, Any]:
    """Deterministic fallback if LLM fails to produce valid JSON."""
    cycle = step % 9
    if cycle < 3:
        return {"action_type": "query_data",
                "parameters": {"metric": VALID_METRICS[cycle]}}
    elif cycle == 3:
        return {"action_type": "analyze_trend",
                "parameters": {"metric": "revenue", "quarters": 2}}
    elif cycle < 7:
        return {"action_type": "consult_stakeholder",
                "parameters": {"stakeholder": VALID_STAKEHOLDERS[cycle - 4]}}
    elif cycle == 7:
        return {"action_type": "simulate_counterfactual",
                "parameters": {"decision": "optimize_growth", "parameters": {}}}
    else:
        return {
            "action_type": "make_decision",
            "parameters": {
                "decision": "balanced_growth",
                "parameters": {},
                "explanation": (
                    "Based on revenue and churn data, a balanced approach is best. "
                    "The analyst recommends caution, the CEO wants growth, and the "
                    "risk officer urges stability. There is uncertainty in the data, "
                    "but the metrics suggest moderate investment."
                ),
            },
        }


# ---------------------------------------------------------------------------
# Episode runner — uses the EnvClient over HTTP/WebSocket
# ---------------------------------------------------------------------------

async def run_episode(
    env: BoardroomEnv,
    llm_client: OpenAI,
    seed: int,
    difficulty: str,
    verbose: bool = False,
) -> float:
    """Run one episode with the LLM agent via the deployed env. Returns final score."""
    result = await env.reset(seed=seed, difficulty=difficulty)
    obs = result.observation
    max_steps = obs.metadata.get("max_steps", 20)
    objective = obs.metadata.get("objective", "")

    history: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"New episode started. Difficulty: {difficulty}\n"
                f"Objective: {objective}\n"
                f"Max steps: {max_steps}\n\n"
                f"Initial observation:\n{obs_to_text(obs)}\n\n"
                "Choose your first action (respond with JSON only):"
            ),
        },
    ]

    while not obs.done:
        # Call the LLM
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=history,
                max_tokens=512,
                temperature=0.3,
            )
            assistant_msg = response.choices[0].message.content.strip()
        except Exception as e:
            if verbose:
                print(f"  LLM error: {e}")
            assistant_msg = ""

        # Parse action from LLM response
        action_dict = extract_json(assistant_msg)
        if action_dict is None:
            if verbose:
                print("  Failed to parse LLM output, using fallback")
            action_dict = fallback_action(obs.step_count)

        action_type = action_dict.get("action_type", "query_data")
        parameters = action_dict.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}

        try:
            action = BoardroomAction(action_type=action_type, parameters=parameters)
        except Exception:
            action_dict = fallback_action(obs.step_count)
            action = BoardroomAction(
                action_type=action_dict["action_type"],
                parameters=action_dict["parameters"],
            )

        # Step the environment via HTTP/WebSocket
        result = await env.step(action)
        obs = result.observation

        if verbose:
            print(f"  Step {obs.step_count}: {action.action_type} -> reward={obs.reward}")

        # Update conversation history
        history.append({"role": "assistant", "content": json.dumps(action_dict)})
        history.append({
            "role": "user",
            "content": f"Observation:\n{obs_to_text(obs)}\n\nChoose your next action (JSON only):",
        })

        # Keep history manageable
        if len(history) > 24:
            history = [history[0]] + history[-23:]

    return obs.metadata.get("final_score", obs.reward or 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Cannot call LLM API.")
        print("Set HF_TOKEN, API_BASE_URL, and MODEL_NAME environment variables.")
        return

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print("OpenBoardroom LLM Inference Agent")
    print(f"Model: {MODEL_NAME}")
    print(f"API: {API_BASE_URL}")
    print(f"Environment: {ENV_SPACE_URL}")
    print(f"Episodes per tier: {EPISODES_PER_TIER}")
    print("=" * 50)

    async with BoardroomEnv(base_url=ENV_SPACE_URL) as env:
        for tier in DIFFICULTIES:
            scores: List[float] = []
            for seed in SEEDS:
                print(f"\n[{tier}] Episode seed={seed}...", end=" ", flush=True)
                score = await run_episode(
                    env, llm_client, seed=seed, difficulty=tier, verbose=False,
                )
                scores.append(score)
                print(f"score={score:.4f}")

            arr = np.array(scores)
            print(f"\n{tier:>8s}: mean={arr.mean():.4f} ± std={arr.std():.4f}")

    print("\n" + "=" * 50)
    print("Inference complete.")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
