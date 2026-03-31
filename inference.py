# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference Script for OpenBoardroom Environment
===================================

MANDATORY
- Before submitting, ensure the following variables are defined in your
  environment configuration:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.
- The inference script must be named `inference.py` and placed in the root
  directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, Optional

import numpy as np
from openai import OpenAI

from my_env.client import BoardroomEnv
from my_env.models import BoardroomAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:novita")
# HuggingFace Space repo id for remote env, or leave unset to use localhost.
HF_ENV_REPO_ID = os.getenv("HF_ENV_REPO_ID", "ayushbhaiPatel/boardroom-env")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "").strip()

DIFFICULTIES = ["easy", "medium", "hard"]
EPISODES_PER_TIER = 2  # Keep low for 20-min runtime on 2 vCPU / 8 GB
MAX_STEPS = 30
TEMPERATURE = 0.3
MAX_TOKENS = 512

VALID_METRICS = ["revenue", "monthly_active_users", "churn_rate",
                 "ad_spend", "cac", "ltv"]
VALID_STAKEHOLDERS = ["analyst", "ceo", "risk_officer"]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Chief Data Officer (CDO) at a SaaS company. You are playing \
a strategic decision-making game. Each turn you must choose ONE action.

Available actions (respond with valid JSON only):

1. Query a metric:
   {"action_type": "query_data", "parameters": {"metric": "<name>"}}
   Valid metrics: revenue, monthly_active_users, churn_rate, ad_spend, cac, ltv

2. Analyze a trend:
   {"action_type": "analyze_trend", "parameters": {"metric": "<name>", "quarters": 2}}

3. Consult a stakeholder:
   {"action_type": "consult_stakeholder", "parameters": {"stakeholder": "<name>"}}
   Valid: analyst, ceo, risk_officer

4. Run a counterfactual simulation:
   {"action_type": "simulate_counterfactual", "parameters": {"decision": "<desc>", "parameters": {}}}

5. Make a final decision (ends the quarter):
   {"action_type": "make_decision", "parameters": {"decision": "<desc>", "parameters": {}, "explanation": "<reasoning>"}}

Strategy: Query key metrics first, consult stakeholders, run simulations, \
then make a well-reasoned decision. Reference data, acknowledge uncertainty, \
and consider stakeholder perspectives in your explanation.

IMPORTANT: Respond with ONLY a single JSON object. No markdown, no commentary.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from LLM output (handles nested objects)."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                obj, _ = json.JSONDecoder().raw_decode(text)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        blob = match.group(1).strip()
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            try:
                obj, _ = json.JSONDecoder().raw_decode(blob)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    start = text.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def obs_to_text(obs, difficulty: str, max_steps: int) -> str:
    """Convert observation to text for the LLM (includes objective every turn)."""
    md = obs.metadata or {}
    objective = md.get("objective", "")
    remaining = max(0, max_steps - obs.step_count)
    parts = [
        f"Difficulty: {difficulty} | Objective: {objective}",
        f"Step {obs.step_count} / max {max_steps} (about {remaining} steps left before timeout)",
        f"Quarter {obs.quarter}",
    ]
    if obs.data_tables:
        parts.append(f"Data: {json.dumps(obs.data_tables, default=str)}")
    if obs.stakeholder_feedback:
        parts.append(f"Stakeholder feedback: {obs.stakeholder_feedback}")
    if obs.simulation_results:
        parts.append(f"Simulation: {json.dumps(obs.simulation_results, default=str)}")
    if md.get("error"):
        parts.append(f"Error: {md['error']}")
    if md.get("final_score") is not None:
        parts.append(f"Final score (episode over): {md['final_score']:.4f}")
    if md.get("step_reward") is not None and obs.done:
        parts.append(f"Last step reward (before final score): {md['step_reward']:+.4f}")
    parts.append(f"Reward this response: {obs.reward}")
    return "\n".join(parts)


def fallback_action(step: int) -> Dict[str, Any]:
    """Deterministic fallback if LLM fails."""
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
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(env, client, difficulty: str, seed: int) -> float:
    """Run one episode. Returns final score."""
    result = await env.reset(seed=seed, difficulty=difficulty)
    obs = result.observation
    max_steps = int((obs.metadata or {}).get("max_steps", 30))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Difficulty: {difficulty}\n"
            f"Observation:\n{obs_to_text(obs, difficulty, max_steps)}\n\n"
            "Choose your first action (JSON only):"
        )},
    ]

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  Model request failed ({exc}). Using fallback.")
            response_text = ""

        # Parse action
        action_dict = extract_json(response_text)
        if action_dict is None:
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

        # Step environment
        result = await env.step(action)
        obs = result.observation
        reward = result.reward or 0.0

        error_flag = " ERROR" if (obs.metadata or {}).get("error") else ""
        print(f"[STEP] Step {step}: {action.action_type} -> "
              f"reward={reward:+.3f} | done={obs.done}{error_flag}")

        # Update messages
        messages.append({"role": "assistant", "content": json.dumps(action_dict)})
        max_steps = int((obs.metadata or {}).get("max_steps", max_steps))
        messages.append({"role": "user", "content": (
            f"Observation:\n{obs_to_text(obs, difficulty, max_steps)}\n\n"
            "Choose next action (JSON only):"
        )})

        # Keep history manageable
        if len(messages) > 20:
            messages = [messages[0]] + messages[-19:]

    return obs.metadata.get("final_score", obs.reward or 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main() -> None:
    if not API_KEY:
        raise SystemExit("HF_TOKEN (or API_KEY) must be set to query the model.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if ENV_BASE_URL:
        env = BoardroomEnv(base_url=ENV_BASE_URL.rstrip("/"))
        await env.connect()
    else:
        env = await BoardroomEnv.from_env(HF_ENV_REPO_ID)

    print("[START] OpenBoardroom Inference")
    try:
        for difficulty in DIFFICULTIES:
            scores = []
            for seed in range(EPISODES_PER_TIER):
                print(f"\n[START] Episode: difficulty={difficulty} seed={seed}")
                # Reconnect before each episode to avoid keepalive timeout
                # on slow free-tier LLMs (HF Space closes WS after ~30s idle).
                try:
                    await env.close()
                except Exception:
                    pass
                if ENV_BASE_URL:
                    env = BoardroomEnv(base_url=ENV_BASE_URL.rstrip("/"))
                    await env.connect()
                else:
                    env = await BoardroomEnv.from_env(HF_ENV_REPO_ID)
                score = await run_episode(env, client, difficulty, seed)
                scores.append(score)
                print(f"[END] Episode: score={score:.4f}")

            arr = np.array(scores)
            print(f"\n  {difficulty}: mean={arr.mean():.4f} std={arr.std():.4f}")
    finally:
        try:
            await env.close()
        except Exception:
            pass

    print("\n[END] OpenBoardroom Inference complete.")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
