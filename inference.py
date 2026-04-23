"""Validator-safe inference runner for the OpenBoardroom environment."""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from my_env.client import BoardroomEnv
from my_env.models import BoardroomAction
from my_env.policy import ScenarioAwarePolicy

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = (os.getenv("ENV_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
HF_ENV_REPO_ID = os.getenv("HF_ENV_REPO_ID", "")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "boardroom")
MAX_STEPS = 30
TEMPERATURE = 0.2
MAX_TOKENS = 350
TASK_CONFIGS: List[Tuple[str, int]] = [
    ("easy", 7),
    ("medium", 17),
    ("hard", 29),
    ("multi_agent_easy", 7),
    ("multi_agent_medium", 17),
    ("multi_agent_hard", 29),
]
MIN_SCORE = 0.01
MAX_SCORE = 0.99

SYSTEM_PROMPT = """You are solving a business operations simulation.
Return exactly one JSON object with this shape:
{"action_type":"query_data|analyze_trend|consult_stakeholder|simulate_counterfactual|make_decision|present_evidence|negotiate","parameters":{...}}
Do not use markdown.
Prioritize evidence gathering, then trends, stakeholders, simulation, and finally a grounded decision.
In multi-agent tasks, use present_evidence to influence board actors and negotiate to counter CEO lobbying before making your final decision."""


def _sanitize(value: Any) -> str:
    text = "null" if value is None else str(value)
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    return text if text else "null"


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={_sanitize(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={_sanitize(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def fallback_action(step: int, task_name: str, policy: ScenarioAwarePolicy) -> Dict[str, Any]:
    return policy.next_action(step)


def build_prompt(task_name: str, step: int, obs: Any) -> str:
    metadata = dict(obs.metadata or {})
    for key in (
        "objective",
        "max_steps",
        "difficulty",
        "brief",
        "actor_messages",
        "board_vote",
        "vote_result",
    ):
        value = getattr(obs, key, None)
        if value not in (None, {}, []):
            metadata.setdefault(key, value)
    return json.dumps(
        {
            "task": task_name,
            "step": step,
            "objective": metadata.get("objective"),
            "max_steps": metadata.get("max_steps"),
            "observation": {
                "data_tables": obs.data_tables,
                "stakeholder_feedback": obs.stakeholder_feedback,
                "simulation_results": obs.simulation_results,
                "quarter": obs.quarter,
                "step_count": obs.step_count,
                "done": obs.done,
                "reward": obs.reward,
                "metadata": metadata,
            },
        }
    )


def choose_action(
    client: Optional[OpenAI],
    task_name: str,
    step: int,
    obs: Any,
    policy: ScenarioAwarePolicy,
) -> Dict[str, Any]:
    if client is None:
        return fallback_action(step, task_name, policy)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(task_name, step, obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = completion.choices[0].message.content or ""
        parsed = extract_json(content)
        return parsed if isinstance(parsed, dict) else fallback_action(step, task_name, policy)
    except Exception:
        return fallback_action(step, task_name, policy)


def normalize_score(score: float) -> float:
    return min(MAX_SCORE, max(MIN_SCORE, float(score)))


async def create_env(is_multi_agent: bool = False) -> BoardroomEnv:
    errors: List[str] = []

    env_base_url = ENV_BASE_URL
    if is_multi_agent and not env_base_url.endswith("/multi"):
        env_base_url = f"{env_base_url}/multi"

    if LOCAL_IMAGE_NAME:
        try:
            env = await BoardroomEnv.from_docker_image(LOCAL_IMAGE_NAME)
            if is_multi_agent:
                # Multi-agent environment is mounted under /multi on the same container
                env.base_url = f"{env.base_url.rstrip('/')}/multi"
            return env
        except Exception as exc:
            errors.append(f"LOCAL_IMAGE_NAME={LOCAL_IMAGE_NAME}: {_sanitize(exc)}")

    if env_base_url:
        try:
            env = BoardroomEnv(base_url=env_base_url)
            await env.connect()
            return env
        except Exception as exc:
            errors.append(f"ENV_BASE_URL={env_base_url}: {_sanitize(exc)}")

    if HF_ENV_REPO_ID:
        try:
            return await BoardroomEnv.from_env(HF_ENV_REPO_ID)
        except Exception as exc:
            errors.append(f"HF_ENV_REPO_ID={HF_ENV_REPO_ID}: {_sanitize(exc)}")

    if errors:
        details = " | ".join(errors)
        raise RuntimeError(
            "Unable to connect to environment. Set a valid "
            "LOCAL_IMAGE_NAME, ENV_BASE_URL, or HF_ENV_REPO_ID. "
            f"Attempts: {details}"
        )

    raise RuntimeError(
        "Unable to connect to environment. Provide one of LOCAL_IMAGE_NAME, "
        "ENV_BASE_URL, or HF_ENV_REPO_ID before running inference."
    )


async def run_task(task_name: str, seed: int, client: Optional[OpenAI]) -> None:
    rewards: List[float] = []
    score = 0.0
    steps_taken = 0
    success = False
    env: Optional[BoardroomEnv] = None

    # Map multi-agent task names to difficulty
    difficulty_map = {
        "multi_agent_easy": "easy",
        "multi_agent_medium": "medium",
        "multi_agent_hard": "hard",
    }
    difficulty = difficulty_map.get(task_name, task_name)
    is_multi_agent = task_name.startswith("multi_agent_")

    log_start(task_name)

    try:
        env = await create_env(is_multi_agent=is_multi_agent)
        result = await env.reset(seed=seed, difficulty=difficulty)
        obs = result.observation
        policy = ScenarioAwarePolicy(difficulty=difficulty, snapshot=obs.data_tables, multi_agent=is_multi_agent)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_dict = choose_action(client, task_name, step, obs, policy)
            try:
                action = BoardroomAction(
                    action_type=action_dict.get("action_type", "query_data"),
                    parameters=action_dict.get("parameters", {}) or {},
                )
            except Exception:
                action_dict = fallback_action(step, task_name, policy)
                action = BoardroomAction(
                    action_type=action_dict["action_type"],
                    parameters=action_dict["parameters"],
                )

            action_str = json.dumps(
                {"action_type": action.action_type, "parameters": action.parameters},
                separators=(",", ":"),
                sort_keys=True,
            )

            error_text = None
            try:
                result = await env.step(action)
                obs = result.observation
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                error_text = (obs.metadata or {}).get("error")
            except Exception as exc:
                reward = 0.0
                done = True
                error_text = _sanitize(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, error_text)

            if done:
                score = float(getattr(obs, "final_score", None) or (obs.metadata or {}).get("final_score", reward))
                break

        score = normalize_score(score)
        success = score > MIN_SCORE
    except Exception as exc:
        score = MIN_SCORE
        if steps_taken == 0:
            log_step(0, "init", 0.0, True, _sanitize(exc))
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as exc:
                print(f"[WARN] env.close() failed: {_sanitize(exc)}")
        log_end(success, steps_taken, score, rewards)


async def async_main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    for task_name, seed in TASK_CONFIGS:
        await run_task(task_name, seed, client)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
