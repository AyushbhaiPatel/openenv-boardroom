---
title: OpenBoardroom Environment
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# OpenBoardroom

An OpenEnv-compatible RL environment where an AI agent plays Chief Data Officer (CDO) at a simulated SaaS company. Navigate noisy data, biased stakeholders, and counterfactual simulations to make strategic business decisions.

## Live Demo

[HuggingFace Space](https://huggingface.co/spaces/ayushbhaiPatel/boardroom-env)

## Features

- 5 action types: query data, analyze trends, consult stakeholders, run simulations, make decisions
- 3 difficulty tiers with distinct scenarios and grading (easy/medium/hard)
- Dense reward shaping with explanation quality scoring
- Seed-deterministic episodes for reproducibility
- PyTorch counterfactual engine, stakeholder bias simulation, noise injection
- Full OpenEnv spec compliance (step/reset/state APIs)
- Deployed as Docker container to HuggingFace Spaces

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest my_env/tests -v
python -m my_env.baseline_agent   # benchmark scores
```

## Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b:novita"
export HF_TOKEN="your_token"
# Preferred for validator/local runs:
# export LOCAL_IMAGE_NAME="your-local-docker-image"
# Or point at a running server:
# export ENV_BASE_URL="http://127.0.0.1:8000"
# Fallback only if you intentionally want OpenEnv to launch from a repo id:
# export HF_ENV_REPO_ID="your-username/boardroom-env"
python inference.py
```

The root `inference.py` emits only the required structured lines:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<json_action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Required env vars for submission:
- `API_BASE_URL`: LLM API endpoint
- `MODEL_NAME`: model identifier used by the OpenAI client
- `HF_TOKEN`: credential for LLM access
- `LOCAL_IMAGE_NAME`: local Docker image name if the validator launches the env from Docker

## Action & Observation Spaces

### Actions

Each step the agent submits one JSON action:

| Action | Parameters | Description |
|---|---|---|
| `query_data` | `{"metric": "<name>"}` | Query a KPI (revenue, monthly_active_users, churn_rate, ad_spend, cac, ltv, support_load, release_risk) |
| `analyze_trend` | `{"metric": "<name>", "quarters": int}` | Get trend data for a metric over N quarters |
| `consult_stakeholder` | `{"stakeholder": "<name>"}` | Get feedback from analyst, ceo, or risk_officer |
| `simulate_counterfactual` | `{"decision": "<desc>", "parameters": {}}` | Run a PyTorch what-if simulation |
| `make_decision` | `{"decision": "<desc>", "parameters": {}, "explanation": "<text>"}` | Submit final decision and end the episode |

For the hard launch-governance task, `make_decision.parameters` supports structured rollout planning fields:

```json
{
  "decision": "delay feature x launch",
  "parameters": {
    "rollout_percentage": 10,
    "support_headcount_delta": 4,
    "rollback_plan": "Gate the release behind a feature flag and rollback within one hour if churn spikes."
  },
  "explanation": "Support capacity and release risk are elevated, so broad launch should wait."
}
```

The hard-task grader gives more credit when the agent commits to an operational plan instead of generic launch prose.

### Observations

Each step returns:

| Field | Type | Description |
|---|---|---|
| `data_tables` | Dict[str, Any] | Queried metric values or trend data |
| `stakeholder_feedback` | str \| None | Bias-coloured feedback from consulted stakeholder |
| `simulation_results` | Dict \| None | Counterfactual engine output (revenue/churn/user deltas) |
| `quarter` | int | Current simulation quarter |
| `step_count` | int | Steps taken this episode |
| `done` | bool | Episode complete (max steps or decision made) |
| `reward` | float | Step reward in [0.0, 1.0] |
| `metadata` | Dict | objective, max_steps, difficulty, final_score, oracle_answer, oracle_hit |

## Tasks And Graders

The benchmark exposes 3 deterministic tasks, each with a programmatic grader that returns a score in `[0.0, 1.0]`:

| Task | Difficulty | Grader Signals |
|---|---|---|
| Find the Growth Bottleneck | Easy | Relevant KPI queries, trend inspection, decision quality, oracle match |
| Diagnose the Revenue Drop | Medium | Noise-aware trend analysis, stakeholder diversity, explanation quality, oracle match |
| Should We Launch Feature X? | Hard | Counterfactual quality, stakeholder navigation, launch reasoning, structured rollout plan, oracle match |

Grading is deterministic for a fixed seed. Final scoring combines dense step rewards from [reward_calculator.py](/Users/ayush/Desktop/openenv-project/my_env/server/reward_calculator.py) and explanation quality from [explanation_grader.py](/Users/ayush/Desktop/openenv-project/my_env/server/explanation_grader.py). On episode completion, the environment emits `final_score`, `oracle_answer`, and `oracle_hit` in observation metadata for auditability.

Illustrative hard-task grading behavior:
- Generic response: `"launch feature x"` with no rollout plan and no rollback path scores poorly.
- Better response: `"delay feature x launch"` plus `rollout_percentage`, `support_headcount_delta`, and `rollback_plan` scores higher when it cites support load, release risk, and churn tradeoffs.

## Baseline Reproducibility

The deterministic baseline uses fixed seeds `0..99` for each difficulty tier and prints reproducible `mean ± std` scores:

```bash
python -m my_env.baseline_agent
```

Latest local run from the project venv produced:

| Difficulty | Mean Score | Std Dev |
|---|---:|---:|
| Easy | 0.9581 | 0.0287 |
| Medium | 0.7988 | 0.0539 |
| Hard | 0.8533 | 0.0122 |

## Project Structure

```
├── inference.py              # LLM inference agent (root, as required)
├── openenv.yaml              # OpenEnv manifest (root, as required)
├── Dockerfile                # Container build
└── my_env/                   # OpenEnv environment package
    ├── server/               # Environment server + subsystems
    ├── tests/                # 41 unit tests
    ├── baseline_agent.py     # Rule-based baseline
    ├── client.py             # BoardroomEnv HTTP client
    └── models.py             # Pydantic Action/Observation models
```

## Why This Environment Matters

Data-driven decision-making inside organisations is notoriously hard to train AI agents on: real boardroom data is confidential, feedback loops are slow, and stakeholder biases are invisible. OpenBoardroom solves this by fully simulating a SaaS company's quarterly review process — noisy KPIs, politically biased advisors, counterfactual "what-if" engines, and a graded final decision — so RL agents can learn persuasive, evidence-backed reasoning at scale. Solving this environment demonstrates an agent can (1) identify causal signals inside noisy tabular data, (2) resist misleading stakeholder framing, and (3) construct defensible strategic decisions. These skills transfer directly to real analyst-support tooling, automated due-diligence systems, and corporate BI co-pilots.

## Evaluation

| What to run | Command |
|---|---|
| Full test suite | `pip install -e ".[dev]" && python -m pytest my_env/tests -v` |
| Baseline agent benchmark | `python -m my_env.baseline_agent` |
| LLM inference end-to-end | `python inference.py` (see Inference section above) |
| OpenEnv validator | `./validate-submission.sh <SPACE_URL>` |
| Docker smoke build | `docker build -t boardroom-env .` |

Each episode terminates with `oracle_answer` and `oracle_hit` in the final observation's `metadata`, allowing automatic grading of whether the agent identified the true bottleneck or reached the correct launch verdict.

Built by [Ayush](https://github.com/AyushbhaiPatel) and [Dharmit](https://github.com/dharmitpatel81).
