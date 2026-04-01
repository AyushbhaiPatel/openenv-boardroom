---
title: OpenBoardroom Environment
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# OpenBoardroom — SaaS CDO Decision-Making Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent plays the role of Chief Data Officer (CDO) at a simulated SaaS company. The agent must gather data, consult biased stakeholders, run counterfactual simulations, and make strategic decisions across quarterly episodes — all while navigating noisy data and organizational politics.

## Quick Start

```python
from my_env import BoardroomAction, BoardroomEnv

# Connect to a running server
with BoardroomEnv(base_url="http://localhost:8000") as client:
    result = client.reset()
    print(f"Objective: {result.observation.metadata['objective']}")

    # Query a metric
    action = BoardroomAction(action_type="query_data", parameters={"metric": "revenue"})
    result = client.step(action)
    print(f"Revenue: {result.observation.data_tables}")
```

## Environment Description

The agent operates as CDO of a procedurally generated SaaS company. Each episode presents a scenario with a specific objective. The agent must investigate company metrics, consult stakeholders with hidden biases, run what-if simulations, and ultimately make a strategic decision with a written explanation.

Episodes are fully deterministic given a seed, and the environment produces dense rewards at every step.

## Action Space

`BoardroomAction` has two fields:
- `action_type`: one of the 5 action types below
- `parameters`: dict with action-specific keys

| Action Type | Parameters | Description |
|---|---|---|
| `query_data` | `{"metric": str}` | Query a company metric (revenue, churn_rate, monthly_active_users, ad_spend, cac, ltv) |
| `analyze_trend` | `{"metric": str, "quarters": int}` | Analyze historical trend for a metric |
| `consult_stakeholder` | `{"stakeholder": str}` | Consult analyst, ceo, or risk_officer |
| `simulate_counterfactual` | `{"decision": str, "parameters": dict}` | Run a what-if simulation |
| `make_decision` | `{"decision": str, "parameters": dict, "explanation": str}` | Make a strategic decision with reasoning |

## Observation Space

`BoardroomObservation` fields:
- `data_tables` (dict) — Queried metrics and data
- `stakeholder_feedback` (str | None) — Feedback from consulted stakeholder
- `simulation_results` (dict | None) — Counterfactual simulation output
- `quarter` (int) — Current quarter number
- `step_count` (int) — Steps taken this episode
- `done` (bool) — Whether the episode has ended
- `reward` (float) — Step reward or final normalized score (0.0–1.0)
- `metadata` (dict) — Seed, difficulty, objective, errors, audit trail (on done)

## Difficulty Tiers (Tasks)

| Tier | Objective | Max Steps | Noise | Score Range |
|---|---|---|---|---|
| **Easy** | Find the growth bottleneck | 10 | None | 0.0–1.0 |
| **Medium** | Diagnose the revenue drop | 20 | ±5–10% Gaussian, occasional NaN | 0.0–1.0 |
| **Hard** | Should we launch Feature X? | 30 | ±15–20%, NaN, misleading signals | 0.0–1.0 |

## Reward Function

Dense rewards at every step:
- `query_data`: +0.2 (relevant metric)
- `analyze_trend`: +0.25 (noise handled)
- `simulate_counterfactual`: +0.3 (insightful simulation)
- `consult_stakeholder`: +0.15 (effective navigation)
- `make_decision`: -0.1 to +0.5 (quality-dependent)

Final episode score is normalized to [0.0, 1.0] combining step rewards, decision quality, explanation score, and stakeholder navigation.

## Setup Instructions

```bash
# Clone and enter the environment directory
cd my_env

# Install dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run the server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Running the Inference Agent

```bash
# Set required environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b:novita"
export HF_TOKEN="your_token_here"

# Run inference
python inference.py
```

## Running the Baseline Agent

```bash
python -m my_env.baseline_agent
```

## Building & Deploying

```bash
# Build Docker image
docker build -t boardroom-env:latest -f server/Dockerfile .

# Deploy to HuggingFace Spaces
openenv push --repo-id your-username/boardroom-env
```

## Project Structure

```
my_env/
├── __init__.py              # Exports BoardroomEnv, BoardroomAction, BoardroomObservation
├── models.py                # Pydantic Action/Observation models + dataclasses
├── client.py                # BoardroomEnv(EnvClient) client
├── baseline_agent.py        # Rule-based baseline agent
├── inference.py             # LLM-based inference agent (OpenAI client)
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies and metadata
├── server/
│   ├── app.py               # FastAPI application
│   ├── boardroom_environment.py  # Main environment orchestrator
│   ├── data_generator.py    # Synthetic SaaS data generation
│   ├── stakeholder_simulator.py  # 3 biased stakeholder personas
│   ├── counterfactual_engine.py  # PyTorch MLP for what-if reasoning
│   ├── reward_calculator.py # Dense multi-component rewards
│   ├── explanation_grader.py # Heuristic explanation scoring
│   ├── noise_injector.py    # Difficulty-scaled data corruption
│   ├── audit_trail.py       # Episode event logging
│   ├── Dockerfile           # Container definition
│   └── requirements.txt     # Server dependencies
└── tests/                   # Unit and property tests
```
