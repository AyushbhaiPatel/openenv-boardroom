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
cd my_env
pip install -e ".[dev]"
python -m pytest tests/ -v        # 41 tests
python -m my_env.baseline_agent   # benchmark scores
```

## Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b:novita"
export HF_TOKEN="your_token"
# Optional: point at a local server instead of Hugging Face Spaces
# export ENV_BASE_URL="http://localhost:8000"
# Optional: override the default Space repo id
# export HF_ENV_REPO_ID="your-username/boardroom-env"
python inference.py
```

## Action & Observation Spaces

### Actions

Each step the agent submits one JSON action:

| Action | Parameters | Description |
|---|---|---|
| `query_data` | `{"metric": "<name>"}` | Query a KPI (revenue, monthly_active_users, churn_rate, ad_spend, cac, ltv) |
| `analyze_trend` | `{"metric": "<name>", "quarters": int}` | Get trend data for a metric over N quarters |
| `consult_stakeholder` | `{"stakeholder": "<name>"}` | Get feedback from analyst, ceo, or risk_officer |
| `simulate_counterfactual` | `{"decision": "<desc>", "parameters": {}}` | Run a PyTorch what-if simulation |
| `make_decision` | `{"decision": "<desc>", "parameters": {}, "explanation": "<text>"}` | Submit final decision and end the episode |

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
| Full test suite (41 tests) | `cd my_env && pip install -e ".[dev]" && python -m pytest tests/ -v` |
| Baseline agent benchmark | `python -m my_env.baseline_agent` |
| LLM inference end-to-end | `python inference.py` (see Inference section above) |
| OpenEnv validator | `./validate-submission.sh <SPACE_URL>` |

Each episode terminates with `oracle_answer` and `oracle_hit` in the final observation's `metadata`, allowing automatic grading of whether the agent identified the true bottleneck or reached the correct launch verdict.

Built by [Ayush](https://github.com/AyushbhaiPatel) and [Dharmit](https://github.com/dharmitpatel81).
