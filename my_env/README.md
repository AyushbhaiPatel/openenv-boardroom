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

# OpenBoardroom — Package Guide

`my_env` contains the reusable environment package for OpenBoardroom: the client, data models, deterministic baseline, shared policy layer, server subsystems, and tests.

## What This Package Includes

- `client.py`: `BoardroomEnv` HTTP/WebSocket client for interacting with a running environment
- `models.py`: action and observation models plus internal scenario state types
- `baseline_agent.py`: deterministic scenario-aware benchmark agent
- `policy.py`: shared archetype-aware policy used by the baseline and inference fallback
- `server/`: FastAPI app and all simulation/grading subsystems
- `tests/`: package-level unit tests

## Quick Start

```bash
pip install -e ".[dev]"
./venv/bin/python -m pytest my_env/tests -q
./venv/bin/python -m my_env.baseline_agent
```

## Running The Server Locally

From the repository root:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then in another terminal:

```bash
export ENV_BASE_URL="http://127.0.0.1:8000"
./venv/bin/python inference.py
```

## Client Example

```python
from my_env import BoardroomAction, BoardroomEnv

with BoardroomEnv(base_url="http://127.0.0.1:8000") as client:
    result = client.reset(seed=7, difficulty="medium")
    print(result.observation.metadata["objective"])

    action = BoardroomAction(
        action_type="query_data",
        parameters={"metric": "revenue"},
    )
    result = client.step(action)
    print(result.observation.data_tables)
```

## Action Space

| Action Type | Parameters | Description |
|---|---|---|
| `query_data` | `{"metric": str}` | Query a company KPI |
| `analyze_trend` | `{"metric": str, "quarters": int}` | Inspect multi-quarter trend data |
| `consult_stakeholder` | `{"stakeholder": str}` | Consult `analyst`, `ceo`, or `risk_officer` |
| `simulate_counterfactual` | `{"decision": str, "parameters": dict}` | Run a what-if simulation |
| `make_decision` | `{"decision": str, "parameters": dict, "explanation": str}` | End the episode with a final recommendation |

## Observation Space

`BoardroomObservation` exposes:

- `data_tables`
- `stakeholder_feedback`
- `simulation_results`
- `quarter`
- `step_count`
- `done`
- `reward`
- `metadata`

Final-step metadata includes `final_score`, `oracle_answer`, `oracle_hit`, and `audit_trail`.

## Difficulty Tiers

| Tier | Objective | Max Steps | Special Challenge |
|---|---|---:|---|
| Easy | Find the Growth Bottleneck | 10 | Cleaner signals, faster diagnosis |
| Medium | Diagnose the Revenue Drop | 20 | Noisy data and missing values |
| Hard | Should We Launch Feature X? | 30 | Conflicting signals, stakeholder pressure, launch-risk tradeoffs |

## Notes For Reviewers

- The package is deterministic for fixed seeds.
- The baseline and fallback policy are intentionally reproducible for benchmarking.
- The authoritative project overview lives in the root [README.md](/Users/ayush/Desktop/openenv-project/README.md).
