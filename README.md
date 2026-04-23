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

OpenBoardroom is designed to test something harder than tool use: whether an agent can make defensible business decisions under uncertainty. Instead of optimizing a toy objective, the agent must separate real signals from noisy dashboards, resist stakeholder pressure, run what-if simulations, and produce a final recommendation that is both operationally grounded and auditable.

The project now ships two environments: the original **single-agent** boardroom and a new **multi-agent** extension where three independent rule-based actors (CEO, CFO, Risk Officer) act every step, requiring the CDO agent to present evidence, counter lobbying, and secure a 2/3 board majority vote.

The project is containerized for Hugging Face Spaces deployment and OpenEnv validation, so the same environment can be tested locally, benchmarked reproducibly, and evaluated through the hackathon tooling.

## Live Demo

[HuggingFace Space](https://huggingface.co/spaces/ayushbhaiPatel/boardroom-env)

## Why It Stands Out

- Trains decision-making, not just retrieval: the agent must gather evidence, synthesize conflicting advice, and choose a course of action.
- Auditable by design: each episode ends with `final_score`, `oracle_answer`, `oracle_hit`, and a full audit trail.
- Real evaluator pressure: medium and hard tasks include noise, ambiguity, and misleading stakeholder framing.
- **Multi-agent boardroom**: three independent rule-based actors (CEO, CFO, Risk Officer) act every step — the CDO must present evidence, counter CEO lobbying, and win a 2/3 board majority vote.
- **Dense reward signal**: every action earns a reward, including `present_evidence` (+0.05–0.15), `negotiate` (+0.05), CFO stance flip (+0.20), hidden metric reveal (+0.30), and board vote outcome (±0.20).
- Strong benchmark story: deterministic baselines are reproducible across both single-agent and multi-agent tasks.
- Robust inference path: if the LLM output is malformed or unavailable, the fallback policy still executes a scenario-aware strategy.
- **GRPO training ready**: `train_grpo.py` provides a Colab-ready training script with curriculum scheduling (easy → medium → hard).

## Features

- 7 action types: query data, analyze trends, consult stakeholders, run simulations, make decisions, **present evidence**, **negotiate**
- 3 difficulty tiers with distinct scenarios and grading (easy/medium/hard)
- **Multi-agent boardroom**: CEO (hidden agenda + lobbying), CFO (budget tracking + stance flipping), Risk Officer (threshold alerts + intel unlocking)
- **Board majority vote**: 2/3 approval required to resolve an episode; one revision chance on rejection
- Dense reward shaping with explanation quality scoring
- Seed-deterministic episodes for reproducibility
- PyTorch counterfactual engine, stakeholder bias simulation, noise injection
- Shared scenario-aware playbook used by both the benchmark baseline and inference fallback
- Full OpenEnv spec compliance (step/reset/state APIs)
- Deployed as Docker container to HuggingFace Spaces
- **GRPO training script** (`train_grpo.py`) with Unsloth + HF TRL and curriculum scheduler

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
# export HF_ENV_REPO_ID="your-repo/boardroom-env"
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
- `OPENAI_API_KEY`: optional fallback alias for the same LLM credential
- `LOCAL_IMAGE_NAME`: *(optional)* local Docker image name if the validator launches the env from Docker

## Action & Observation Spaces

### Actions

Each step the agent submits one JSON action:

| Action | Parameters | Description |
|---|---|---|
| `query_data` | `{"metric": "<name>"}` | Query a KPI (revenue, monthly_active_users, churn_rate, ad_spend, cac, ltv, support_load, release_risk) |
| `analyze_trend` | `{"metric": "<name>", "quarters": int}` | Get trend data for a metric over N quarters |
| `consult_stakeholder` | `{"stakeholder": "<name>"}` | Get feedback from analyst, ceo, or risk_officer |
| `simulate_counterfactual` | `{"decision": "<desc>", "parameters": {}}` | Run a PyTorch what-if simulation |
| `make_decision` | `{"decision": "<desc>", "parameters": {}, "explanation": "<text>"}` | Submit final decision and trigger board vote |
| `present_evidence` | `{"target": "<actor>", "metric": "<name>", "value": any, "interpretation": "<text>"}` | Present data evidence to a board actor (ceo/cfo/risk_officer) |
| `negotiate` | `{"target": "<actor>", "position": "<text>"}` | Negotiate with a board actor to counter lobbying |

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
| `data_tables` | Dict[str, Any] | Queried metric values, trend data, or risk intel payload |
| `stakeholder_feedback` | str \| None | Bias-coloured feedback from consulted stakeholder or actor response |
| `simulation_results` | Dict \| None | Counterfactual engine output (revenue/churn/user deltas) |
| `quarter` | int | Current simulation quarter |
| `step_count` | int | Steps taken this episode |
| `done` | bool | Episode complete (max steps, board majority, or second make_decision) |
| `reward` | float | Step reward in [0.0, 1.0] |
| `metadata` | Dict | objective, max_steps, difficulty, final_score, oracle_answer, oracle_hit, **actor_messages**, **board_vote**, **vote_result** |

The `actor_messages` key is present on every step in `MultiAgentBoardroomEnvironment`:

```json
{
  "actor_messages": {
    "ceo": "The market window is closing. We need to move fast.",
    "cfo": "Evidence-based decisions are what I need to approve this.",
    "risk_officer": "⚠️ ALERT: support_load threshold breached."
  }
}
```

On `make_decision` steps, `board_vote` and `vote_result` are also included:

```json
{
  "board_vote": {"ceo": "reject", "cfo": "approve", "risk_officer": "approve"},
  "vote_result": "approved"
}
```

## Tasks And Graders

The benchmark exposes 6 deterministic tasks (3 single-agent, 3 multi-agent), each with a programmatic grader that returns a score in `[0.0, 1.0]`:

| Task | Environment | Difficulty | Grader Signals |
|---|---|---|---|
| Find the Growth Bottleneck | boardroom | Easy | Relevant KPI queries, trend inspection, decision quality, oracle match |
| Diagnose the Revenue Drop | boardroom | Medium | Noise-aware trend analysis, stakeholder diversity, explanation quality, oracle match |
| Should We Launch Feature X? | boardroom | Hard | Counterfactual quality, stakeholder navigation, launch reasoning, structured rollout plan, oracle match |
| Multi-Agent: Find the Growth Bottleneck | multi_agent_boardroom | Easy | + board majority vote, actor evidence, lobby counter |
| Multi-Agent: Diagnose the Revenue Drop | multi_agent_boardroom | Medium | + CFO stance flip, CEO lobbying resistance |
| Multi-Agent: Should We Launch Feature X? | multi_agent_boardroom | Hard | + hidden metric reveal, Risk Officer alert handling, board vote |

### Multi-Agent Dense Reward Table

| Event | Reward |
|---|---|
| `present_evidence` (valid, relevant) | +0.05 to +0.15 |
| `negotiate` (successful lobby reduction) | +0.05 |
| Hidden metric revealed (CEO deception caught) | +0.30 |
| CFO stance flip to evidence_based (one-time) | +0.20 |
| Risk Officer alert addressed in final decision | +0.15 |
| Risk Officer alert ignored in final decision | -0.15 |
| Board majority vote (≥2 approvals) | +0.20 |
| Board rejection (<2 approvals) | -0.20 |
| `make_decision` called 3+ times | -0.10 |

The reward logic is implemented as composable dense reward components instead of a single opaque final score. OpenBoardroom keeps this custom calculator because the signal depends on continuous actor state, hidden-metric discovery, CFO belief shifts, risk-alert latching, and board-vote outcome; those multi-agent transitions are easier to audit directly in the environment than as a detached rubric table.

## Multi-Agent Boardroom

The `MultiAgentBoardroomEnvironment` extends the base environment with three independent rule-based actors:

**CEO** — has a hidden launch agenda (hard difficulty). Lobbies the CFO every step the CDO ignores him. Suppresses `support_load` and `release_risk` metrics on hard. Can be countered with `negotiate`.

**CFO** — tracks budget independently. Stance flips between `neutral`, `pro_launch`, and `evidence_based` based on the balance of CEO lobbying vs CDO evidence. Flipping to `evidence_based` awards +0.20 (once per episode).

**Risk Officer** — monitors `support_load` against a difficulty-scaled threshold. Emits proactive alerts every step when breached. Unlocks deeper intel when CDO presents evidence referencing `support_load` or `release_risk`.

### Usage

```python
from my_env import MultiAgentBoardroomEnvironment
from my_env.models import BoardroomAction

env = MultiAgentBoardroomEnvironment()
obs = env.reset(seed=42, difficulty="hard")

# Present evidence to CFO
obs = env.step(BoardroomAction(
    action_type="present_evidence",
    parameters={
        "target": "cfo",
        "metric": "support_load",
        "value": 0.92,
        "interpretation": "Support load is critically high — launch risk is elevated.",
    },
))

# Negotiate with CEO to counter launch lobbying
obs = env.step(BoardroomAction(
    action_type="negotiate",
    parameters={
        "target": "ceo",
        "position": "We should delay the launch until support capacity improves.",
    },
))

# Make final decision — triggers board vote
obs = env.step(BoardroomAction(
    action_type="make_decision",
    parameters={
        "decision": "delay feature x launch",
        "parameters": {"rollout_percentage": 10, "rollback_plan": "Feature flag rollback within 1 hour."},
        "explanation": "Support load and release risk are elevated. Delaying is the safer choice.",
    },
))

print(obs.metadata["board_vote"])    # {"ceo": "reject", "cfo": "approve", "risk_officer": "approve"}
print(obs.metadata["vote_result"])   # "approved"
```

### GRPO Training

```bash
# Install dependencies (Colab or local GPU)
pip install unsloth trl transformers datasets

# Run training with curriculum scheduling
python train_grpo.py
```

The script starts at `difficulty="easy"` and advances to `"medium"` then `"hard"` when the rolling mean score over the last 50 episodes exceeds 0.65.

Final scoring combines dense step rewards from [reward_calculator.py](my_env/server/reward_calculator.py), [multi_agent_reward_calculator.py](my_env/server/multi_agent_reward_calculator.py), and explanation quality from [explanation_grader.py](my_env/server/explanation_grader.py). On episode completion, the environment emits `final_score`, `oracle_answer`, and `oracle_hit` in observation metadata for auditability.

When a GPU GRPO run is complete, commit the generated artifacts:

| Artifact | Meaning |
|---|---|
| `grpo_output/rewards.csv` | per-episode final score, difficulty, and rolling mean from `train_grpo.py` |
| `grpo_output/reward_curve.png` | actual GRPO reward curve generated from that CSV |

The repository may also contain a one-episode smoke-test CSV from CPU-only validation. That proves the training interface runs, but the final hackathon submission should replace it with a real GPU training run.

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
| Easy | 0.9593 | 0.0338 |
| Medium | 0.9428 | 0.0000 |
| Hard | 0.9846 | 0.0003 |

Multi-agent baseline (same seeds, `multi_agent=True` policy):

| Difficulty | Mean Score | Std Dev |
|---|---:|---:|
| Easy | 0.9749 | 0.0210 |
| Medium | 0.9278 | 0.0000 |
| Hard | 0.9108 | 0.0067 |

This is a useful project signal for reviewers: the environment is not only creative, it is stable enough to benchmark, compare policies, and support reproducible progress.

## Reward Improvement Demo

Generate judge-friendly reward evidence without a GPU:

```bash
python demo_rewards.py
```

The script compares a naive multi-agent agent against the boardroom policy and writes:

| Artifact | Purpose |
|---|---|
| `demo_output/reward_improvement.png` | before/after chart for the pitch |
| `demo_output/improvement_summary.csv` | exact naive-vs-policy gains |
| `demo_output/reward_curve_multi_agent.png` | multi-agent reward curve by difficulty |
| `demo_output/reward_curve_comparison.png` | single-agent vs multi-agent baseline comparison |

This directly supports the judging criterion for observable reward improvement as a reproducible baseline comparison. It is separate from the GRPO training curve above; for final submission, show both the policy-vs-naive comparison and the real `grpo_output/reward_curve.png` from a GPU run.

## Project Structure

```
├── inference.py              # LLM inference agent (root, as required)
├── train_grpo.py             # GRPO training script (Unsloth + HF TRL, curriculum scheduling)
├── openenv.yaml              # OpenEnv manifest (6 tasks: 3 single-agent + 3 multi-agent)
├── Dockerfile                # Container build
└── my_env/                   # OpenEnv environment package
    ├── server/               # Environment server + subsystems
    │   ├── boardroom_environment.py          # Single-agent environment
    │   ├── multi_agent_boardroom_environment.py  # Multi-agent environment (NEW)
    │   ├── multi_agent_reward_calculator.py  # Multi-agent reward components (NEW)
    │   └── ...               # data_generator, reward_calculator, grader, etc.
    ├── tests/                # 164 unit + property-based + integration tests
    ├── baseline_agent.py     # Scenario-aware deterministic baseline (single + multi-agent)
    ├── client.py             # BoardroomEnv HTTP client
    ├── policy.py             # Shared archetype-aware policy (now includes present_evidence + negotiate)
    └── models.py             # Pydantic Action/Observation models + ActorState dataclass
```

## Why This Environment Matters

Data-driven decision-making inside organisations is notoriously hard to train AI agents on: real boardroom data is confidential, feedback loops are slow, and stakeholder biases are invisible. OpenBoardroom solves this by fully simulating a SaaS company's quarterly review process — noisy KPIs, politically biased advisors, counterfactual "what-if" engines, and a graded final decision — so RL agents can learn persuasive, evidence-backed reasoning at scale. Solving this environment demonstrates an agent can (1) identify causal signals inside noisy tabular data, (2) resist misleading stakeholder framing, and (3) construct defensible strategic decisions. These skills transfer directly to real analyst-support tooling, automated due-diligence systems, and corporate BI co-pilots.

## Judge-Friendly Summary

- Problem: there are very few open environments that test strategic decision quality in messy, high-stakes business settings.
- Core idea: simulate a SaaS boardroom where agents reason over KPIs, stakeholder bias, and counterfactual launch scenarios.
- Technical depth: custom generators, noise injection, stakeholder simulation, explanation grading, reward shaping, and deterministic evaluation.
- Practical value: relevant to BI copilots, analyst automation, due diligence, and enterprise decision support.
- Validation: Dockerized deployment, OpenEnv-compatible APIs, 164 passing tests, reproducible benchmark scores, and a live multi-agent demo.

## Evaluation

| What to run | Command |
|---|---|
| Full test suite | `pip install -e ".[dev]" && python -m pytest my_env/tests -v` |
| Baseline agent benchmark | `python -m my_env.baseline_agent` |
| Reward improvement demo | `python demo_rewards.py` |
| Multi-agent smoke test | `python -c "from my_env import MultiAgentBoardroomEnvironment; e=MultiAgentBoardroomEnvironment(); e.reset(seed=0)"` |
| GRPO training (Colab/GPU) | `python train_grpo.py` |
| LLM inference end-to-end | `python inference.py` (see Inference section above) |
| OpenEnv validator | `./validate-submission.sh <SPACE_URL>` |
| Docker smoke build | `docker build -t boardroom-env .` |

Each episode terminates with `oracle_answer` and `oracle_hit` in the final observation's `metadata`, allowing automatic grading of whether the agent identified the true bottleneck or reached the correct launch verdict.

Built by [Ayush](https://github.com/AyushbhaiPatel) and [Dharmit](https://github.com/dharmitpatel81).
