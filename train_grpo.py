"""
train_grpo.py — Colab-ready GRPO training script for MultiAgentBoardroomEnvironment.

Usage (Colab):
    !pip install unsloth trl transformers datasets matplotlib
    !python train_grpo.py

Requirements:
    - unsloth
    - trl >= 0.8
    - transformers
    - matplotlib (for reward curves)
"""

from __future__ import annotations

import csv
import copy
import json
import os
import sys
import warnings
# Suppress pkg_resources deprecation from trl/transformers internals
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="pkg_resources")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "grpo_output", ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Imports — graceful fallback for environments without GPU libs
# ---------------------------------------------------------------------------

try:
    from unsloth import FastLanguageModel
    _UNSLOTH_AVAILABLE = True
except ImportError:
    _UNSLOTH_AVAILABLE = False
    print("[WARN] unsloth not available — model loading will be skipped in smoke-test mode.")

try:
    from trl import GRPOConfig, GRPOTrainer
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False
    print("[WARN] trl not available — GRPOTrainer will be mocked in smoke-test mode.")

from my_env.models import BoardroomAction

# This script is an offline/server-side training harness. It imports the
# environment class directly so GRPO can score local rollouts without going
# through the HTTP client/server boundary used by external agents.
from my_env.server.multi_agent_boardroom_environment import MultiAgentBoardroomEnvironment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-6
MAX_EPISODES = 500
CURRICULUM_WINDOW = 50
CURRICULUM_THRESHOLD = 0.65
DIFFICULTIES = ["easy", "medium", "hard"]
LOG_DIR = "./grpo_output"

# ---------------------------------------------------------------------------
# Curriculum Scheduler
# ---------------------------------------------------------------------------


@dataclass
class CurriculumScheduler:
    """Advances difficulty tier when rolling mean score exceeds threshold."""

    difficulty: str = "easy"
    rolling_scores: deque = field(default_factory=lambda: deque(maxlen=CURRICULUM_WINDOW))
    threshold: float = CURRICULUM_THRESHOLD
    _difficulty_index: int = 0

    def record(self, score: float) -> None:
        self.rolling_scores.append(score)

    def maybe_advance(self) -> bool:
        """Returns True if difficulty was advanced."""
        if len(self.rolling_scores) < CURRICULUM_WINDOW:
            return False
        mean = sum(self.rolling_scores) / len(self.rolling_scores)
        if mean >= self.threshold and self._difficulty_index < len(DIFFICULTIES) - 1:
            self._difficulty_index += 1
            self.difficulty = DIFFICULTIES[self._difficulty_index]
            self.rolling_scores.clear()
            print(f"[CURRICULUM] Advanced to difficulty: {self.difficulty} (mean={mean:.4f})")
            return True
        return False


# ---------------------------------------------------------------------------
# Reward Logger — saves CSV + plots reward curve
# ---------------------------------------------------------------------------


class RewardLogger:
    """Logs per-episode rewards to CSV and generates a reward curve plot."""

    def __init__(self, log_dir: str = LOG_DIR) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "rewards.csv")
        self.plot_path = os.path.join(log_dir, "reward_curve.png")
        self.episodes: List[Dict[str, Any]] = []
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["episode", "final_score", "difficulty", "rolling_mean"],
        )
        self._writer.writeheader()
        self._rolling: deque = deque(maxlen=20)

    def log(self, episode: int, final_score: float, difficulty: str) -> None:
        self._rolling.append(final_score)
        rolling_mean = sum(self._rolling) / len(self._rolling)
        row = {
            "episode": episode,
            "final_score": round(final_score, 4),
            "difficulty": difficulty,
            "rolling_mean": round(rolling_mean, 4),
        }
        self._writer.writerow(row)
        self._csv_file.flush()
        self.episodes.append(row)

    def plot(self) -> None:
        """Save a reward curve PNG. Requires matplotlib."""
        try:
            os.makedirs(os.path.join(LOG_DIR, ".matplotlib"), exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", os.path.join(LOG_DIR, ".matplotlib"))
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            episodes = [e["episode"] for e in self.episodes]
            scores = [e["final_score"] for e in self.episodes]
            rolling = [e["rolling_mean"] for e in self.episodes]
            difficulties = [e["difficulty"] for e in self.episodes]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(episodes, scores, alpha=0.3, s=10, color="steelblue", label="Episode score")
            ax.plot(episodes, rolling, color="orange", linewidth=2, label="Rolling mean (20 ep)")

            # Shade by difficulty
            colors = {"easy": "#d4edda", "medium": "#fff3cd", "hard": "#f8d7da"}
            prev_diff = difficulties[0] if difficulties else "easy"
            start = 0
            for i, d in enumerate(difficulties):
                if d != prev_diff or i == len(difficulties) - 1:
                    ax.axvspan(start, i, alpha=0.15, color=colors.get(prev_diff, "white"), label=f"_{prev_diff}")
                    start = i
                    prev_diff = d

            ax.set_xlabel("Episode")
            ax.set_ylabel("Final Score")
            ax.set_title("MultiAgentBoardroom — GRPO Training Reward Curve")
            ax.legend(loc="lower right")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_path, dpi=150)
            plt.close()
            print(f"[PLOT] Reward curve saved to {self.plot_path}")
        except ImportError:
            print("[WARN] matplotlib not available — skipping reward curve plot.")

    def close(self) -> None:
        self._csv_file.close()
        self.plot()


# ---------------------------------------------------------------------------
# Reward function factory
# ---------------------------------------------------------------------------


def _snapshot_env(env: MultiAgentBoardroomEnvironment) -> Dict[str, Any]:
    return copy.deepcopy(env.__dict__)


def _restore_env(env: MultiAgentBoardroomEnvironment, snapshot: Dict[str, Any]) -> None:
    env.__dict__.clear()
    env.__dict__.update(copy.deepcopy(snapshot))


def make_reward_fn(env: MultiAgentBoardroomEnvironment, scheduler: CurriculumScheduler,
                   logger: Optional["RewardLogger"] = None) -> Callable:
    """Returns a reward function compatible with GRPOTrainer."""
    episode_counter = [0]

    def reward_fn(completions: List[str], prompts: Optional[List[str]] = None, **kwargs: Any) -> List[float]:
        rewards: List[float] = []
        parsed_actions: List[Optional[BoardroomAction]] = []
        batch_state = _snapshot_env(env)

        for completion in completions:
            try:
                action_dict = json.loads(completion.strip())
                parsed_actions.append(BoardroomAction(
                    action_type=action_dict.get("action_type", "query_data"),
                    parameters=action_dict.get("parameters", {"metric": "revenue"}),
                ))
            except (json.JSONDecodeError, KeyError, ValueError):
                parsed_actions.append(None)
                rewards.append(0.0)
                continue

            _restore_env(env, batch_state)
            action = parsed_actions[-1]
            if action is None:
                continue
            obs = env.step(action)
            reward = float(obs.reward)

            print(f"[STEP] action={action.action_type} reward={reward:.4f} done={obs.done}")
            rewards.append(reward)

        _restore_env(env, batch_state)
        canonical_action = next((action for action in parsed_actions if action is not None), None)
        if canonical_action is not None:
            obs = env.step(canonical_action)
            if obs.done:
                final_score = float(obs.metadata.get("final_score", obs.reward))
                episode_counter[0] += 1
                scheduler.record(final_score)
                scheduler.maybe_advance()
                if logger:
                    logger.log(episode_counter[0], final_score, scheduler.difficulty)
                print(
                    f"[END] episode={episode_counter[0]} final_score={final_score:.4f} "
                    f"difficulty={scheduler.difficulty}"
                )
                env.reset(difficulty=scheduler.difficulty)

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_system_prompt(obs_metadata: Dict[str, Any]) -> str:
    objective = obs_metadata.get("objective", "Make a strategic decision.")
    brief = obs_metadata.get("brief", "")
    difficulty = obs_metadata.get("difficulty", "medium")
    return (
        f"You are the Chief Data Officer (CDO) of a SaaS company.\n"
        f"Difficulty: {difficulty}\n"
        f"Objective: {objective}\n"
        f"Brief: {brief}\n\n"
        f"Respond with a JSON object representing your next action. Example:\n"
        f'{{"action_type": "query_data", "parameters": {{"metric": "revenue"}}}}\n\n'
        f"Valid action types: query_data, analyze_trend, simulate_counterfactual, "
        f"consult_stakeholder, make_decision, present_evidence, negotiate."
    )


# ---------------------------------------------------------------------------
# Dataset builder (minimal — GRPO uses online rollouts)
# ---------------------------------------------------------------------------


def build_prompt_dataset(env: MultiAgentBoardroomEnvironment, n_samples: int = 64) -> List[Dict]:
    """Build a small seed dataset of prompts for GRPO warm-start."""
    samples = []
    for i in range(n_samples):
        obs = env.reset(seed=i, difficulty="easy")
        prompt = build_system_prompt(obs.metadata)
        samples.append({"prompt": prompt})
    return samples


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    print("[INIT] Initialising MultiAgentBoardroomEnvironment...")
    os.makedirs(LOG_DIR, exist_ok=True)
    env = MultiAgentBoardroomEnvironment()
    scheduler = CurriculumScheduler()
    logger = RewardLogger(log_dir=LOG_DIR)
    obs = env.reset(seed=0, difficulty=scheduler.difficulty)

    # Load model
    if _UNSLOTH_AVAILABLE:
        print(f"[INIT] Loading model: {MODEL_NAME}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    else:
        print("[INIT] Unsloth not available — using mock model for smoke test.")
        model = None
        tokenizer = None

    # Build reward function
    reward_fn = make_reward_fn(env, scheduler, logger)

    # Build dataset
    dataset_samples = build_prompt_dataset(env, n_samples=64)

    if _TRL_AVAILABLE and model is not None:
        try:
            from datasets import Dataset
            dataset = Dataset.from_list(dataset_samples)
        except ImportError:
            print("[WARN] datasets not available — using list directly.")
            dataset = dataset_samples  # type: ignore

        config = GRPOConfig(
            output_dir=LOG_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            max_new_tokens=MAX_NEW_TOKENS,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=100,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[reward_fn],
            args=config,
            train_dataset=dataset,
        )

        print("[TRAIN] Starting GRPO training...")
        try:
            trainer.train()
        finally:
            logger.close()
            print(f"[TRAIN] Rewards CSV: {logger.csv_path}")
            print(f"[TRAIN] Reward curve: {logger.plot_path}")
        print("[TRAIN] Training complete.")
    else:
        # Smoke-test mode: simulate episodes to generate a reward curve
        print("[SMOKE] Running smoke-test rollouts (no model)...")
        env.reset(seed=0, difficulty=scheduler.difficulty)
        test_completions = [
            '{"action_type": "query_data", "parameters": {"metric": "revenue"}}',
            '{"action_type": "query_data", "parameters": {"metric": "churn_rate"}}',
            '{"action_type": "present_evidence", "parameters": {"target": "cfo", "metric": "support_load", "value": 0.9, "interpretation": "risk data shows trend"}}',
            '{"action_type": "make_decision", "parameters": {"decision": "reduce churn", "parameters": {}, "explanation": "churn is the bottleneck based on data analysis"}}',
        ]
        rewards: List[float] = []
        for completion in test_completions:
            rewards.extend(reward_fn([completion]))
        print(f"[SMOKE] Rewards: {rewards}")
        logger.close()
        print(f"[SMOKE] Rewards CSV: {logger.csv_path}")
        print(f"[SMOKE] Reward curve: {logger.plot_path}")
        print("[SMOKE] Smoke test passed.")


if __name__ == "__main__":
    main()
