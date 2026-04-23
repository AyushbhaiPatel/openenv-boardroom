"""
demo_rewards.py — Generate reward curves for the hackathon demo WITHOUT a GPU.

Runs a naive policy and the deterministic baseline policy across all difficulty
tiers, then plots reward curves showing:
  - Per-episode scores
  - Rolling mean
  - Difficulty progression (easy → medium → hard)
  - Before/after improvement from naive behavior to the boardroom policy

Usage:
    pip install matplotlib
    python demo_rewards.py

Output:
    demo_output/reward_curve_single_agent.png
    demo_output/reward_curve_multi_agent.png
    demo_output/reward_curve_comparison.png
    demo_output/reward_improvement.png
    demo_output/improvement_summary.csv
    demo_output/scores.csv
"""

from __future__ import annotations

import csv
import os
from typing import List, Dict

import numpy as np

from my_env.models import BoardroomAction
from my_env.policy import ScenarioAwarePolicy
from my_env.server.boardroom_environment import BoardroomEnvironment
from my_env.server.multi_agent_boardroom_environment import MultiAgentBoardroomEnvironment

OUTPUT_DIR = "demo_output"
EPISODES_PER_TIER = 30  # Fast enough to run in <30s
DIFFICULTIES = ["easy", "medium", "hard"]


def naive_action(step: int, difficulty: str) -> Dict:
    """A deliberately weak policy: two obvious queries, then an early generic decision."""
    if difficulty == "hard":
        if step == 1:
            return {"action_type": "query_data", "parameters": {"metric": "revenue"}}
        if step == 2:
            return {"action_type": "query_data", "parameters": {"metric": "churn_rate"}}
        return {
            "action_type": "make_decision",
            "parameters": {
                "decision": "launch feature x",
                "parameters": {},
                "explanation": "Revenue growth matters, so we should launch.",
            },
        }
    if step == 1:
        return {"action_type": "query_data", "parameters": {"metric": "revenue"}}
    if step == 2:
        return {"action_type": "query_data", "parameters": {"metric": "monthly_active_users"}}
    return {
        "action_type": "make_decision",
        "parameters": {
            "decision": "increase growth",
            "parameters": {},
            "explanation": "The company should improve growth based on the available dashboard.",
        },
    }


def run_episodes(EnvClass, multi_agent: bool, episodes_per_tier: int) -> List[Dict]:
    """Run baseline policy across all tiers and return episode records."""
    records = []
    episode_num = 0
    for difficulty in DIFFICULTIES:
        for seed in range(episodes_per_tier):
            env = EnvClass()
            obs = env.reset(seed=seed, difficulty=difficulty)
            policy = ScenarioAwarePolicy(
                difficulty=difficulty,
                snapshot=obs.data_tables,
                multi_agent=multi_agent,
            )
            max_steps = int(obs.metadata.get("max_steps", 20))
            for step in range(1, max_steps + 1):
                if obs.done:
                    break
                a = policy.next_action(step)
                obs = env.step(BoardroomAction(
                    action_type=a["action_type"],
                    parameters=a["parameters"],
                ))
            score = obs.final_score if obs.final_score is not None else obs.metadata.get("final_score", obs.reward or 0.0)
            episode_num += 1
            records.append({
                "episode": episode_num,
                "difficulty": difficulty,
                "score": score,
                "seed": seed,
            })
            print(f"  ep={episode_num:3d} diff={difficulty:6s} seed={seed:2d} score={score:.4f}")
    return records


def run_naive_episodes(EnvClass, multi_agent: bool, episodes_per_tier: int) -> List[Dict]:
    """Run the intentionally weak policy across all tiers."""
    records = []
    episode_num = 0
    for difficulty in DIFFICULTIES:
        for seed in range(episodes_per_tier):
            env = EnvClass()
            obs = env.reset(seed=seed, difficulty=difficulty)
            max_steps = int(obs.metadata.get("max_steps", 20))
            for step in range(1, max_steps + 1):
                if obs.done:
                    break
                action_dict = naive_action(step, difficulty)
                obs = env.step(BoardroomAction(
                    action_type=action_dict["action_type"],
                    parameters=action_dict["parameters"],
                ))
            score = obs.final_score if obs.final_score is not None else obs.metadata.get("final_score", obs.reward or 0.0)
            episode_num += 1
            records.append({
                "episode": episode_num,
                "difficulty": difficulty,
                "score": score,
                "seed": seed,
            })
            print(f"  ep={episode_num:3d} diff={difficulty:6s} seed={seed:2d} score={score:.4f}")
    return records


def rolling_mean(values: List[float], window: int = 10) -> List[float]:
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def plot_curve(records: List[Dict], title: str, output_path: str) -> None:
    try:
        os.makedirs(os.path.join(OUTPUT_DIR, ".matplotlib"), exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUTPUT_DIR, ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        episodes = [r["episode"] for r in records]
        scores = [r["score"] for r in records]
        difficulties = [r["difficulty"] for r in records]
        rm = rolling_mean(scores, window=10)

        diff_colors = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"}
        bg_colors = {"easy": "#d5f5e3", "medium": "#fef9e7", "hard": "#fdedec"}

        fig, ax = plt.subplots(figsize=(12, 5))

        # Shade background by difficulty tier
        prev_diff = difficulties[0]
        start_ep = episodes[0]
        for i, (ep, diff) in enumerate(zip(episodes, difficulties)):
            if diff != prev_diff or i == len(episodes) - 1:
                end_ep = ep if diff != prev_diff else ep + 1
                ax.axvspan(start_ep - 0.5, end_ep - 0.5,
                           alpha=0.25, color=bg_colors[prev_diff], zorder=0)
                start_ep = ep
                prev_diff = diff

        # Scatter per-episode scores coloured by difficulty
        for diff in DIFFICULTIES:
            ep_d = [r["episode"] for r in records if r["difficulty"] == diff]
            sc_d = [r["score"] for r in records if r["difficulty"] == diff]
            ax.scatter(ep_d, sc_d, color=diff_colors[diff], alpha=0.5, s=18, zorder=2)

        # Rolling mean line
        ax.plot(episodes, rm, color="#2c3e50", linewidth=2.5, zorder=3, label="Rolling mean (10 ep)")

        # Difficulty tier labels
        tier_starts = {}
        for r in records:
            if r["difficulty"] not in tier_starts:
                tier_starts[r["difficulty"]] = r["episode"]
        for diff, ep in tier_starts.items():
            ax.axvline(ep, color=diff_colors[diff], linestyle="--", alpha=0.6, linewidth=1)
            ax.text(ep + 0.5, 0.02, diff.upper(), color=diff_colors[diff],
                    fontsize=9, fontweight="bold", va="bottom")

        # Legend
        patches = [mpatches.Patch(color=diff_colors[d], label=d.capitalize()) for d in DIFFICULTIES]
        patches.append(plt.Line2D([0], [0], color="#2c3e50", linewidth=2.5, label="Rolling mean"))
        ax.legend(handles=patches, loc="lower right", fontsize=9)

        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Final Score", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, zorder=1)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping plot. Run: pip install matplotlib")


def plot_comparison(single_records: List[Dict], multi_records: List[Dict], output_path: str) -> None:
    """Side-by-side comparison of single-agent vs multi-agent scores per difficulty."""
    try:
        os.makedirs(os.path.join(OUTPUT_DIR, ".matplotlib"), exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUTPUT_DIR, ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        colors = {"single": "#3498db", "multi": "#e74c3c"}

        for ax, diff in zip(axes, DIFFICULTIES):
            s_scores = [r["score"] for r in single_records if r["difficulty"] == diff]
            m_scores = [r["score"] for r in multi_records if r["difficulty"] == diff]

            x = np.arange(len(s_scores))
            ax.plot(x, s_scores, color=colors["single"], alpha=0.6, linewidth=1, label="Single-agent")
            ax.plot(x, m_scores, color=colors["multi"], alpha=0.6, linewidth=1, label="Multi-agent")
            ax.axhline(np.mean(s_scores), color=colors["single"], linestyle="--", linewidth=1.5,
                       label=f"Single mean: {np.mean(s_scores):.3f}")
            ax.axhline(np.mean(m_scores), color=colors["multi"], linestyle="--", linewidth=1.5,
                       label=f"Multi mean: {np.mean(m_scores):.3f}")
            ax.set_title(diff.capitalize(), fontweight="bold")
            ax.set_xlabel("Episode")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            if ax == axes[0]:
                ax.set_ylabel("Final Score")
            ax.legend(fontsize=7)

        fig.suptitle("Single-Agent vs Multi-Agent Boardroom — Baseline Policy Scores", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping comparison plot.")


def plot_improvement(naive_records: List[Dict], policy_records: List[Dict], output_path: str) -> None:
    """Bar chart showing before/after improvement by difficulty."""
    try:
        os.makedirs(os.path.join(OUTPUT_DIR, ".matplotlib"), exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUTPUT_DIR, ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        naive_means = [np.mean([r["score"] for r in naive_records if r["difficulty"] == d]) for d in DIFFICULTIES]
        policy_means = [np.mean([r["score"] for r in policy_records if r["difficulty"] == d]) for d in DIFFICULTIES]

        x = np.arange(len(DIFFICULTIES))
        width = 0.34
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, naive_means, width, label="Naive agent", color="#95a5a6")
        ax.bar(x + width / 2, policy_means, width, label="Boardroom policy", color="#2ecc71")

        for i, (before, after) in enumerate(zip(naive_means, policy_means)):
            gain = after - before
            ax.text(i, max(before, after) + 0.03, f"+{gain:.2f}", ha="center", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in DIFFICULTIES])
        ax.set_ylabel("Mean final score")
        ax.set_ylim(0, 1.08)
        ax.set_title("Reward Improvement: Naive Agent → Multi-Agent Boardroom Policy", fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping improvement plot.")


def save_csv(records: List[Dict], path: str) -> None:
    fieldnames = sorted({key for record in records for key in record.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved: {path}")


def save_improvement_summary(naive_records: List[Dict], policy_records: List[Dict], path: str) -> None:
    rows = []
    for diff in DIFFICULTIES:
        naive_scores = [r["score"] for r in naive_records if r["difficulty"] == diff]
        policy_scores = [r["score"] for r in policy_records if r["difficulty"] == diff]
        naive_mean = float(np.mean(naive_scores))
        policy_mean = float(np.mean(policy_scores))
        rows.append({
            "difficulty": diff,
            "naive_mean": round(naive_mean, 4),
            "policy_mean": round(policy_mean, 4),
            "absolute_gain": round(policy_mean - naive_mean, 4),
            "relative_gain_pct": round(((policy_mean - naive_mean) / max(naive_mean, 1e-9)) * 100, 1),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["difficulty", "naive_mean", "policy_mean", "absolute_gain", "relative_gain_pct"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def print_summary(label: str, records: List[Dict]) -> None:
    print(f"\n{label} Summary:")
    for diff in DIFFICULTIES:
        scores = [r["score"] for r in records if r["difficulty"] == diff]
        print(f"  {diff:>8s}: mean={np.mean(scores):.4f}  std={np.std(scores):.4f}  "
              f"min={np.min(scores):.4f}  max={np.max(scores):.4f}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print("  OpenBoardroom — Reward Curve Demo")
    print(f"{'='*55}")

    print(f"\n[1/5] Running naive multi-agent baseline ({EPISODES_PER_TIER} eps/tier)...")
    naive_multi_records = run_naive_episodes(MultiAgentBoardroomEnvironment, multi_agent=True, episodes_per_tier=EPISODES_PER_TIER)
    print_summary("Naive Multi-Agent", naive_multi_records)

    print(f"\n[2/5] Running single-agent baseline ({EPISODES_PER_TIER} eps/tier)...")
    single_records = run_episodes(BoardroomEnvironment, multi_agent=False, episodes_per_tier=EPISODES_PER_TIER)
    print_summary("Single-Agent", single_records)

    print(f"\n[3/5] Running multi-agent baseline ({EPISODES_PER_TIER} eps/tier)...")
    multi_records = run_episodes(MultiAgentBoardroomEnvironment, multi_agent=True, episodes_per_tier=EPISODES_PER_TIER)
    print_summary("Multi-Agent", multi_records)

    print("\n[4/5] Saving CSV...")
    all_records = [{"env": "single", **r} for r in single_records] + \
                  [{"env": "multi", **r} for r in multi_records] + \
                  [{"env": "naive_multi", **r} for r in naive_multi_records]
    save_csv(naive_multi_records, os.path.join(OUTPUT_DIR, "naive_multi_agent_scores.csv"))
    save_csv(single_records, os.path.join(OUTPUT_DIR, "single_agent_scores.csv"))
    save_csv(multi_records, os.path.join(OUTPUT_DIR, "multi_agent_scores.csv"))
    save_csv(all_records, os.path.join(OUTPUT_DIR, "all_scores.csv"))
    save_improvement_summary(
        naive_multi_records,
        multi_records,
        os.path.join(OUTPUT_DIR, "improvement_summary.csv"),
    )

    print("\n[5/5] Plotting reward curves...")
    plot_curve(
        single_records,
        "Single-Agent Boardroom — Baseline Policy (Easy → Medium → Hard)",
        os.path.join(OUTPUT_DIR, "reward_curve_single_agent.png"),
    )
    plot_curve(
        multi_records,
        "Multi-Agent Boardroom — Baseline Policy (Easy → Medium → Hard)",
        os.path.join(OUTPUT_DIR, "reward_curve_multi_agent.png"),
    )
    plot_comparison(
        single_records, multi_records,
        os.path.join(OUTPUT_DIR, "reward_curve_comparison.png"),
    )
    plot_improvement(
        naive_multi_records,
        multi_records,
        os.path.join(OUTPUT_DIR, "reward_improvement.png"),
    )

    print(f"\n{'='*55}")
    print(f"  Done! Charts saved to ./{OUTPUT_DIR}/")
    print(f"  Use these in your HuggingFace blog post and pitch deck.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
