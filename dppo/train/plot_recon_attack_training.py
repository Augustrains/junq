"""Plot recon/attack training episodes and post-update evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON on line {0}: {1}".format(line_number, exc)) from exc
    return rows


def load_episode_metrics(path):
    return [row for row in _load_rows(path) if "episode" in row and "episode_total_reward" in row]


def load_policy_evaluation_metrics(path):
    return [row for row in _load_rows(path) if row.get("record_type") == "policy_evaluation"]


def rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    result = np.empty_like(values)
    window = max(1, int(window))
    for index in range(values.size):
        start = max(0, index - window + 1)
        result[index] = float(np.mean(values[start:index + 1]))
    return result


def plot_training_curve(rows, output_path, window=10, evaluation_rows=None):
    if not rows:
        raise ValueError("metrics file has no completed recon/attack episodes")
    evaluation_rows = list(evaluation_rows or [])
    episodes = np.asarray([int(row["episode"]) for row in rows], dtype=np.int64)
    total_rewards = np.asarray([float(row.get("episode_total_reward", 0.0)) for row in rows])
    team_rewards = np.asarray([float(row.get("episode_team_reward", 0.0)) for row in rows])
    terminal_rewards = np.asarray([float(row.get("terminal_bonus", 0.0)) for row in rows])
    sim_hours = np.asarray([float(row.get("episode_sim_seconds", 0.0)) / 3600.0 for row in rows])
    qualified = np.asarray([bool(row.get("landing_quality", {}).get("landing_combat_conditions_met", False)) for row in rows])

    fig, (reward_ax, time_ax, eval_ax) = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw={"height_ratios": [2.0, 1.0, 1.3]})
    reward_ax.plot(episodes, total_rewards, color="#2563eb", alpha=0.45, linewidth=1.2, label="Training total reward")
    reward_ax.plot(episodes, rolling_mean(total_rewards, window), color="#0f3d91", linewidth=2.2, label="{0}-episode moving average".format(max(1, int(window))))
    reward_ax.plot(episodes, team_rewards, color="#0f9d76", alpha=0.7, linewidth=1.0, label="Training team reward")
    reward_ax.plot(episodes, terminal_rewards, color="#d97706", alpha=0.75, linewidth=1.0, label="Training terminal reward")
    if np.any(qualified):
        reward_ax.scatter(episodes[qualified], total_rewards[qualified], color="#16a34a", s=28, zorder=4, label="Combat-qualified")
    reward_ax.axhline(0.0, color="#6b7280", linewidth=0.8)
    reward_ax.set_xlabel("Training episode")
    reward_ax.set_ylabel("Reward")
    reward_ax.set_title("Recon + Attack HAPPO Training")
    reward_ax.grid(True, alpha=0.22)
    reward_ax.legend(loc="best", ncol=2)

    time_ax.plot(episodes, sim_hours, color="#7c3aed", marker="o", markersize=3, linewidth=1.3)
    time_ax.axhline(3.0, color="#dc2626", linestyle="--", linewidth=1.0, label="3-hour deadline")
    time_ax.set_xlabel("Training episode")
    time_ax.set_ylabel("Training simulation hours")
    time_ax.grid(True, alpha=0.22)
    time_ax.legend(loc="best")

    if evaluation_rows:
        updates = np.asarray([int(row.get("update", 0)) for row in evaluation_rows], dtype=np.int64)
        eval_rewards = np.asarray([float(row.get("evaluation_total_reward", 0.0)) for row in evaluation_rows])
        eval_hours = np.asarray([float(row.get("evaluation_sim_seconds", 0.0)) / 3600.0 for row in evaluation_rows])
        successes = np.asarray([bool(row.get("success", False)) for row in evaluation_rows])
        eval_ax.plot(updates, eval_rewards, color="#0891b2", marker="o", linewidth=1.5, label="Evaluation total reward")
        if np.any(successes):
            eval_ax.scatter(updates[successes], eval_rewards[successes], color="#16a34a", s=34, zorder=4, label="Evaluation success")
        end_ax = eval_ax.twinx()
        end_ax.plot(updates, eval_hours, color="#be123c", marker="s", markersize=4, linewidth=1.2, label="Evaluation end time")
        end_ax.axhline(3.0, color="#dc2626", linestyle="--", linewidth=1.0)
        end_ax.set_ylabel("Evaluation simulation hours", color="#be123c")
        lines, labels = eval_ax.get_legend_handles_labels()
        lines2, labels2 = end_ax.get_legend_handles_labels()
        eval_ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        eval_ax.text(0.5, 0.5, "No post-update evaluation recorded yet", ha="center", va="center", transform=eval_ax.transAxes)
    eval_ax.set_xlabel("Policy update")
    eval_ax.set_ylabel("Evaluation reward")
    eval_ax.set_title("Fresh evaluation after each policy update")
    eval_ax.grid(True, alpha=0.22)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot recon/attack training metrics.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.jsonl.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--window", type=int, default=10, help="Moving-average episode window.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_episode_metrics(args.metrics)
    evaluations = load_policy_evaluation_metrics(args.metrics)
    output = plot_training_curve(rows, args.output, args.window, evaluations)
    print("training_curve_saved", output, "episodes", len(rows), "evaluations", len(evaluations))


if __name__ == "__main__":
    main()