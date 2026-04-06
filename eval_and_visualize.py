from Q_learning import encode_state
from poker_rlcard import LimitHoldEmEnv
import numpy as np
import random
import csv
import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)


def evaluate(Q_table, eval_hands=500):
    env = LimitHoldEmEnv()
    eval_rewards = []
    eval_metrics = []

    for i in range(eval_hands):
        state, player_id = env.reset()
        total_reward = 0
        done = False

        while not done:
            current_state = encode_state(state, player_id)
            legal_actions = list(state["legal_actions"].keys())

            if player_id == 0:
                action = (
                    np.argmax(Q_table[current_state])
                    if current_state in Q_table
                    else random.choice(legal_actions)
                )
            else:
                action = random.choice(legal_actions)

            state, player_id, done = env.step(action)

            if done:
                total_reward = float(env.get_payoffs()[0])

        eval_rewards.append(total_reward)

        if (i + 1) % 50 == 0:
            window_rewards = eval_rewards[-50:]
            eval_metrics.append({
                "hand": i + 1,
                "avg_reward": round(np.mean(window_rewards), 4),
                "win_rate": round(sum(r > 0 for r in window_rewards) / 50, 4),
            })

    summary = {
        "avg_reward": round(np.mean(eval_rewards), 4),
        "win_rate": round(sum(r > 0 for r in eval_rewards) / eval_hands, 4),
        "avg_loss": round(np.mean([r for r in eval_rewards if r < 0] or [0]), 4),
        "avg_win": round(np.mean([r for r in eval_rewards if r > 0] or [0]), 4),
        "total_hands": eval_hands,
    }

    return eval_metrics, summary


def evaluate_random(eval_hands=500):
    env = LimitHoldEmEnv()
    eval_rewards = []

    for _ in range(eval_hands):
        state, player_id = env.reset()
        total_reward = 0
        done = False

        while not done:
            legal_actions = list(state["legal_actions"].keys())
            action = random.choice(legal_actions)
            state, player_id, done = env.step(action)
            if done:
                total_reward = float(env.get_payoffs()[0])

        eval_rewards.append(total_reward)

    return {
        "avg_reward": round(np.mean(eval_rewards), 4),
        "win_rate": round(sum(r > 0 for r in eval_rewards) / eval_hands, 4),
        "avg_loss": round(np.mean([r for r in eval_rewards if r < 0] or [0]), 4),
        "avg_win": round(np.mean([r for r in eval_rewards if r > 0] or [0]), 4),
        "total_hands": eval_hands,
    }


def save_eval_metrics_csv(eval_metrics, label=""):
    filename = f"results/eval_metrics_{label}.csv" if label else "results/eval_metrics.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hand", "avg_reward", "win_rate"])
        writer.writeheader()
        writer.writerows(eval_metrics)
    print(f"✓ Saved {filename}")


def plot_eval_curves(eval_metrics, label=""):
    hands = [m["hand"] for m in eval_metrics]
    rewards = [m["avg_reward"] for m in eval_metrics]
    win_rates = [m["win_rate"] for m in eval_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title = f"Evaluation Curves — {label}" if label else "Evaluation Curves"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(hands, rewards, marker="o", markersize=4, color="steelblue")
    axes[0].set_title("Rolling Avg Reward")
    axes[0].set_xlabel("Eval Hand #")
    axes[0].set_ylabel("Avg Reward")
    axes[0].grid(alpha=0.3)

    axes[1].plot(hands, win_rates, marker="o", markersize=4, color="darkorange")
    axes[1].set_title("Rolling Win Rate")
    axes[1].set_xlabel("Eval Hand #")
    axes[1].set_ylabel("Win Rate")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    filename = f"results/eval_curves_{label}.png" if label else "results/eval_curves.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ Saved {filename}")


def plot_comparison_table(q_summary, random_summary, label=""):
    metrics = ["avg_reward", "win_rate", "avg_win", "avg_loss"]
    q_vals = [q_summary[m] for m in metrics]
    r_vals = [random_summary[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    rows = [[m, str(q), str(r)] for m, q, r in zip(metrics, q_vals, r_vals)]
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Q-Learning Agent", "Random Agent"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 1.8)

    title = f"Q-Learning vs Random Agent — {label}" if label else "Q-Learning vs Random Agent"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    filename = f"results/comparison_table_{label}.png" if label else "results/comparison_table.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {filename}")


def run_eval(Q_table, label="", eval_hands=500):
    print(f"\nEvaluating Q-Learning agent ({label})...")
    eval_metrics, q_summary = evaluate(Q_table, eval_hands=eval_hands)

    print(f"Evaluating random agent baseline...")
    random_summary = evaluate_random(eval_hands=eval_hands)

    print(f"\n{'Metric':<15} {'Q-Learning':<15} {'Random':<10}")
    print("-" * 42)
    for key in ["avg_reward", "win_rate", "avg_win", "avg_loss"]:
        print(f"{key:<15} {str(q_summary[key]):<15} {str(random_summary[key]):<10}")

    save_eval_metrics_csv(eval_metrics, label=label)
    plot_eval_curves(eval_metrics, label=label)
    plot_comparison_table(q_summary, random_summary, label=label)

    return eval_metrics, q_summary, random_summary
