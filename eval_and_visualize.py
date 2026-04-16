from Q_learning import encode_state
from poker_rlcard import LimitHoldEmEnv
import numpy as np
import random
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results", exist_ok=True)

STREETS = ["Preflop", "Flop", "Turn", "River"]
HAND_BUCKETS = ["Weak", "Mediocre", "Strong"]
ACTION_NAMES = ["Call", "Raise", "Fold", "Check"]

def evaluate(Q_table, eval_hands=500):
    """
    Parameters: Q-Table, number of evaluation hands (int)
    Does: Runs evaluation loop based on policy established by Q-table, tracks running metrics, and calculates overall evaluation metrics and actions taken
    Returns: running evaluation metrics (list of dictionaries), summary of overall metrics (dictionary), action distribution per situation (dictionary)
    """
    env = LimitHoldEmEnv()
    eval_rewards = []
    eval_metrics = []
 
    situation_actions = {(street, hb): np.zeros(env.num_actions) for street in range(4) for hb in range(3)}
    total_actions = 0
 
    for i in range(eval_hands):
        state, player_id = env.reset()
        total_reward = 0
        done = False
 
        while not done:
            current_state = encode_state(state, player_id)
            legal_actions = list(state["legal_actions"].keys())

            if player_id == 0:
                total_actions += 1

                if current_state in Q_table:
                    action = np.argmax(Q_table[current_state])
                else:
                    action = random.choice(legal_actions)

                street, hand_bucket = current_state[0], current_state[1]
                situation_actions[(street, hand_bucket)][action] += 1

            else:
                action = random.choice(legal_actions)

            state, player_id, done = env.step(action)

            if done:
                total_reward = float(env.get_payoffs()[0])

        eval_rewards.append(total_reward)

        if (i + 1) % 50 == 0:
            window_rewards = eval_rewards[-50:]
            eval_metrics.append({"hand": i + 1, "avg_reward": round(np.mean(window_rewards), 4), "win_rate": round(sum(r > 0 for r in window_rewards) / 50, 4)})
 
    summary = {
        "avg_reward": round(np.mean(eval_rewards), 4),
        "win_rate": round(sum(r > 0 for r in eval_rewards) / eval_hands, 4),
        "avg_loss": round(np.mean([r for r in eval_rewards if r < 0] or [0]), 4),
        "avg_win": round(np.mean([r for r in eval_rewards if r > 0] or [0]), 4),
        "total_hands": eval_hands,
    }
 
    return eval_metrics, summary, situation_actions

def evaluate_random(eval_hands=500):
    """
    Parameters: number of evaluation hands (int)
    Does: Evaluates performance of an individual/agent playing randomly
    Returns: summary of key metrics (average reward, win rate, average loss, average win, and total hands played) (dictionary)
    """
    env = LimitHoldEmEnv()
    eval_rewards = []

    for i in range(eval_hands):
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

        summary = {
        "avg_reward": round(np.mean(eval_rewards), 4),
        "win_rate": round(sum(r > 0 for r in eval_rewards) / eval_hands, 4),
        "avg_loss": round(np.mean([r for r in eval_rewards if r < 0] or [0]), 4),
        "avg_win": round(np.mean([r for r in eval_rewards if r > 0] or [0]), 4),
        "total_hands": eval_hands,
        }

    return summary


def save_eval_metrics_csv(eval_metrics, label=""):
    """
    Parameters: running evaluation metrics (list of dictionaries), label (str) to indicate which model was used
    Does: Saves running evaluation metrics to a csv
    Returns: N/A
    """
    filename = f"results/eval_metrics_{label}.csv" 
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hand", "avg_reward", "win_rate"])
        writer.writeheader()
        writer.writerows(eval_metrics)


def plot_eval_curves(eval_metrics, label=""):
    """
    Parameters: running evaluation metrics (list of dictionaries), label (str) to indicate which model was used
    Does: Plots running average reward and win rate over evaluation hands
    Returns: N/A
    """
    hands = [m["hand"] for m in eval_metrics]
    rewards = [m["avg_reward"] for m in eval_metrics]
    win_rates = [m["win_rate"] for m in eval_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title = f"Evaluation Curves — {label}"  
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
    filename = f"results/eval_curves_{label}.png" 
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_comparison_table(q_summary, random_summary, label=""):
    """
    Parameters: summary metrics from trained agent (dict), summary metrics from random agent (dict), label to indicate model used (str)
    Does: Creates a table comparing all of the summary metrics between the random and trained agent from evaluation
    Returns: N/A
    """
    metrics = ["avg_reward", "win_rate", "avg_win", "avg_loss"]
    q_vals = [q_summary[m] for m in metrics]
    r_vals = [random_summary[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    rows = [[m, str(q), str(r)] for m, q, r in zip(metrics, q_vals, r_vals)]
    table = ax.table(cellText=rows, colLabels=["Metric", "Q-Learning Agent", "Random Agent"], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 1.8)

    title = f"Q-Learning vs Random Agent — {label}" 
    fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    filename = f"results/comparison_table_{label}.png" 
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

def plot_situation_heatmap(situation_actions, label=""):
    """
    Parameters: action distribution across situations (dictionary), label to indicate model used (str)
    Does: Creates a heatmap displaying the normalized distribution of actions taken in each possible situation (combination of street and hand bucket)
    Returns: N/A
    """
    row_labels = []
    matrix = []
 
    for street in range(4):
        for hand_bucket in range(3):
            counts = situation_actions[(street, hand_bucket)]
            total = counts.sum()
            row_labels.append(f"{STREETS[street]} / {HAND_BUCKETS[hand_bucket]}")
            matrix.append(counts / total)
 
    sns.heatmap(np.array(matrix), annot=True, fmt = ".2f", cmap = "magma", 
                xticklabels=ACTION_NAMES, yticklabels=row_labels, cbar_kws={"label": "Proportion of Actions"})

    plt.xlabel("Action")
    plt.ylabel("Situation (Street / Hand Strength)")
    plt.title('Normalized Distribution of Actions by Situation')
 
    plt.tight_layout()
    filename = f"results/situation_heatmap_{label}.png" 
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def run_eval(Q_table, label="", eval_hands=500):
    """
    Parameters: Q_table, label to indicate Q_table/model used (str), number of evaluation hands (int)
    Does: Runs full evaluation loop, saves running evaluation metrics in csv, and creates performance/comparison plots (evaluation curves, comparison table, situation heatmap)
    Returns: running evaluation metrics, summary of trained agent's metrics, summary of random agent's metrics, and dictionary containing comparison metrics
    """
    print(f"Evaluating Q-Learning agent ({label})")
    eval_metrics, q_summary, situation_actions = evaluate(Q_table, eval_hands=eval_hands)
 
    print(f"Evaluating random agent baseline")
    random_summary = evaluate_random(eval_hands=eval_hands)

    print(f"{'Metric':<15} {'Q-Learning':<15} {'Random':<10}")
    for key in ["avg_reward", "win_rate", "avg_win", "avg_loss"]:
        print(f"{key:<15} {str(q_summary[key]):<15} {str(random_summary[key]):<10}")
 
    save_eval_metrics_csv(eval_metrics, label=label)
    plot_eval_curves(eval_metrics, label=label)
    plot_comparison_table(q_summary, random_summary, label=label)
    plot_situation_heatmap(situation_actions, label=label)
 
    comparison = {
        "label": label,
        "q_avg_reward": q_summary["avg_reward"],
        "q_win_rate": q_summary["win_rate"],
        "q_avg_win": q_summary["avg_win"],
        "q_avg_loss": q_summary["avg_loss"],
        "random_avg_reward": random_summary["avg_reward"],
        "random_win_rate": random_summary["win_rate"],
        "advantage": round(q_summary["avg_reward"] - random_summary["avg_reward"], 4),
    }
 
    return eval_metrics, q_summary, random_summary, comparison