import pandas as pd
import pickle
import numpy as np
from eval_and_visualize import evaluate, evaluate_random, run_eval
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("experiment_results.csv")

# Compute baseline random agent 
random_summary = evaluate_random(eval_hands=500)
rand_wr = random_summary["win_rate"]
rand_avg = random_summary["avg_reward"]
print(f"Random agent — win rate: {rand_wr:.3f} | avg reward: {rand_avg:.4f}")

# Add improvement columns
df["wr_improvement"] = df["win_rate"] - rand_wr
df["avg_improvement"] = df["avg_reward"] - rand_avg

# Rank all models based on win rate with 500 eval hands
print("FULL MODEL LEADERBOARD BY WIN RATE")
ranked = df.sort_values("win_rate", ascending=False).reset_index(drop=True)
ranked.index += 1
print(ranked[["gamma", "decay_rate", "train_hands", "avg_reward", "win_rate", "wr_improvement", "avg_improvement"]])
print()

print("BEST MODEL PER GAMMA (by win rate)")
for g, grp in df.groupby("gamma"):
    best = grp.loc[grp["win_rate"].idxmax()]
    print(f"γ={g:<6} train_hands={int(best['train_hands']):<8} decay={best['decay_rate']:<8} win_rate={best['win_rate']:.3f} avg_reward={best['avg_reward']:.4f} wr_improvement={best['wr_improvement']:+.3f}")
print()

print("BEST MODEL PER DECAY RATE (by win rate)")
for d, grp in df.groupby("decay_rate"):
    best = grp.loc[grp["win_rate"].idxmax()]
    print(f"decay={d:<10} train_hands={int(best['train_hands']):<8} γ={best['gamma']} win_rate={best['win_rate']:.3f} avg_reward={best['avg_reward']:.4f} wr_improvement={best['wr_improvement']:+.3f}")
print()

print("BEST MODEL PER TRAIN HANDS (by win rate)")
for h, grp in df.groupby("train_hands"):
    best = grp.loc[grp["win_rate"].idxmax()]
    print(f"hands={int(h):<8} γ={best['gamma']} decay={best['decay_rate']:<8} win_rate={best['win_rate']:.3f} avg_reward={best['avg_reward']:.4f} wr_improvement={best['wr_improvement']:+.3f}")
print()

print("TOP 5 BY AVG REWARD")
top_reward = df.nlargest(5, "avg_reward")
top_reward = top_reward[["gamma", "decay_rate", "train_hands", "avg_reward", "avg_improvement", "win_rate"]].reset_index(drop=True)
top_reward.index += 1
print(top_reward)
print()

print("TOP 5 BY WIN RATE")
top_wr = df.nlargest(5, "win_rate")
top_wr = top_wr[["gamma", "decay_rate", "train_hands", "win_rate", "wr_improvement", "avg_reward"]].reset_index(drop=True)
top_wr.index += 1
print(top_wr)
print()

# Run full eval & visualizations on top 5 models by win rate
comparative_results = []

for _, row in top_wr.iterrows():
    label = f"{int(row['train_hands'])}_{row['gamma']}_{row['decay_rate']}"
    path = f"pickle_files/Q_table_{int(row['train_hands'])}_{row['gamma']}_{row['decay_rate']}.pickle"
    with open(path, "rb") as f:
        Q_table = pickle.load(f)

    eval_metrics, q_summary, random_summary, comparison = run_eval(Q_table, label=f"Top_{label}", eval_hands=1000)

    comparison["wr_improvement"] = round(q_summary["win_rate"] - random_summary["win_rate"],  4)
    comparison["avg_improvement"] = round(q_summary["avg_reward"] - random_summary["avg_reward"], 4)
    comparative_results.append(comparison)

comparison_df = pd.DataFrame(comparative_results)
comparison_df.to_csv("analysis_results.csv", index=False)

print("COMPARISON SUMMARY (Q-Agent vs Random)")
print(comparison_df[["label", "q_win_rate", "random_win_rate", "wr_improvement", "q_avg_reward", "random_avg_reward", "avg_improvement"]])

# Line chart for win rate by number of training hands (line per gamma)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Win Rate & Avg Reward by Training Hands", fontweight="bold")

for g, grp in df.groupby("gamma"):
    agg = (grp.groupby("train_hands")[["win_rate", "avg_reward"]].mean().reset_index())
    axes[0].plot(agg["train_hands"], agg["win_rate"], marker="o", label=f"γ={g}")
    axes[1].plot(agg["train_hands"], agg["avg_reward"], marker="o", label=f"γ={g}")

axes[0].axhline(rand_wr, color="red", linestyle="--", linewidth=1.2, label="Random baseline")
axes[0].set_xscale("log")
axes[0].set_xlabel("Training Hands (log scale)")
axes[0].set_ylabel("Win Rate")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].axhline(rand_avg, color="red", linestyle="--", linewidth=1.2, label="Random baseline")
axes[1].set_xscale("log")
axes[1].set_xlabel("Training Hands (log scale)")
axes[1].set_ylabel("Avg Reward")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/improvement_by_train_hands.png", dpi=150)
plt.close()

# Create bar chart for top 10 models by win rate
top10 = df.nlargest(10, "win_rate")
# AI Citation - Used AI to help with formatting the labels for each column
top10["label"] = top10.apply(lambda r: f"γ={r['gamma']}\nd={r['decay_rate']}\nh={int(r['train_hands'])/1000:.0f}k", axis = 1)

plt.figure(figsize=(13, 5))
x = np.arange(len(top10))
bars = plt.bar(x, top10["win_rate"], color="steelblue", label="Q-Agent Win Rate")
plt.xticks(x, top10["label"], fontsize=8)
plt.ylabel("Win Rate")
plt.title("Top 10 Models By Win Rate", fontweight="bold")
plt.ylim(0.8, 1.0)
plt.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
plt.tight_layout()
plt.savefig("results/top10_win_rate.png", dpi=150)
plt.close()

# Create bar chart for top 10 models by average reward
top10 = df.nlargest(10, "avg_reward")
top10["label"] = top10.apply(lambda r: f"γ={r['gamma']}\nd={r['decay_rate']}\nh={int(r['train_hands'])/1000:.0f}k", axis = 1)

plt.figure(figsize=(13, 5))
x = np.arange(len(top10))
bars = plt.bar(x, top10["avg_reward"], color="steelblue", label="Q-Agent Average Reward")
plt.xticks(x, top10["label"], fontsize=8)
plt.ylabel("Average Reward")
plt.title("Top 10 Models By Average Reward", fontweight="bold")
plt.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
plt.tight_layout()
plt.savefig("results/top10_avg_reward.png", dpi=150)
plt.close()


