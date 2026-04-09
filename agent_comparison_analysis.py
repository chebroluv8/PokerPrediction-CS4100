"""
Full Improvement Analysis — Q-Learning Agent vs Random Agent
Task 5: Per-hyperparameter breakdown, leaderboard, and improvement visualizations
Run: python analysis_full.py
(Keep analysis.py for quick top-2 checks)
"""

import pandas as pd
import pickle
import numpy as np
from eval_and_visualize import evaluate, evaluate_random, run_eval
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

os.makedirs("results", exist_ok=True)

# ── Load & rank all experiment results ───────────────────────────────────────
df = pd.read_csv("experiment_results.csv")

# Compute baseline random agent once (used for improvement calculations)
print("=" * 65)
print("  Running random agent baseline (500 hands)...")
print("=" * 65)
random_summary = evaluate_random(eval_hands=500)
rand_wr  = random_summary["win_rate"]
rand_avg = random_summary["avg_reward"]
print(f"  Random agent — win rate: {rand_wr:.3f} | avg reward: {rand_avg:.4f}\n")

# Add improvement columns
df["wr_improvement"]  = df["win_rate"]  - rand_wr
df["avg_improvement"] = df["avg_reward"] - rand_avg

# ── Section 1: Full leaderboard ───────────────────────────────────────────────
print("=" * 65)
print("  FULL MODEL LEADERBOARD  (sorted by win rate)")
print("=" * 65)
ranked = df.sort_values("win_rate", ascending=False).reset_index(drop=True)
ranked.index += 1
pd.set_option("display.float_format", "{:.4f}".format)
print(ranked[["gamma", "decay_rate", "train_hands",
              "avg_reward", "win_rate", "wr_improvement", "avg_improvement"]].to_string())
print()

# ── Section 2: Best model per hyperparameter dimension ───────────────────────
print("=" * 65)
print("  BEST MODEL PER GAMMA (by win rate)")
print("=" * 65)
for g, grp in df.groupby("gamma"):
    best = grp.loc[grp["win_rate"].idxmax()]
    print(f"  γ={g:<6}  →  train_hands={int(best['train_hands']):<8}"
          f"  decay={best['decay_rate']}  "
          f"  win_rate={best['win_rate']:.3f}  "
          f"  avg_reward={best['avg_reward']:.4f}  "
          f"  wr_improvement={best['wr_improvement']:+.3f}")
print()

print("=" * 65)
print("  BEST MODEL PER DECAY RATE (by win rate)")
print("=" * 65)
for d, grp in df.groupby("decay_rate"):
    best = grp.loc[grp["win_rate"].idxmax()]
    print(f"  decay={d:<10}  →  train_hands={int(best['train_hands']):<8}"
          f"  γ={best['gamma']}  "
          f"  win_rate={best['win_rate']:.3f}  "
          f"  avg_reward={best['avg_reward']:.4f}  "
          f"  wr_improvement={best['wr_improvement']:+.3f}")
print()

print("=" * 65)
print("  BEST MODEL PER TRAIN HANDS (by win rate)")
print("=" * 65)
for h, grp in df.groupby("train_hands"):
    best = grp.loc[grp["win_rate"].idxmax()]
    print(f"  hands={int(h):<8}  →  γ={best['gamma']}  "
          f"  decay={best['decay_rate']}  "
          f"  win_rate={best['win_rate']:.3f}  "
          f"  avg_reward={best['avg_reward']:.4f}  "
          f"  wr_improvement={best['wr_improvement']:+.3f}")
print()

# ── Section 3: Top 5 by avg_reward and top 5 by win_rate ─────────────────────
print("=" * 65)
print("  TOP 5 BY AVG REWARD")
print("=" * 65)
top_reward = df.nlargest(5, "avg_reward")[
    ["gamma", "decay_rate", "train_hands", "avg_reward", "avg_improvement", "win_rate"]
].reset_index(drop=True)
top_reward.index += 1
print(top_reward.to_string())
print()

print("=" * 65)
print("  TOP 5 BY WIN RATE")
print("=" * 65)
top_wr = df.nlargest(5, "win_rate")[
    ["gamma", "decay_rate", "train_hands", "win_rate", "wr_improvement", "avg_reward"]
].reset_index(drop=True)
top_wr.index += 1
print(top_wr.to_string())
print()

# ── Section 4: Deep eval on top models (run_eval) ────────────────────────────
TOP_N = 3  # run full eval+visualizations on top N models by win rate
top_models = df.nlargest(TOP_N, "win_rate")

print("=" * 65)
print(f"  DEEP EVALUATION — Top {TOP_N} models by win rate (1000 hands each)")
print("=" * 65)

comparative_results = []

for _, row in top_models.iterrows():
    label = f"{int(row['train_hands'])}_{row['gamma']}_{row['decay_rate']}"
    path  = f"pickle_files/Q_table_{label}.pickle"
    print(f"\n  Loading: {path}")
    with open(path, "rb") as f:
        Q_table = pickle.load(f)

    eval_metrics, q_summary, random_summary, comparison = run_eval(
        Q_table, label=f"Top_{label}", eval_hands=1000
    )
    comparison["wr_improvement"]  = round(q_summary["win_rate"]  - random_summary["win_rate"],  4)
    comparison["avg_improvement"] = round(q_summary["avg_reward"] - random_summary["avg_reward"], 4)
    comparative_results.append(comparison)

comparison_df = pd.DataFrame(comparative_results)
comparison_df.to_csv("analysis_results.csv", index=False)

print("\n" + "=" * 65)
print("  COMPARISON SUMMARY  (Q-Agent vs Random)")
print("=" * 65)
print(comparison_df[[
    "label", "q_win_rate", "random_win_rate", "wr_improvement",
    "q_avg_reward", "random_avg_reward", "avg_improvement"
]].to_string(index=False))
print()

# ── Section 5: Improvement visualizations ─────────────────────────────────────

# 5a: Win rate improvement heatmap (gamma x decay_rate, best train_hands per cell)
def improvement_heatmap(df, metric, title, fname):
    pivot = df.groupby(["gamma", "decay_rate"])[metric].max().unstack()
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Decay Rate")
    ax.set_ylabel("Gamma")
    ax.set_title(title, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=9, color="black")
    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    plt.savefig(f"results/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: results/{fname}.png")

improvement_heatmap(df, "wr_improvement",
                    "Win Rate Improvement Over Random (best per γ/decay)",
                    "heatmap_wr_improvement")
improvement_heatmap(df, "avg_improvement",
                    "Avg Reward Improvement Over Random (best per γ/decay)",
                    "heatmap_avg_improvement")

# 5b: Win rate by train_hands (line per gamma)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Win Rate & Avg Reward by Training Hands", fontweight="bold")

for g, grp in df.groupby("gamma"):
    agg = grp.groupby("train_hands")[["win_rate", "avg_reward"]].mean().reset_index()
    axes[0].plot(agg["train_hands"], agg["win_rate"],  marker="o", label=f"γ={g}")
    axes[1].plot(agg["train_hands"], agg["avg_reward"], marker="o", label=f"γ={g}")

for ax, metric, ylabel in zip(axes,
                               ["win_rate", "avg_reward"],
                               ["Win Rate", "Avg Reward"]):
    ax.axhline(rand_wr if metric == "win_rate" else rand_avg,
               color="red", linestyle="--", linewidth=1.2, label="Random baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Training Hands (log scale)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/improvement_by_train_hands.png", dpi=150)
plt.close()
print("  Saved: results/improvement_by_train_hands.png")

# 5c: Bar chart — top 10 models win rate vs random
top10 = df.nlargest(10, "win_rate").copy()
top10["label"] = top10.apply(
    lambda r: f"γ={r['gamma']}\nd={r['decay_rate']}\nh={int(r['train_hands'])/1000:.0f}k", axis=1
)
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(top10))
bars = ax.bar(x, top10["win_rate"], color="steelblue", label="Q-Agent Win Rate")
ax.axhline(rand_wr, color="red", linestyle="--", linewidth=1.5, label=f"Random baseline ({rand_wr:.3f})")
ax.set_xticks(x)
ax.set_xticklabels(top10["label"], fontsize=8)
ax.set_ylabel("Win Rate")
ax.set_title("Top 10 Models — Win Rate vs Random Agent Baseline", fontweight="bold")
ax.legend()
ax.set_ylim(0.8, 1.0)
for bar, val in zip(bars, top10["win_rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("results/top10_win_rate_vs_random.png", dpi=150)
plt.close()
print("  Saved: results/top10_win_rate_vs_random.png")

print("\n All analysis complete. Results saved to results/\n")