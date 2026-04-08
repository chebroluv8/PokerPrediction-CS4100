"""
Analyzing Evaluation Plots & Comparing to Random Agent
"""

import pandas as pd
import pickle
from eval_and_visualize import evaluate, evaluate_random

# Load experiment results
df = pd.read_csv("experiment_results.csv")

# Print best models by average reward
df = df.sort_values(by = "avg_reward", ascending = False)
print("Top Models by Average Reward:")
print(df)


comparative_results = []
random_summary = evaluate_random(eval_hands=1000)

for _, row in df.iterrows():
    label = row["label"]
    model_path = f"models/{label}.pickle"

    # Load model
    with open(model_path, "rb") as f:
        Q_table = pickle.load(f)

    # Evaluate Q-learning agent
    eval_metrics, q_summary = evaluate(Q_table, eval_hands=1000)

    # Store comparison
    comparative_results.append({
        "label": label,
        "q_avg_reward": q_summary["avg_reward"],
        "q_win_rate": q_summary["win_rate"],
        "random_avg_reward": random_summary["avg_reward"],
        "random_win_rate": random_summary["win_rate"],
        "advantage": q_summary["avg_reward"] - random_summary["avg_reward"]})

    # NEED TO ADD VISUALIZATIONS & MORE COMPARISON ANALYSIS

# Save and print comparative results
comparison_df = pd.DataFrame(comparative_results)
comparison_df.to_csv("analysis_results.csv", index=False)
print(comparison_df.sort_values(by="advantage", ascending=False).to_string(index=False))