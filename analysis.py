"""
Analyse Evaluation Results for Specific Q-Learning Configurations and Compare to Random Agent
"""

import pandas as pd
import pickle
from eval_and_visualize import evaluate, evaluate_random, run_eval

# Load experiment results
df = pd.read_csv("experiment_results.csv")

# Print best models by average reward or win rate
df = df.sort_values(by = "win_rate", ascending = False)
print("Top Models by Average Reward:")
print(df)

# Return visualizations for top two Q-learning hyperparameters by win rate
filtered_df = df[:2]
comparative_results = []

for _, row in filtered_df.iterrows():
    model_path = f"pickle_files/Q_table_{int(row['train_hands'])}_{row['gamma']}_{row['decay_rate']}.pickle"

    with open(model_path, "rb") as f:
        Q_table = pickle.load(f)
    
    eval_metrics, q_summary, random_summary, comparison = run_eval(Q_table, label = f"Top_Win_{row['train_hands']}_{row['gamma']}_{row['decay_rate']}", eval_hands=1000)
    comparative_results.append(comparison)

# Save and print comparative results
comparison_df = pd.DataFrame(comparative_results)
comparison_df.to_csv("analysis_results.csv", index=False)
print(comparison_df.sort_values(by="advantage", ascending=False).to_string(index=False))