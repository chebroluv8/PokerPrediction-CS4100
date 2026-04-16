"""
Run Training Experiments, Save Pickle Files, and Collect Initial Evaluation Metrics 
"""

from Q_learning import Q_learning
from eval_and_visualize import evaluate, save_eval_metrics_csv
import csv
import itertools
import pickle
import os
import pandas as pd

os.makedirs("training_metrics", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("pickle_files", exist_ok=True)


gammas = [0.7, 0.8, 0.9, 0.95]
decay_rates = [0.999, 0.9999, 0.99999, 0.999995]
num_hands = [10000, 100000, 1000000]

results = []

for gamma, decay_rate, num_hands in itertools.product(gammas, decay_rates, num_hands):

    # Train & Build Q-Table
    print(f"Running experiment: gamma={gamma}, decay_rate={decay_rate}, num_hands={num_hands}")
    Q_table, metrics = Q_learning(num_hands=num_hands, gamma=gamma, epsilon=1, decay_rate=decay_rate)

    # Save Q-Table in Pickle File
    with open(f"pickle_files/Q_table_{num_hands}_{gamma}_{decay_rate}.pickle", "wb") as f:
        pickle.dump(Q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
 
    # Save Training Metrics
    with open(f"training_metrics/training_metrics_{num_hands}_{gamma}_{decay_rate}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hand", "avg_reward", "win_rate"])
        writer.writeheader()
        writer.writerows(metrics)
    
    print(f"Saved Q-Table and Metrics for gamma={gamma}, decay_rate={decay_rate}, num_hands={num_hands}")

    # Evaluate Q-Table Performance
    eval_metrics, summary, situation_actions = evaluate(Q_table, eval_hands = 500)

    # Save Evaluation Summary Metrics
    results.append({"gamma": gamma, 
                    "decay_rate": decay_rate, 
                    "train_hands": num_hands,
                    "avg_reward": summary["avg_reward"],
                    "win_rate": summary["win_rate"],
                    "avg_loss": summary["avg_loss"],
                    "avg_win": summary["avg_win"],
                    "eval_hands": summary["total_hands"]})

    # Save Windowed Evaluation Metrics
    save_eval_metrics_csv(eval_metrics, label = f"{num_hands}_{gamma}_{decay_rate}")

    print(f"Saved Evaluation Metrics for gamma={gamma}, decay_rate={decay_rate}, num_hands={num_hands}")

# Print and save Evaluation Metrics into CSV
eval_results = pd.DataFrame(results)
eval_results.to_csv("experiment_results.csv", index=False)
print(eval_results)