from Q_learning import Q_learning, encode_state
from poker_rlcard import LimitHoldEmEnv
import numpy as np
import random
import csv
import itertools

gammas      = [0.7, 0.9, 0.99]
decay_rates = [0.999, 0.9999, 0.99999]
num_hands   = 10000

results = []

for gamma, decay_rate in itertools.product(gammas, decay_rates):
    print(f"Running experiment: gamma={gamma}, decay_rate={decay_rate} ...")

    Q_table = Q_learning(
        num_hands=num_hands,
        gamma=gamma,
        epsilon=1,
        decay_rate=decay_rate
    )

    env = LimitHoldEmEnv()
    eval_hands = 500
    eval_rewards = []

    for _ in range(eval_hands):
        state, player_id = env.reset()
        total_reward = 0
        done = False

        while not done:
            current_state = encode_state(state, player_id)
            legal_actions = list(state["legal_actions"].keys())

            if player_id == 0:
                if current_state in Q_table:
                    action = np.argmax(Q_table[current_state])
                else:
                    action = random.choice(legal_actions)
            else:
                action = random.choice(legal_actions)

            state, player_id, done = env.step(action)

            if done and player_id == 0:
                total_reward = float(env.get_payoffs()[0])

        eval_rewards.append(total_reward)

    avg_reward = round(np.mean(eval_rewards), 4)
    win_rate   = round(sum(r > 0 for r in eval_rewards) / eval_hands, 4)

    print(f"  → avg reward: {avg_reward}, win rate: {win_rate}\n")

    results.append({
        "gamma": gamma,
        "decay_rate": decay_rate,
        "num_hands": num_hands,
        "avg_reward": avg_reward,
        "win_rate": win_rate
    })

csv_file = "experiment_results.csv"
fieldnames = ["gamma", "decay_rate", "num_hands", "avg_reward", "win_rate"]

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {csv_file}")

print(f"\n{'gamma':<8} {'decay_rate':<12} {'avg_reward':<12} {'win_rate':<10}")
print("-" * 45)
for r in sorted(results, key=lambda x: x["avg_reward"], reverse=True):
    print(f"{r['gamma']:<8} {r['decay_rate']:<12} {r['avg_reward']:<12} {r['win_rate']:<10}")