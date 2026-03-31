from Q_learning import encode_state
from poker_rlcard import LimitHoldEmEnv
import numpy as np
import random
import csv

def Q_learning_with_metrics(num_hands=10000, gamma=0.99, epsilon=1, decay_rate=0.9999, window=500):
    env = LimitHoldEmEnv()
    Q_table = {}
    num_updates = {}
    reward_list = []
    metrics = []

    for i in range(num_hands):
        state, player_id = env.reset()
        total_reward = 0
        done = False

        while not done:
            current_state = encode_state(state, player_id)
            legal_actions = list(state["legal_actions"].keys())

            if current_state not in Q_table:
                Q_table[current_state] = np.zeros(env.num_actions)
                num_updates[current_state] = np.zeros(env.num_actions)

            if np.random.random() <= epsilon:
                action = random.choice(legal_actions)
            else:
                action = np.argmax(Q_table[current_state])

            next_state_raw, next_player_id, done = env.step(action)

            if player_id == 0:
                if done:
                    reward = float(env.get_payoffs()[player_id])
                    total_reward += reward
                    eta = 1 / (1 + num_updates[current_state][action])
                    Q_table[current_state][action] = ((1 - eta) * Q_table[current_state][action]) + (eta * reward)
                    num_updates[current_state][action] += 1
                else:
                    reward = 0
                    next_state = encode_state(next_state_raw, 0)
                    if next_state not in Q_table:
                        Q_table[next_state] = np.zeros(env.num_actions)
                        num_updates[next_state] = np.zeros(env.num_actions)
                    eta = 1 / (1 + num_updates[current_state][action])
                    V = np.max(Q_table[next_state])
                    Q_table[current_state][action] = ((1 - eta) * Q_table[current_state][action]) + (eta * (reward + (gamma * V)))
                    num_updates[current_state][action] += 1

            state, player_id = next_state_raw, next_player_id

        reward_list.append(total_reward)
        epsilon *= decay_rate

        if (i + 1) % window == 0:
            window_rewards = reward_list[-window:]
            avg_reward = round(np.mean(window_rewards), 4)
            win_rate = round(sum(r > 0 for r in window_rewards) / window, 4)
            metrics.append({"hand": i + 1, "avg_reward": avg_reward, "win_rate": win_rate})
            print(f"Hand {i+1}: avg_reward={avg_reward}, win_rate={win_rate}")

    return Q_table, metrics


Q_table, metrics = Q_learning_with_metrics(
    num_hands=10000,
    gamma=0.99,
    epsilon=1,
    decay_rate=0.9999,
    window=500
)

with open("metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["hand", "avg_reward", "win_rate"])
    writer.writeheader()
    writer.writerows(metrics)

print("\nMetrics saved to metrics.csv")
print(f"\n{'hand':<10} {'avg_reward':<14} {'win_rate':<10}")
print("-" * 35)
for m in metrics:
    print(f"{m['hand']:<10} {m['avg_reward']:<14} {m['win_rate']:<10}")