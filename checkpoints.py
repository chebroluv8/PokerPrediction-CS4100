from Q_learning import encode_state
from poker_rlcard import LimitHoldEmEnv
import numpy as np
import random
import pickle
import os

def Q_learning_with_checkpoints(num_hands=10000, gamma=0.99, epsilon=1, decay_rate=0.9999, checkpoint_every=2000):
    env = LimitHoldEmEnv()
    Q_table = {}
    num_updates = {}
    reward_list = []

    os.makedirs("checkpoints", exist_ok=True)

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

        if (i + 1) % checkpoint_every == 0:
            avg_reward = round(np.mean(reward_list[-(checkpoint_every):]), 4)
            path = f"checkpoints/Q_table_hand_{i+1}.pickle"
            with open(path, "wb") as f:
                pickle.dump(Q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Checkpoint saved at hand {i+1} | avg_reward={avg_reward} | file: {path}")

    print("\nTraining complete.")
    return Q_table


Q_table = Q_learning_with_checkpoints(
    num_hands=10000,
    gamma=0.99,
    epsilon=1,
    decay_rate=0.9999,
    checkpoint_every=2000
)