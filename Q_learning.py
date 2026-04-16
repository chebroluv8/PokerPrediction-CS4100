from poker_rlcard import LimitHoldEmEnv
import numpy as np
import sys
import time
import pickle
from tqdm import tqdm
import random
import csv

env = LimitHoldEmEnv()

def encode_state(state, player_id):
    """
    Parameters: state (dictionary), player_id (int)
    Does: Encodes state into a set with format (street, hand_bucket, my_chips, opponent_chips, raises_so_far).
    An explanation of each of these parameters is documented in the data_definition.pdf file. 
    Returns: Q_table (dict): Dictionary containing the Q-values for each state-action pair
    """
    raw = state["raw_obs"]

    street = env.get_street(raw["public_cards"])
    hand_bucket = env.get_hand_strength_bucket(raw["hand"])
    my_chips = raw["my_chips"]
    opponent_chips = raw["all_chips"][1 - player_id]
    raises_so_far = sum(raw["raise_nums"])

    return (street, hand_bucket, my_chips, opponent_chips, raises_so_far)

def Q_learning(num_hands = 1000, gamma = 0.9, epsilon = 1, decay_rate = 0.999, window = 500):
    """
    Parameters:
    - num_hands (int): Number of hands to train on
    - gamma (float): Discount factor
    - epsilon (float): Exploration rate
    - decay_rate (float): Rate at which epsilon decays after each episode
    - window (int): Number of hands to consider at a time when calculating rolling average

    Does: Run Q-learning algorithm for a specified number of hands
    Returns: Q_table (dict) containing the Q-values for each state-action pair 
    """
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

            # Check if current state in Q-table
            if current_state not in Q_table:
                Q_table[current_state] = np.zeros(env.num_actions)
                num_updates[current_state] = np.zeros(env.num_actions)
            
            # Pick action with epsilon greedy policy
            if np.random.random() <= epsilon:
                action = random.choice(legal_actions)
            else:
                action = np.argmax(Q_table[current_state]) 

            # Get a new observation, reward, done using the env.step() function
            next_state_raw, next_player_id, done = env.step(action)

            # Only learn from Player 0 -> Check if game is done to get reward and update Q-table and num_updates regardless
            if player_id == 0:
                if done:
                    reward = float(env.get_payoffs()[player_id])
                    total_reward += reward
                    next_state = None

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

        # Add episode reward and decay epsilon at end of episode
        reward_list.append(total_reward)
        epsilon *= decay_rate

        # Track rolling average of reward and win rate
        if (i + 1) % window == 0:
            window_rewards = reward_list[-window:]
            avg_reward = round(np.mean(window_rewards), 4)
            win_rate = round(sum(r > 0 for r in window_rewards) / window, 4)
            metrics.append({"hand": i + 1, "avg_reward": avg_reward, "win_rate": win_rate})
    
    return Q_table, metrics
