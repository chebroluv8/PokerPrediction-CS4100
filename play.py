"""
Interactive Terminal UI — Watch Q-Learning Agent vs Random Agent
Run: python play.py
"""

import pickle
import random
import numpy as np
import time
import os
from poker_rlcard import LimitHoldEmEnv
from Q_learning import encode_state

STREETS      = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}
HAND_BUCKETS = {0: "Weak", 1: "Mediocre", 2: "Strong"}
ACTION_NAMES = {0: "Call", 1: "Raise", 2: "Fold", 3: "Check"}
ACTION_EMOJI = {0: "📞", 1: "📈", 2: "🏳️ ", 3: "✅"}
WIDTH        = 60

def clear(): os.system("clear" if os.name == "posix" else "cls")

def box(title, lines, width=WIDTH):
    print("╔" + "═" * (width - 2) + "╗")
    t = f"  {title}  "
    print("║" + t.center(width - 2) + "║")
    print("╠" + "═" * (width - 2) + "╣")
    for line in lines:
        print("║  " + str(line).ljust(width - 4) + "║")
    print("╚" + "═" * (width - 2) + "╝")

def separator(label=""):
    if label:
        print(f"\n{'─' * 20} {label} {'─' * 20}\n")
    else:
        print("─" * WIDTH)

def fmt_card(card):
    suit_map = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
    if len(card) == 2:
        rank, suit = card[1], card[0]
    else:
        rank, suit = card[1:], card[0]
    return f"[{rank}{suit_map.get(suit, suit)}]"

def fmt_hand(cards):
    return "  ".join(fmt_card(c) for c in cards) if cards else "(none)"

def wait(msg="Press Enter to continue..."):
    input(f"\n  ⏎  {msg}")

def pick_model():
    default = "pickle_files/Q_table_1000000_0.95_0.999.pickle"
    print("\n" + "═" * WIDTH)
    print("  🃏  Q-LEARNING POKER AGENT  —  Interactive Viewer")
    print("═" * WIDTH)
    print(f"\n  Default model: Q_table_1000000_0.95_0.999  (best by win rate)")
    choice = input("  Load default? [Y/n]: ").strip().lower()
    if choice in ("", "y"):
        return default
    path = input("  Enter pickle path: ").strip()
    return path

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_agent_action(Q_table, state, player_id, env):
    enc = encode_state(state, player_id)
    legal = list(state["legal_actions"].keys())
    if enc in Q_table:
        q_vals = Q_table[enc]
        # mask illegal actions
        masked = {a: q_vals[a] for a in legal}
        action = max(masked, key=masked.get)
        confidence = q_vals[action]
        known = True
    else:
        action = random.choice(legal)
        confidence = None
        known = False
    return action, confidence, known, enc

def play_hand(Q_table, hand_num):
    env = LimitHoldEmEnv()
    state, player_id = env.reset()
    done = False
    step = 0
    history = []  

    clear()
    print(f"\n{'═' * WIDTH}")
    print(f"  🃏  HAND #{hand_num}".center(WIDTH))
    print(f"{'═' * WIDTH}\n")

    while not done:
        raw = state["raw_obs"]
        legal = list(state["legal_actions"].keys())

        street_idx   = env.get_street(raw["public_cards"])
        hand_bucket  = env.get_hand_strength_bucket(raw["hand"])
        street_name  = STREETS[street_idx]
        bucket_name  = HAND_BUCKETS[hand_bucket]

        is_agent = (player_id == 0)
        actor    = "🤖 Q-Agent (P0)" if is_agent else "🎲 Random  (P1)"

        separator(street_name)
        info_lines = [
            f"Actor       : {actor}",
            f"Hand        : {fmt_hand(raw['hand'])}",
            f"Board       : {fmt_hand(raw['public_cards'])}",
            f"Hand Strength: {bucket_name}",
            f"Chips  P0/P1: {raw['all_chips'][0]} / {raw['all_chips'][1]}",
            f"Raises so far: {sum(raw['raise_nums'])}",
            f"Legal actions: {[ACTION_NAMES.get(a, a) for a in legal]}",
        ]
        box(f"Step {step + 1}", info_lines)

        if is_agent:
            action, conf, known, enc_state = get_agent_action(Q_table, state, player_id, env)
            q_line = f"Q-values: {[round(Q_table[enc_state][a], 3) if enc_state in Q_table else '?' for a in range(env.num_actions)]}" if known else "State unseen — random fallback"
            print(f"\n  🤖 Agent Decision:")
            print(f"     → {ACTION_EMOJI.get(action, '?')} {ACTION_NAMES.get(action, action)}", end="")
            print(f"  (confidence: {round(conf, 4)})" if conf is not None else "  (random)")
            print(f"     {q_line}")
            history.append(f"[{street_name}] Agent → {ACTION_NAMES.get(action, action)}")
        else:
            action = random.choice(legal)
            print(f"\n  🎲 Random Agent picks: {ACTION_EMOJI.get(action, '?')} {ACTION_NAMES.get(action, action)}")
            history.append(f"[{street_name}] Random → {ACTION_NAMES.get(action, action)}")

        wait()
        state, player_id, done = env.step(action)
        step += 1

    payoffs = env.get_payoffs()
    agent_reward = float(payoffs[0])
    winner = "🤖 Q-Agent wins!" if agent_reward > 0 else ("🎲 Random wins!" if agent_reward < 0 else "🤝 Tie!")

    separator("RESULT")
    result_lines = [
        f"Outcome     : {winner}",
        f"Agent reward: {agent_reward:+.2f}",
        "",
        "─ Action History ─",
    ] + history
    box("Hand Over", result_lines)
    print()

    return agent_reward

def run_session(Q_table):
    rewards = []
    hand_num = 1

    while True:
        reward = play_hand(Q_table, hand_num)
        rewards.append(reward)
        hand_num += 1

        wins    = sum(r > 0 for r in rewards)
        win_pct = wins / len(rewards) * 100
        avg_r   = np.mean(rewards)

        separator("SESSION STATS")
        print(f"  Hands played : {len(rewards)}")
        print(f"  Win rate     : {win_pct:.1f}%  ({wins}/{len(rewards)})")
        print(f"  Avg reward   : {avg_r:+.4f}")
        print()

        again = input("  Play another hand? [Y/n]: ").strip().lower()
        if again == "n":
            break

    print(f"\n  Final session: {len(rewards)} hands | Win rate: {win_pct:.1f}% | Avg reward: {avg_r:+.4f}")
    print("  Thanks for watching! 🃏\n")

if __name__ == "__main__":
    path = pick_model()
    print(f"\n  Loading model from {path} ...")
    Q_table = load_model(path)
    print(f"Loaded {len(Q_table)} states\n")
    wait("Press Enter to start the first hand...")
    run_session(Q_table)