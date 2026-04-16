# PokerPrediction-CS4100

A Q-learning agent trained to play **Limit Texas Hold'em** poker, built using the [RLCard](https://github.com/datamllab/rlcard) framework. This project explores how a tabular reinforcement learning approach performs in a partially observable card game environment through extensive hyperparameter experimentation.

---

## Overview

The agent learns a Q-table by playing thousands of hands against a random opponent. States are encoded as compact feature tuples and actions are selected via an epsilon-greedy policy. After training, the agent is evaluated against both a random agent and itself across various hyperparameter configurations.

---

## Project Structure

```
PokerPrediction-CS4100/
├── Q_learning.py                 # Core Q-learning training loop
├── poker_rlcard.py               # LimitHoldEmEnv wrapper + state encoding helpers
├── agent_comparison_analysis.py  # Hyperparameter sweep analysis & visualizations
├── data_definition.pdf           # State space feature definitions
├── metrics.csv                   # Training metrics (rolling avg reward & win rate)
├── results/                      # Saved Q-tables, eval CSVs, and plots
└── README.md
```

---

## State Representation

Each game state is encoded as a 5-tuple:

| Feature | Description |
|---|---|
| `street` | Current betting round (0=preflop, 1=flop, 2=turn, 3=river) |
| `hand_bucket` | Hand strength (0=weak, 1=mediocre, 2=strong) |
| `my_chips` | Agent's current chip count |
| `opponent_chips` | Opponent's current chip count |
| `raises_so_far` | Total raises across all streets |

See `data_definition.pdf` for full details.

---

## Setup

**Requirements**
```
rlcard
numpy
pandas
matplotlib
tqdm
```

**Install**
```bash
pip install rlcard numpy pandas matplotlib tqdm
```

---

## Usage

### Train the agent

```python
from Q_learning import Q_learning

Q_table, metrics = Q_learning(
    num_hands=100000,
    gamma=0.9,
    epsilon=1.0,
    decay_rate=0.9999
)
```

### Run the hyperparameter sweep analysis

```bash
python agent_comparison_analysis.py
```

This reads `experiment_results.csv`, computes improvements over a random baseline, and outputs rankings and plots to `results/`.

---

## Hyperparameters

| Parameter | Description |
|---|---|
| `num_hands` | Number of training episodes |
| `gamma` | Discount factor for future rewards |
| `epsilon` | Initial exploration rate (decays over time) |
| `decay_rate` | Multiplicative decay applied to epsilon each episode |
| `window` | Rolling window size for tracking training metrics |

Sweep experiments tested combinations of `num_hands` ∈ {5k, 10k, 25k, 100k, 1M}, `gamma` ∈ {0.7, 0.8, 0.9, 0.95}, and various decay rates. Top configurations achieved **win rates above 92%** against a random agent.

---

## Results

Trained Q-tables and evaluation outputs are saved in `results/`. Key artifacts include:

- `sweep_summary.csv` — aggregated train/eval metrics across all hyperparameter combinations
- `eval_curves_*.png` — win rate and reward curves over evaluation hands
- `comparison_table_*.png` — Q-agent vs random agent side-by-side comparisons
- `sweep_heatmaps.png` — heatmap of win rate across the hyperparameter grid
- `Q_table_*.pickle` — serialized Q-tables for top-performing configs

---

## Team

Built for **CS4100 (Artificial Intelligence)** at Northeastern University.

Contributors: Naisha, Vaishnavi, Anjali, Madhav