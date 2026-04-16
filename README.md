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
├── experiments.py                # Runs full hyperparameter sweep, saves Q-tables & eval metrics
├── eval_and_visualize.py         # Evaluation functions and plot generation
├── agent_comparison_analysis.py  # Sweep analysis & visualizations
├── checkpoints.py                # Saves Q-table checkpoints during training
├── play.py                       # Play against the trained agent
├── env_setup.ipynb               # Environment setup and exploration notebook
├── data_definition.pdf           # State space feature definitions
├── experiment_results.csv        # Aggregated results across all sweep experiments
├── analysis_results.csv          # Q-agent vs random agent comparison summary
├── pickle_files/                 # Trained Q-tables for all hyperparameter combinations
├── training_metrics/             # Per-experiment training metric CSVs
└── results/                      # Evaluation CSVs and plots
    ├── eval_curves_*.png
    ├── comparison_table_*.png
    ├── situation_heatmap_*.png
    ├── improvement_by_train_hands.png
    ├── top10_win_rate_vs_random.png
    └── eval_metrics_*.csv
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

### Run the full hyperparameter sweep

```bash
python experiments.py
```

Trains all combinations of gamma, decay rate, and training hands, saving Q-tables to `pickle_files/`, training metrics to `training_metrics/`, and evaluation results to `experiment_results.csv`.

### Run the comparison analysis

```bash
python agent_comparison_analysis.py
```

Reads `experiment_results.csv`, ranks models against a random baseline, and outputs plots to `results/`.

### Play against the agent

```bash
python play.py
```

---

## Hyperparameters

| Parameter | Description |
|---|---|
| `num_hands` | Number of training episodes |
| `gamma` | Discount factor for future rewards |
| `epsilon` | Initial exploration rate (decays over time) |
| `decay_rate` | Multiplicative decay applied to epsilon each episode |
| `window` | Rolling window size for tracking training metrics |

Sweep experiments tested combinations of `num_hands` ∈ {10k, 100k, 1M}, `gamma` ∈ {0.7, 0.8, 0.9, 0.95}, and decay rates ∈ {0.999, 0.9999, 0.99999, 0.999995}. Top configurations achieved **win rates above 92%** against a random agent.

---

## Team

Built for **CS4100 (Artificial Intelligence)** at Northeastern University.

Contributors: Naisha, Anjali, Vaishnavi, Madhav