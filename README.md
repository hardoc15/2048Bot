# 2048Bot: Comparing Decision-Making Agents in 2048

An empirical study of how different AI decision-making strategies perform in the puzzle game **2048**, and whether the gains from more sophisticated search are worth their computational cost.

Built as a final project for **COMP 560: Artificial Intelligence**.

> **Status:** In development. The agents and experiments described below are being implemented. No performance results are reported yet. They will be added once experiments have been run and checked.

---

## Research Question

> How do random selection, greedy heuristics, adversarial search, and probabilistic search compare in 2048, and is the better performance of the more sophisticated methods worth the extra computation?

2048 is a useful AI testbed. On each turn the agent picks one of four directions, and the environment then places a new tile at random. Good play means balancing immediate rewards against long-term board structure while dealing with that uncertainty.

## Agents

| Agent | Strategy | AI concept |
| --- | --- | --- |
| **Random** | Picks a uniformly random legal move. | Baseline |
| **Greedy** | Scores the board after each legal move and takes the best immediate result. | Heuristic evaluation |
| **Minimax** | Searches ahead and treats each tile spawn as a worst-case adversarial choice. | Adversarial search |
| **Expectimax** | Searches ahead and weights each tile spawn by its actual probability. | Probabilistic reasoning / decision theory |

To keep the comparison fair, the Greedy, Minimax, and Expectimax agents share the same board evaluation function wherever possible.

### Evaluation Heuristic

The board heuristic combines several features:

- **Empty cells:** prefer boards with more open space.
- **Monotonicity:** reward rows and columns whose tile values are ordered.
- **Smoothness:** reward neighboring tiles with similar values.
- **Corner placement:** reward keeping the largest tile in a corner.

The heuristic is implemented in [`agents/heuristic.py`](agents/heuristic.py). Tile values are compared on a log2 scale (2 → 1, 2048 → 11). The board score is a weighted sum:

| Feature | Definition | Weight |
| --- | --- | ---: |
| Empty cells | Number of empty cells | 1.0 |
| Monotonicity | Minus, for each row and column, the smaller of its "breaks increasing" and "breaks decreasing" totals | 1.0 |
| Smoothness | Minus the sum of log2 differences between adjacent non-empty tiles | 0.5 |
| Corner | log2 of the largest tile if it sits in a corner, else 0 | 3.0 |

The weights come from a grid search with the Greedy agent (81 combinations, 60 games each on seeds 1000–1059) and were then checked on 100 held-out games (seeds 5000–5099). Smoothness had the largest effect. The other agents can import `evaluate` and `HeuristicWeights` directly.

### Greedy Agent

[`agents/greedy.py`](agents/greedy.py) simulates each legal move with `apply_move` (before any new tile appears), scores each resulting board with the shared heuristic, and plays the highest-scoring one. It has no lookahead. Ties go to the first move in the order up, down, left, right, so the agent is deterministic.

### Modeling Randomness

After every valid move in standard 2048, a new tile appears in a uniformly random empty cell. It is a **2 with 90% probability** and a **4 with 10% probability**.

- **Expectimax** uses these probabilities directly (chance nodes).
- **Minimax** treats the spawn as a worst-case opponent (min nodes).

This is a deliberate modeling comparison. 2048 has no real opponent; the question is whether pessimistic or probability-aware planning works better.

## Experiments

Each agent plays **at least 100 games** under identical rules with comparable random seeds.

| Metric | What it measures |
| --- | --- |
| Average score | Overall performance |
| Highest tile reached | Progress toward 2048 and beyond |
| 2048 success rate | Fraction of games that reach a 2048 tile |
| Time per move / total runtime | Computational cost |

Minimax and Expectimax are first compared at the same search depth. Depth is then varied to measure how lookahead trades off against runtime.

**Questions we want to answer:**

1. How much does a heuristic improve on random play?
2. Does lookahead improve on a greedy strategy?
3. Does probability-aware Expectimax beat worst-case Minimax?
4. How does deeper search affect performance and computation time?

## Results

*Preliminary: only the Greedy agent is done so far.* Full comparison tables will be added once all four agents are finished.

**Greedy, 100 games (seeds 0–99):**

| Metric | Value |
| --- | ---: |
| Average score | 11,930 |
| Median score | 11,812 |
| 2048 success rate | 12% |
| Reached 1024 or higher | 53% |
| Average time per move | ~0.08 ms |

Reproduce with:

```bash
python3 -m experiments.evaluate --agent greedy --games 100 --csv experiments/results/greedy.csv
```

Timing depends on the machine; scores are deterministic for a given seed.

### Results notebook and figures

[`experiments/results.ipynb`](experiments/results.ipynb) runs the Greedy agent on 100 seeded games and produces its tables and figures for the slides and paper (saved to `experiments/figures/`): a summary table, tile-reach curve, score distribution, highest-tile distribution, heuristic ablation, weight-sensitivity heatmap, and a single-game demo.

```bash
pip install -r requirements.txt
jupyter notebook experiments/results.ipynb   # then Kernel → Restart & Run All
```

## Repository Structure

```text
2048Bot/
├── README.md
├── requirements.txt      # Python dependencies (none yet; standard library only)
├── game/                 # 2048 engine (on main)
│   ├── board.py          #   Pure board functions: moves, merges, spawns, game over
│   ├── game.py           #   Game class: board + score + seeded random spawns
│   ├── runner.py         #   play_game / play_games: run an agent and record results
│   └── __main__.py       #   Play 2048 yourself in the terminal
├── agents/
│   ├── base.py           # Agent interface every algorithm implements
│   ├── heuristic.py      # Shared board evaluation (Greedy, Minimax, Expectimax)
│   └── greedy.py         # Greedy agent
├── tests/                # Unit tests for the engine and agents
├── experiments/
│   ├── evaluate.py       # Run N seeded games for an agent and report metrics
│   └── results/          # Per-game CSV output
├── slides/               # (planned) Presentation slides and demo
└── paper/                # (planned) LaTeX source and compiled PDF
```

### Branches

`main` holds the shared 2048 engine. Each agent is developed on its own branch (one per algorithm: Random, Greedy, Minimax, Expectimax) and merged back for the final experiments.

## Getting Started

**Requirements:** Python 3.8+ (no third-party packages)

```bash
git clone https://github.com/hardoc15/2048Bot.git
cd 2048Bot
```

**Play 2048 yourself** (w/a/s/d, then Enter; q quits):

```bash
python3 -m game
python3 -m game --seed 42   # same tile spawns every time
```

**Run the tests:**

```bash
python3 -m unittest discover -s tests
```

## Writing an Agent

Every agent subclasses `Agent` and implements `choose_move(board)`, which returns a legal `Direction`:

```python
from agents.base import Agent
from game import Direction, legal_moves

class MyAgent(Agent):
    name = "my-agent"

    def choose_move(self, board):
        return legal_moves(board)[0]
```

Run it through full games:

```python
from game import play_game, play_games

result = play_game(MyAgent(), seed=0)
print(result.score, result.max_tile, result.reached_2048, result.time_per_move)

results = play_games(MyAgent(), seeds=range(100))  # same seeds for every agent
```

### Engine API for search

A board is an immutable tuple of rows (`0` = empty). Every function returns a new board, so agents can explore hypothetical moves safely.

| Function | Returns | Useful for |
| --- | --- | --- |
| `legal_moves(board)` | Directions that change the board | All agents |
| `apply_move(board, direction)` | `(new_board, score_gained, changed)`, with **no** tile spawn | Greedy (evaluating each move), search |
| `spawn_outcomes(board)` | `[(probability, board), ...]` for every possible spawn | Expectimax (weighted), Minimax (worst case) |
| `empty_cells(board)` | `[(row, col), ...]` | Heuristics |
| `max_tile(board)` / `is_game_over(board)` | Largest tile / no legal moves | Heuristics, terminal checks |

Commands for reproducing the experiments, including the seeds and search depths used, will be added once the agents are complete.

## Paper and Presentation

- **Paper:** `paper/` (LaTeX source and compiled PDF, in an AI conference style)
- **Slides and demo:** `slides/`

## Team

| Member | Primary responsibility |
| --- | --- |
| *TBD* | 2048 game engine and Random agent |
| *TBD* | Greedy agent and heuristic evaluation |
| *TBD* | Minimax agent |
| *TBD* | Expectimax agent |

All members contribute to evaluation, code review, the paper, and the presentation.

## Acknowledgments

External code, libraries, and references used in this project will be credited here.
