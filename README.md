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

The final weights and exact definitions will be documented here and in the paper.

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

*Results will appear here after the experiments are complete.* This section will include summary tables, score and tile distributions, and runtime comparisons.

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
│   └── base.py           # Agent interface every algorithm implements
├── tests/                # Unit tests for the engine
├── experiments/          # (planned) Evaluation scripts and results/figures
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

### Reproduce the random baseline figure

From the repository root, run `python -m experiments.random_baseline`.
This plays 100 games with game seeds 0–99 and a separate agent RNG seeded
with 0 at the start of the batch. It regenerates the
[score and highest-tile distributions](slides/random-baseline.svg).

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
