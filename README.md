# 2048Bot: Comparing Decision-Making Agents in 2048

An empirical study of how different AI decision-making strategies perform in the puzzle game **2048**, and whether the gains from more sophisticated search are worth their computational cost.

Built as a final project for **COMP 560: Artificial Intelligence**.

> **Status:** All five agents (Random, Greedy, Minimax, Expectimax, and our trained bot Tuplet) are implemented and evaluated against all three tile spawners. All tables and figures are in [`experiments/results.ipynb`](experiments/results.ipynb). The paper is in progress.

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

### What If the Tiles Fight Back? (Villain and Helper Spawners)

To test those modeling assumptions directly, we replace 2048's random tile placement with an AI **spawner** ([`spawners/`](spawners/)) that chooses where every new tile goes:

- **Villain:** places each tile where it does the most damage to the player.
- **Helper:** places each tile where it helps the player most.
- **Random:** the standard rule. It reproduces a normal seeded game exactly.

The villain and helper use adversarial search over the alternating spawner/player game tree. They look `depth` player turns ahead, model the player as greedy, and score positions with the shared heuristic. A spawn that leaves the player with no legal move ends the game, so the villain always takes one when it can. The villain and helper may choose a 2 or a 4 freely, unlike the normal 90/10 rule.

This turns Minimax's assumption (an adversarial spawn) and Expectimax's assumption (a random spawn) into real opponents, so each agent can be measured under both.

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

### All five agents vs. every tile spawner

Normal game and villain: seeds 0-99 (100 games each). Helper: seeds 0-19 with a 5,000-move cap (8 of Expectimax's 20 helper games hit the cap, so its helper average is a lower bound). Minimax and Expectimax search 2 moves ahead.

| Agent | Normal game: avg score | Reached 2048 | Vs. villain: avg score | Moves survived vs. villain | With helper: avg score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 1,085 | 0% | 344 | 46 | 2,993 |
| Greedy | 11,930 | 12% | 356 | 45 | 33,484 |
| Minimax | 17,867 | 34% | **1,217** | **99** | 173,176 |
| Expectimax | 32,290 | 74% | 791 | 73 | 189,876 |
| Tuplet (trained) | **40,585** | **83%** | 686 | 68 | 103,115 |

- **Normal game:** Tuplet > Expectimax > Minimax > Greedy > Random. Expectimax, which assumes random tiles, is the best search agent when tiles really are random.
- **Against the villain the ranking flips:** Minimax, which plans for hostile tiles, survives longest. Every non-random agent loses 93% or more of its normal-game score.
- **Sample size:** against the villain and helper, games are deterministic for every agent except Random, so the 100 villain games contain only 9-17 distinct outcomes per agent (see the notebook's "Distinct games" column).

Reproduce (about 40 minutes on 8 cores for the normal and villain runs; the helper run is slower per game):

```bash
python3 -m experiments.all_agents --games 100 --spawners random villain --csv experiments/results/all_agents_main.csv
python3 -m experiments.all_agents --games 20 --spawners helper --chunk 2 --max-moves 5000 --csv experiments/results/all_agents_helper.csv
# then combine the two files into experiments/results/all_agents.csv (the notebook reads that file)
```

### Greedy agent

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

### Greedy vs. villain, random, and helper spawners

Same agent, same 100 starting positions (seeds 0–99); only the tile placement changes:

| Spawner | Average score | Highest tile | Distinct games | Time per spawn |
| --- | ---: | --- | ---: | ---: |
| Villain, depth 2 | 356 | 64 at most | 10 | ~7 ms |
| Villain, depth 1 | 765 | 128 at most | 12 | ~0.5 ms |
| Random (normal game) | 11,930 | 2048 in 12% | 99 | — |
| Helper, depth 1 | 11,780 | always 1024 | 2 | ~0.6 ms |
| Helper, depth 2 | 33,484 | 2048 every game | 2 | ~7 ms |

Findings so far:

- **Tile placement matters enormously.** With the agent unchanged, the average score ranges about 90-fold, from 356 to 33,484.
- **Sabotage is easy; help takes planning.** Even a one-move-deep villain wrecks the game. A one-move-deep helper doesn't help at all: it mostly hands out 4s that look good right now and jams the board at 1024. Looking two moves ahead, the helper reaches 2048 every time.
- **The spawner removes the luck.** After the two starting tiles, a villain or helper game has no randomness, and different starts quickly converge to the same few games (see "Distinct games"). That column, not 100, is the effective sample size for those rows.
- **Caveat:** the villain and helper model the player with the same heuristic Greedy uses, so against Greedy they know their opponent exactly.

Reproduce (about 15 minutes, mostly the depth-2 helper games):

```bash
python3 -m experiments.spawner_experiment --games 100 --depths 1 2
```

Play against the villain or helper yourself (w/a/s/d, then Enter):

```bash
python3 -m spawners --spawner villain
python3 -m spawners --spawner helper
```

### Tuplet: our trained bot, and the villain

**Tuplet** (the TD-learning bot in [`td/`](td/), named after the n-tuple network it is built on) learns 2048 from self-play with temporal-difference learning on an n-tuple network (method from Szubert & Jaskowski, 2014, implemented from scratch). The network is 5 four-cell patterns (two rows, three 2×2 squares), each shared across the board's 8 rotations/reflections. It trains only on normal random tiles. A fast 64-bit board compiled with Numba (checked move-for-move against the shared engine) trains 1,000,000 games in about 23 minutes on a laptop.

Each saved checkpoint plays the same 100 seeded games, with random tiles and against the full-force villain (depth 2):

| Games trained | Normal game: avg score | Normal game: reached 2048 | Vs. villain: avg score |
| ---: | ---: | ---: | ---: |
| 0 (untrained) | 3,316 | 0% | 278 |
| 1,000 | 13,213 | 7% | 927 |
| 10,000 | 27,978 | 62% | 584 |
| 30,000 | 40,462 | 83% | 943 |
| 100,000 | 40,643 | 85% | 903 |
| 1,000,000 | 40,585 | 83% | 686 |

- **It learns to play 2048 well:** 83% of games reach 2048 (Greedy: 12%), and 43% of late training games reach 4096. Normal-game performance plateaus after about 30,000 games with this network size.
- **It does not learn to survive the villain:** from 100 to 1,000,000 training games its normal-game score grows 6.9×, but against the villain only 1.2×, with no consistent trend. The villain predicts its opponent with the hand-written heuristic, not the bot's learned values, and still wins.
- **Caveat:** bot and villain are both deterministic, so each checkpoint's 100 villain games contain only 10–18 distinct outcomes.

Reproduce:

```bash
python3 -m td.train --games 1000000 --seed 0        # ~23 minutes; writes td/checkpoints/
python3 -m experiments.td_learning_curve            # every checkpoint vs. random tiles and the villain
```

### Results notebook and figures

[`experiments/results.ipynb`](experiments/results.ipynb) collects every result in the project and saves its figures to `experiments/figures/` for the slides and paper:

| Figure | Shows |
| --- | --- |
| `fig1_agents_normal` | Average score and 2048 rate for all five agents, normal game |
| `fig2_agents_tile_reach` | Share of games reaching each tile, per agent |
| `fig3_agents_cost` | Score vs. computation time per move |
| `fig4_agents_vs_spawners` | Every agent against the villain, normal game, and helper |
| `fig5_villain_survival` | Moves survived against the full-force villain |
| `fig6_greedy_ablation` | Greedy with one heuristic feature removed at a time |
| `fig7_greedy_weight_sensitivity` | Greedy score across heuristic weights |
| `fig8_spawner_depth` | Spawner search depth: sabotage is easy, help takes planning |
| `fig9_spawner_final_boards` | The same start played against the villain, random tiles, and the helper |
| `fig10_td_training_curve` | Tuplet learning to reach 2048 and 4096 |
| `fig11_td_vs_villain_curve` | Tuplet's learning curve in the normal game vs. against the villain |

The Minimax branch's own depth and weight study is in [`experiments/minimax.ipynb`](experiments/minimax.ipynb), and the original Random baseline analysis is in [`experiments/results/random_baseline.ipynb`](experiments/results/random_baseline.ipynb).

```bash
pip install -r requirements.txt
jupyter notebook experiments/results.ipynb   # then Kernel, Restart & Run All
```

## Repository Structure

```text
2048Bot/
├── README.md
├── requirements.txt      # numpy + numba (Tuplet); matplotlib, pandas, jupyter (notebook, slides)
├── game/                 # 2048 engine (on main)
│   ├── board.py          #   Pure board functions: moves, merges, spawns, game over
│   ├── game.py           #   Game class: board + score + seeded random spawns
│   ├── runner.py         #   play_game / play_games: run an agent and record results
│   └── __main__.py       #   Play 2048 yourself in the terminal
├── agents/
│   ├── base.py           # Agent interface every algorithm implements
│   ├── heuristic.py      # Shared board evaluation (Greedy, Minimax, Expectimax)
│   └── greedy.py         # Greedy agent
├── spawners/             # Who places the tiles: random, villain, or helper
│   ├── base.py           #   Spawner interface + RandomSpawner (the normal game)
│   ├── adversarial.py    #   VillainSpawner / HelperSpawner (adversarial search)
│   ├── runner.py         #   play_game_with_spawner
│   └── __main__.py       #   Play against the villain or helper yourself
├── td/                   # TD-learning bot (n-tuple network)
│   ├── bitboard.py       #   Fast 64-bit board (Numba), checked move-for-move against game/
│   ├── ntuple.py         #   N-tuple network + TD(0) afterstate learning
│   ├── train.py          #   Training with checkpoints
│   ├── agent.py          #   TDAgent: plays from a saved checkpoint
│   └── checkpoints/      #   Saved networks + training log
├── slides/
│   ├── src/              #   Slide template + browser game/villain (game.js)
│   ├── build.py          #   Builds the self-contained presentation from results
│   └── presentation.html #   Open in any browser; no internet needed
├── tests/                # Unit tests for the engine, agents, spawners, Tuplet, and browser game
├── experiments/
│   ├── evaluate.py       # Run N seeded games for an agent and report metrics
│   ├── spawner_experiment.py  # Greedy vs. villain / random / helper
│   ├── td_learning_curve.py   # TD checkpoints vs. random tiles and the villain
│   ├── results.ipynb     # All tables and figures
│   ├── figures/          # PNGs for slides and paper
│   └── results/          # Per-game CSV output
├── slides/               # (planned) Presentation slides and demo
└── paper/                # (planned) LaTeX source and compiled PDF
```

### Branches

`main` holds the shared 2048 engine. Each agent is developed on its own branch (one per algorithm: Random, Greedy, Minimax, Expectimax) and merged back for the final experiments.

## Getting Started

**Requirements:** Python 3.8+ for the engine and agents. Install the packages in `requirements.txt` to run the notebooks.

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

### Run Minimax

`MinimaxAgent` uses the shared board heuristic and alpha–beta pruning. A depth of two covers two player moves and their tile spawns. Search assumes a worst-case spawn, while the game uses the standard random-spawn rules.

```python
from agents.minimax import MinimaxAgent
from game import play_game

print(play_game(MinimaxAgent(depth=2), seed=0))
```

Open [the executed notebook](experiments/minimax.ipynb) to inspect the heuristic sweep, held-out score tables, and plots. The sweep uses depth one on seeds 2000–2023. Evaluation uses seeds 3000–3099. The shared weights stay fixed for the depth-one versus depth-two comparison.

With shared weights, depth two averaged 17,591 points and reached 2048 in 27 of 100 games. Depth one averaged 8,371 points and reached 2048 in none of those games. The selected weight adjustment raised the depth-one mean to 9,020 points, with a paired 95% bootstrap interval that includes zero for the gain.

Run these commands from the repository root to regenerate the notebook outputs, [per-game scores](experiments/results/minimax_games.csv), [summary table](experiments/results/minimax_summary.csv), and [figure](experiments/figures/minimax_comparison.png). The depth-two run can take several minutes, and measured times vary with hardware.

```bash
python3 -m pip install -r requirements.txt
python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 experiments/minimax.ipynb
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

### Running Expectimax

Expectimax uses the shared Greedy board heuristic. Depth counts player moves,
including the random tile spawn after each move. At depth 2, the agent considers
every possible tile spawn after its first move, picks the best continuation for
each outcome, and evaluates the boards after the second move's spawn. It uses
the engine's exact spawn probabilities and breaks ties in up, down, left, right
order.

```python
from agents.expectimax import ExpectimaxAgent
from game import play_games

for result in play_games(ExpectimaxAgent(depth=2), seeds=[0, 1, 2]):
    print(result.seed, result.score, result.max_tile, result.reached_2048,
          result.moves, result.time_per_move)
```

Commands for reproducing the experiments, including the seeds and search depths used, will be added once the agents are complete.

## Presentation

[`slides/presentation.html`](slides/presentation.html) is a single self-contained file: open it in any browser, with no internet needed. A live game against the full-force villain runs in the corner of every slide so someone can play while we present. [`slides/presentation.pdf`](slides/presentation.pdf) is a static backup.

| Key | Action |
| --- | --- |
| Arrow keys | Play the game |
| D / A (or Page Down / Page Up) | Next / previous slide |
| B | Enlarge the game panel (slide content narrows to make room) |
| P | Open a full-screen board in a second window (for a laptop screen with an extended display) |
| G | Switch between the demo strength set with `--gentle` (default 1.0, full force) and full force |
| R / Shift+R | New game / reset the scoreboard |
| F | Full screen |
| H | Hide / show the game panel |
| ? | Show the key list |
| 1–5 | Jump to a slide |

Every number on the slides is read from the result CSVs at build time. Rebuild after re-running experiments:

```bash
python3 -m slides.build --team "Name 1, Name 2, Name 3, Name 4"
```

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
