"""Learning curve: the TD bot at each training checkpoint vs. random tiles and the villain.

Usage:
    python3 -m experiments.td_learning_curve
    python3 -m experiments.td_learning_curve --games 100 --villain-depth 2

Each checkpoint plays the same seeded games (seeds 0..games-1) as every other
agent in the project, once with normal random tiles and once against the
full-force villain. Results go to experiments/results/td_learning_curve.csv.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time

from spawners import RandomSpawner, VillainSpawner, play_game_with_spawner
from td.agent import TDAgent
from td.train import CHECKPOINT_DIR

DEFAULT_CSV = os.path.join("experiments", "results", "td_learning_curve.csv")
FIELDS = ["games_trained", "spawner", "seed", "score", "max_tile", "moves", "capped"]


def available_checkpoints(directory=CHECKPOINT_DIR):
    found = []
    for path in glob.glob(os.path.join(directory, "ntuple_*.npz")):
        m = re.search(r"ntuple_(\d+)\.npz$", path)
        if m:
            found.append(int(m.group(1)))
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TD checkpoints vs. random tiles and the villain.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--villain-depth", type=int, default=2)
    parser.add_argument("--max-moves", type=int, default=20000)
    parser.add_argument("--checkpoints", type=int, nargs="*", help="default: every saved checkpoint")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()

    checkpoints = args.checkpoints or available_checkpoints()
    seeds = range(args.games)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for games in checkpoints:
            agent = TDAgent.load(games)
            for name, make in [("random", RandomSpawner), ("villain", lambda: VillainSpawner(depth=args.villain_depth))]:
                start, total = time.perf_counter(), 0
                spawner = make()
                for seed in seeds:
                    r = play_game_with_spawner(agent, spawner, seed=seed, max_moves=args.max_moves)
                    total += r.score
                    writer.writerow({"games_trained": games, "spawner": name, "seed": seed, "score": r.score,
                                     "max_tile": r.max_tile, "moves": r.moves, "capped": r.capped})
                f.flush()
                print(f"{games:>9,} games trained  vs {name:<7}  avg score {total / len(seeds):>9,.0f}"
                      f"  {time.perf_counter() - start:6.0f}s", flush=True)
    print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
