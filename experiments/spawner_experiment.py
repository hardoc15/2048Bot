"""Greedy vs. villain, random, and helper tile spawners.

Usage:
    python3 -m experiments.spawner_experiment                  # 100 games per condition
    python3 -m experiments.spawner_experiment --games 20 --depths 1 2

Every condition uses the same seeds (so the same two starting tiles). The
random condition reproduces the normal game exactly. Results are written to
experiments/results/greedy_vs_spawners.csv. Helper games run long and are
the slow part (several seconds per game at depth 2).
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

from agents.greedy import GreedyAgent
from spawners import HelperSpawner, RandomSpawner, VillainSpawner, play_game_with_spawner

DEFAULT_CSV = os.path.join("experiments", "results", "greedy_vs_spawners.csv")
FIELDS = ["spawner", "depth", "seed", "score", "max_tile", "moves", "capped", "ms_per_spawn"]


def conditions(depths):
    yield "random", 0, RandomSpawner()
    for depth in depths:
        yield "villain", depth, VillainSpawner(depth=depth)
        yield "helper", depth, HelperSpawner(depth=depth)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Greedy against each tile spawner.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-moves", type=int, default=10000, help="cap on moves per game")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()

    seeds = range(args.start_seed, args.start_seed + args.games)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for name, depth, spawner in conditions(args.depths):
            start = time.perf_counter()
            scores = []
            for seed in seeds:
                r = play_game_with_spawner(GreedyAgent(), spawner, seed=seed, max_moves=args.max_moves)
                scores.append(r.score)
                writer.writerow({
                    "spawner": name, "depth": depth, "seed": seed, "score": r.score,
                    "max_tile": r.max_tile, "moves": r.moves, "capped": r.capped,
                    "ms_per_spawn": f"{1000 * r.spawner_time / max(r.moves, 1):.4f}",
                })
                f.flush()
            label = name if name == "random" else f"{name} (depth {depth})"
            print(f"{label:<18} avg score {statistics.mean(scores):>8,.0f}   "
                  f"{time.perf_counter() - start:6.0f}s", flush=True)
    print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
