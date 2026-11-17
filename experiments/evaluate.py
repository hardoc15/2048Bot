"""Run an agent for many games and report the project's metrics.

Usage:
    python3 -m experiments.evaluate --agent greedy --games 100
    python3 -m experiments.evaluate --agent greedy --games 100 --csv experiments/results/greedy.csv

Games use seeds start-seed, start-seed+1, ..., so every agent can be run on
exactly the same games. To add an agent, register it in AGENTS below.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import Counter

from agents.greedy import GreedyAgent
from game.runner import play_games

AGENTS = {
    "greedy": GreedyAgent,
}


def summarize(name, results) -> str:
    scores = [r.score for r in results]
    tiles = Counter(r.max_tile for r in results)
    n = len(results)
    lines = [
        f"Agent: {name}   Games: {n}",
        f"  Average score:      {statistics.mean(scores):,.0f}",
        f"  Median score:       {statistics.median(scores):,.0f}",
        f"  Best score:         {max(scores):,}",
        f"  2048 success rate:  {sum(r.reached_2048 for r in results) / n:.1%}",
        f"  Avg time per move:  {statistics.mean(r.time_per_move for r in results) * 1000:.3f} ms",
        f"  Total agent time:   {sum(r.total_time for r in results):.2f} s",
        "  Highest tile reached:",
    ]
    for tile in sorted(tiles, reverse=True):
        lines.append(f"    {tile:>5}: {tiles[tile]:>4} games ({tiles[tile] / n:.0%})")
    return "\n".join(lines)


def write_csv(path, name, results) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "seed", "score", "max_tile", "moves", "total_time_s", "time_per_move_s"])
        for r in results:
            writer.writerow([name, r.seed, r.score, r.max_tile, r.moves,
                             f"{r.total_time:.6f}", f"{r.time_per_move:.8f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a 2048 agent.")
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--csv", help="optional path to save per-game results")
    args = parser.parse_args()

    agent = AGENTS[args.agent]()
    seeds = range(args.start_seed, args.start_seed + args.games)
    results = play_games(agent, seeds)
    print(summarize(args.agent, results))
    if args.csv:
        write_csv(args.csv, args.agent, results)
        print(f"\nPer-game results saved to {args.csv}")


if __name__ == "__main__":
    main()
