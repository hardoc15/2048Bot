"""Every agent vs. every tile spawner, on the same seeded games.

Usage:
    python3 -m experiments.all_agents                    # 5 agents x 3 spawners x 100 games, all CPU cores
    python3 -m experiments.all_agents --games 20 --workers 4

Agents: Random, Greedy, Minimax (depth 2), Expectimax (depth 2), TD bot
(1,000,000 training games). Spawners: random tiles (the normal game), the
full-force villain (depth 2), and the helper (depth 2). Every agent plays
seeds 0..games-1 against every spawner, so all comparisons use the same
starting positions. Writes experiments/results/all_agents.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_CSV = os.path.join("experiments", "results", "all_agents.csv")
AGENTS = ["random", "greedy", "minimax", "expectimax", "td"]
SPAWNERS = ["random", "villain", "helper"]
FIELDS = ["agent", "spawner", "seed", "score", "max_tile", "moves", "capped", "agent_ms_per_move"]


def make_agent(name: str, seed: int):
    if name == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent(seed=seed)
    if name == "greedy":
        from agents.greedy import GreedyAgent
        return GreedyAgent()
    if name == "minimax":
        from agents.minimax import MinimaxAgent
        return MinimaxAgent(depth=2)
    if name == "expectimax":
        from agents.expectimax import ExpectimaxAgent
        return ExpectimaxAgent(depth=2)
    if name == "td":
        from td.agent import TDAgent
        return TDAgent.load(1_000_000)
    raise ValueError(name)


def make_spawner(name: str):
    from spawners import HelperSpawner, RandomSpawner, VillainSpawner
    return {"random": RandomSpawner, "villain": lambda: VillainSpawner(depth=2),
            "helper": lambda: HelperSpawner(depth=2)}[name]()


def run_chunk(agent: str, spawner: str, seeds, max_moves: int):
    from spawners import play_game_with_spawner
    rows = []
    for seed in seeds:
        r = play_game_with_spawner(make_agent(agent, seed), make_spawner(spawner), seed=seed, max_moves=max_moves)
        rows.append({"agent": agent, "spawner": spawner, "seed": seed, "score": r.score, "max_tile": r.max_tile,
                     "moves": r.moves, "capped": r.capped,
                     "agent_ms_per_move": f"{1000 * r.total_time / max(r.moves, 1):.4f}"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every agent against every tile spawner.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--agents", nargs="+", default=AGENTS, choices=AGENTS)
    parser.add_argument("--spawners", nargs="+", default=SPAWNERS, choices=SPAWNERS)
    parser.add_argument("--chunk", type=int, default=5, help="games per parallel task")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--max-moves", type=int, default=20000)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()

    seeds = list(range(args.games))
    tasks = [(a, s, seeds[i:i + args.chunk]) for a in args.agents for s in args.spawners
             for i in range(0, len(seeds), args.chunk)]
    # Slow combinations first, so the pool stays busy until the end.
    slow = {"expectimax": 3, "minimax": 2}
    tasks.sort(key=lambda t: -(slow.get(t[0], 0) + (2 if t[1] == "helper" else 0)))

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    start, done = time.perf_counter(), 0
    with open(args.csv, "w", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        futures = [pool.submit(run_chunk, a, s, chunk, args.max_moves) for a, s, chunk in tasks]
        for fut in as_completed(futures):
            rows = fut.result()
            writer.writerows(rows)
            f.flush()
            done += 1
            print(f"[{done}/{len(tasks)}] {rows[0]['agent']} vs {rows[0]['spawner']} "
                  f"seeds {rows[0]['seed']}-{rows[-1]['seed']}  {time.perf_counter() - start:6.0f}s", flush=True)
    print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
