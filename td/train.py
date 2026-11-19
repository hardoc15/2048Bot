"""Train the n-tuple TD bot and save checkpoints along the way.

Usage:
    python3 -m td.train                                  # 1,000,000 games, default checkpoints
    python3 -m td.train --games 100000 --seed 1

Writes td/checkpoints/ntuple_<games>.npz at each checkpoint (including an
untrained one at 0 games) and td/checkpoints/training_log.csv with progress.
Training is seeded, so a run can be reproduced exactly.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from td.ntuple import BASE_TUPLES, new_weights, seed_rng, train

DEFAULT_CHECKPOINTS = [0, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
CHECKPOINT_DIR = os.path.join("td", "checkpoints")


def checkpoint_path(games: int, directory: str = CHECKPOINT_DIR) -> str:
    return os.path.join(directory, f"ntuple_{games}.npz")


def save(weights, games, alpha, seed, directory):
    np.savez_compressed(checkpoint_path(games, directory), weights=weights.astype(np.float32),
                        games=games, alpha=alpha, seed=seed, tuples=np.array(BASE_TUPLES))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the n-tuple TD 2048 bot.")
    parser.add_argument("--games", type=int, default=1_000_000)
    parser.add_argument("--alpha", type=float, default=0.0025, help="learning rate per weight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=CHECKPOINT_DIR)
    parser.add_argument("--log-every", type=int, default=10_000)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    marks = sorted({c for c in DEFAULT_CHECKPOINTS if c <= args.games} | {args.games})
    stops = sorted(set(marks) | set(range(args.log_every, args.games, args.log_every)))

    weights = new_weights()
    seed_rng(args.seed)
    save(weights, 0, args.alpha, args.seed, args.out)

    log_path = os.path.join(args.out, "training_log.csv")
    done, start = 0, time.perf_counter()
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["games", "avg_score", "rate_2048", "rate_4096", "max_tile", "elapsed_s"])
        for stop in stops:
            if stop <= done:
                continue
            scores, tiles, _ = train(weights, stop - done, args.alpha)
            done = stop
            elapsed = time.perf_counter() - start
            writer.writerow([done, f"{scores.mean():.1f}", f"{(tiles >= 11).mean():.4f}",
                             f"{(tiles >= 12).mean():.4f}", 2 ** int(tiles.max()), f"{elapsed:.1f}"])
            f.flush()
            if done in marks:
                save(weights, done, args.alpha, args.seed, args.out)
            print(f"{done:>9,} games  avg score {scores.mean():>8,.0f}  reached 2048 {(tiles >= 11).mean():5.1%}"
                  f"  {elapsed:7.0f}s{'  [checkpoint]' if done in marks else ''}", flush=True)
    print(f"Done. Checkpoints in {args.out}")


if __name__ == "__main__":
    main()
