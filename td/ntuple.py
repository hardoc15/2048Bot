"""N-tuple network trained by temporal-difference (TD) learning.

Method: Szubert & Jaskowski, "Temporal Difference Learning of N-Tuple
Networks for the Game 2048" (IEEE CIG 2014). Implemented from scratch here.

The value of a board is the sum of entries from several lookup tables. Each
table covers a fixed group of 4 cells (an "n-tuple"); the 4 tile exponents in
those cells form an index into the table (16^4 = 65,536 entries). Every
pattern is applied in all 8 rotations/reflections of the board, sharing one
table, so the network learns symmetric knowledge with less data.

Learning uses afterstates (the board right after a slide, before the random
tile appears). The agent picks the move maximizing reward + V(afterstate).
After each move, V(previous afterstate) is nudged toward
reward_next + V(next afterstate), the TD(0) target; at game end the target
is 0.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from td.bitboard import (
    ROW_LEFT,
    ROW_RIGHT,
    SCORE_LEFT,
    SCORE_RIGHT,
    max_exponent,
    move,
    spawn_random,
)

# Base patterns as (row, col) cells. Each is expanded to 8 symmetric copies.
BASE_TUPLES = [
    [(0, 0), (0, 1), (0, 2), (0, 3)],  # outer row
    [(1, 0), (1, 1), (1, 2), (1, 3)],  # inner row
    [(0, 0), (0, 1), (1, 0), (1, 1)],  # corner square
    [(0, 1), (0, 2), (1, 1), (1, 2)],  # edge square
    [(1, 1), (1, 2), (2, 1), (2, 2)],  # center square
]
TABLE_SIZE = 16 ** 4


def _symmetries(r, c):
    """The 8 rotations/reflections of cell (r, c) on a 4x4 board."""
    n = 3
    return [(r, c), (c, n - r), (n - r, n - c), (n - c, r),       # rotations
            (r, n - c), (n - c, n - r), (n - r, c), (c, r)]       # reflections


def build_features():
    """Array [num_tuples, 8, 4] of cell indices (4 * row + col)."""
    feats = np.zeros((len(BASE_TUPLES), 8, 4), dtype=np.int64)
    for t, cells in enumerate(BASE_TUPLES):
        for k, (r, c) in enumerate(cells):
            for s, (rr, cc) in enumerate(_symmetries(r, c)):
                feats[t, s, k] = 4 * rr + cc
    return feats


FEATURES = build_features()


def new_weights():
    return np.zeros((len(BASE_TUPLES), TABLE_SIZE), dtype=np.float64)


@njit(cache=True)
def _index(board, cells):
    idx = 0
    for k in range(cells.shape[0]):
        idx |= int((board >> np.uint64(4 * cells[k])) & np.uint64(0xF)) << (4 * k)
    return idx


@njit(cache=True)
def value(board, weights, feats):
    total = 0.0
    for t in range(feats.shape[0]):
        for s in range(feats.shape[1]):
            total += weights[t, _index(board, feats[t, s])]
    return total


@njit(cache=True)
def _update(board, weights, feats, delta):
    for t in range(feats.shape[0]):
        for s in range(feats.shape[1]):
            weights[t, _index(board, feats[t, s])] += delta


@njit(cache=True)
def best_move(board, weights, feats, row_left, row_right, score_left, score_right):
    """Return (direction, afterstate, reward) maximizing reward + V(afterstate); direction -1 if none.

    Ties go to the first direction in order UP, DOWN, LEFT, RIGHT.
    """
    best_d, best_after, best_r, best_v = -1, board, 0, -1e300
    for d in range(4):
        after, r = move(board, d, row_left, row_right, score_left, score_right)
        if after == board:
            continue
        v = r + value(after, weights, feats)
        if v > best_v:
            best_d, best_after, best_r, best_v = d, after, r, v
    return best_d, best_after, best_r


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def train_games(weights, feats, n_games, alpha, row_left, row_right, score_left, score_right):
    """Play n_games of self-play against random tile spawns, learning as it goes.

    Returns arrays of each game's score, highest tile exponent, and move count.
    """
    scores = np.zeros(n_games, dtype=np.int64)
    max_tiles = np.zeros(n_games, dtype=np.int64)
    moves = np.zeros(n_games, dtype=np.int64)
    for g in range(n_games):
        board = spawn_random(spawn_random(np.uint64(0)))
        d, after, r = best_move(board, weights, feats, row_left, row_right, score_left, score_right)
        score, n = 0, 0
        while d != -1:
            score += r
            n += 1
            board = spawn_random(after)
            d2, after2, r2 = best_move(board, weights, feats, row_left, row_right, score_left, score_right)
            target = 0.0 if d2 == -1 else r2 + value(after2, weights, feats)
            error = target - value(after, weights, feats)
            _update(after, weights, feats, alpha * error)
            d, after, r = d2, after2, r2
        scores[g] = score
        max_tiles[g] = max_exponent(board)
        moves[g] = n
    return scores, max_tiles, moves


def train(weights, n_games, alpha=0.0025):
    """Python-friendly wrapper using the module's tables."""
    return train_games(weights, FEATURES, n_games, alpha, ROW_LEFT, ROW_RIGHT, SCORE_LEFT, SCORE_RIGHT)


def choose(weights, bitboard):
    d, _, _ = best_move(np.uint64(bitboard), weights, FEATURES, ROW_LEFT, ROW_RIGHT, SCORE_LEFT, SCORE_RIGHT)
    return d
