"""Fast 2048 board for training: the whole board packed into one 64-bit integer.

Each cell uses 4 bits holding the tile's exponent (0 = empty, 1 = 2, 2 = 4,
..., 11 = 2048, up to 15 = 32768). Cell (row, col) lives at bits
4 * (4 * row + col), so row r is the 16 bits starting at bit 16 * r, with
column 0 in the lowest 4 bits.

Sliding is precomputed: every possible row (65,536 of them) is slid left and
right once at import time, so a move is just four table lookups. All hot
functions are compiled with Numba.

Directions use the same order as game.board.ALL_DIRECTIONS:
0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from game.board import ALL_DIRECTIONS, Board, Direction

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRECTION_FROM_INDEX = {i: d for i, d in enumerate(ALL_DIRECTIONS)}
MAX_EXPONENT = 15  # 32768; higher tiles cannot be stored


def _slide_row_left(cells):
    """Slide one row of exponents left. Returns (new_cells, score_gained)."""
    tiles = [c for c in cells if c]
    out, score, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < MAX_EXPONENT:
            out.append(tiles[i] + 1)
            score += 1 << (tiles[i] + 1)
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (4 - len(out)), score


def _build_tables():
    left = np.zeros(65536, dtype=np.uint16)
    right = np.zeros(65536, dtype=np.uint16)
    score = np.zeros(65536, dtype=np.int64)  # same for left and right of the reversed row
    score_right = np.zeros(65536, dtype=np.int64)
    for row in range(65536):
        cells = [(row >> (4 * i)) & 0xF for i in range(4)]
        new, gained = _slide_row_left(cells)
        left[row] = sum(v << (4 * i) for i, v in enumerate(new))
        score[row] = gained
        rev_new, rev_gained = _slide_row_left(cells[::-1])
        right[row] = sum(v << (4 * i) for i, v in enumerate(rev_new[::-1]))
        score_right[row] = rev_gained
    return left, right, score, score_right


ROW_LEFT, ROW_RIGHT, SCORE_LEFT, SCORE_RIGHT = _build_tables()


@njit(cache=True)
def transpose(board):
    out = np.uint64(0)
    for r in range(4):
        for c in range(4):
            v = (board >> np.uint64(4 * (4 * r + c))) & np.uint64(0xF)
            out |= v << np.uint64(4 * (4 * c + r))
    return out


@njit(cache=True)
def _slide_rows(board, table, score_table):
    out = np.uint64(0)
    score = 0
    for r in range(4):
        row = (board >> np.uint64(16 * r)) & np.uint64(0xFFFF)
        out |= np.uint64(table[row]) << np.uint64(16 * r)
        score += score_table[row]
    return out, score


@njit(cache=True)
def move(board, direction, row_left, row_right, score_left, score_right):
    """Apply a move without spawning. Returns (new_board, score_gained)."""
    if direction == LEFT:
        return _slide_rows(board, row_left, score_left)
    if direction == RIGHT:
        return _slide_rows(board, row_right, score_right)
    t = transpose(board)
    if direction == UP:
        moved, score = _slide_rows(t, row_left, score_left)
    else:
        moved, score = _slide_rows(t, row_right, score_right)
    return transpose(moved), score


@njit(cache=True)
def count_empty(board):
    n = 0
    for i in range(16):
        if (board >> np.uint64(4 * i)) & np.uint64(0xF) == 0:
            n += 1
    return n


@njit(cache=True)
def spawn_random(board):
    """Place a 2 (90%) or 4 (10%) in a uniformly random empty cell (Numba's RNG)."""
    empty = count_empty(board)
    if empty == 0:
        return board
    k = np.random.randint(empty)
    exponent = np.uint64(1) if np.random.random() < 0.9 else np.uint64(2)
    for i in range(16):
        if (board >> np.uint64(4 * i)) & np.uint64(0xF) == 0:
            if k == 0:
                return board | (exponent << np.uint64(4 * i))
            k -= 1
    return board


@njit(cache=True)
def max_exponent(board):
    m = 0
    for i in range(16):
        v = (board >> np.uint64(4 * i)) & np.uint64(0xF)
        if v > m:
            m = v
    return m


def to_bitboard(board: Board) -> np.uint64:
    """Convert an engine board (tuple of rows of tile values) to a bitboard."""
    out = 0
    for r, row in enumerate(board):
        for c, value in enumerate(row):
            if value:
                exponent = value.bit_length() - 1
                if exponent > MAX_EXPONENT:
                    raise ValueError(f"Tile {value} is too large for the bitboard")
                out |= exponent << (4 * (4 * r + c))
    return np.uint64(out)


def from_bitboard(bitboard) -> Board:
    """Convert a bitboard back to an engine board."""
    b = int(bitboard)
    return tuple(
        tuple((1 << e) if (e := (b >> (4 * (4 * r + c))) & 0xF) else 0 for c in range(4))
        for r in range(4)
    )


def apply(bitboard, direction: int):
    """Convenience wrapper around move() using the module's tables."""
    return move(np.uint64(bitboard), direction, ROW_LEFT, ROW_RIGHT, SCORE_LEFT, SCORE_RIGHT)
