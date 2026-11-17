"""Stateful 2048 game: tracks the board, score, and random tile spawns."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from game.board import (
    SIZE,
    SPAWN_2_PROBABILITY,
    Board,
    Direction,
    apply_move,
    empty_board,
    empty_cells,
    format_board,
    is_game_over,
    legal_moves,
    max_tile,
    place_tile,
)

STARTING_TILES = 2


@dataclass
class MoveResult:
    """What happened when a move was played."""

    moved: bool
    score_gained: int
    spawned: Optional[tuple]  # (row, col, value) of the new tile, or None


class Game:
    """A single game of 2048.

    Pass a ``seed`` to make tile spawns reproducible, so different agents can
    be compared on the same sequence of random numbers.
    """

    def __init__(self, seed: Optional[int] = None, size: int = SIZE):
        self.size = size
        self.seed = seed
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        """Start a new game with two random tiles."""
        self.board: Board = empty_board(self.size)
        self.score = 0
        self.moves = 0
        for _ in range(STARTING_TILES):
            self._spawn_tile()

    def _spawn_tile(self) -> Optional[tuple]:
        cells = empty_cells(self.board)
        if not cells:
            return None
        row, col = self.rng.choice(cells)
        value = 2 if self.rng.random() < SPAWN_2_PROBABILITY else 4
        self.board = place_tile(self.board, row, col, value)
        return row, col, value

    def step(self, direction: Direction) -> MoveResult:
        """Play a move. An illegal move leaves the game unchanged."""
        new_board, gained, changed = apply_move(self.board, direction)
        if not changed:
            return MoveResult(moved=False, score_gained=0, spawned=None)
        self.board = new_board
        self.score += gained
        self.moves += 1
        spawned = self._spawn_tile()
        return MoveResult(moved=True, score_gained=gained, spawned=spawned)

    def legal_moves(self) -> List[Direction]:
        return legal_moves(self.board)

    def is_over(self) -> bool:
        return is_game_over(self.board)

    def max_tile(self) -> int:
        return max_tile(self.board)

    def __str__(self) -> str:
        return f"Score: {self.score}   Moves: {self.moves}\n{format_board(self.board)}"
