"""The trained TD bot as a regular Agent, so it plays through the shared game engine."""

from __future__ import annotations

import numpy as np

from agents.base import Agent
from game.board import Board, Direction

from td.bitboard import DIRECTION_FROM_INDEX, to_bitboard
from td.ntuple import choose, new_weights
from td.train import checkpoint_path


class TDAgent(Agent):
    """Plays the move with the highest reward + learned afterstate value. No search, no learning."""

    name = "td"

    def __init__(self, weights=None, games_trained: int = 0):
        self.weights = new_weights() if weights is None else np.asarray(weights, dtype=np.float64)
        self.games_trained = games_trained

    @classmethod
    def load(cls, games: int, directory: str = None) -> "TDAgent":
        path = checkpoint_path(games) if directory is None else checkpoint_path(games, directory)
        data = np.load(path)
        return cls(data["weights"], int(data["games"]))

    def choose_move(self, board: Board) -> Direction:
        d = choose(self.weights, to_bitboard(board))
        if d < 0:
            raise ValueError("No legal moves available")
        return DIRECTION_FROM_INDEX[int(d)]
