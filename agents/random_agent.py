"""Uniform random baseline for 2048."""

import random
from typing import Optional

from agents.base import Agent
from game import Board, Direction, legal_moves


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def choose_move(self, board: Board) -> Direction:
        return self._rng.choice(legal_moves(board))
