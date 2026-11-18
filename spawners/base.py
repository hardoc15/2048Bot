"""Tile spawners: whoever decides where the next tile appears.

In standard 2048 the environment places each new tile at random. Here that
role is a pluggable *spawner*, so it can be random (the normal game), an
adversary trying to make the player lose (villain), or an ally trying to
help (helper).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Tuple

from game.board import SPAWN_2_PROBABILITY, Board, empty_cells

Spawn = Tuple[int, int, int]  # (row, col, value)


class Spawner(ABC):
    name = "spawner"

    def reset(self, rng: random.Random) -> None:
        """Called at the start of each game with that game's random generator."""

    @abstractmethod
    def choose_spawn(self, board: Board) -> Spawn:
        """Given the board right after the player's move, return (row, col, value).

        The cell must be empty and the value must be 2 or 4.
        """


class RandomSpawner(Spawner):
    """The standard 2048 rule: a uniformly random empty cell; 2 with 90% probability, else 4.

    It draws from the game's own random generator in the same order as
    ``game.Game``, so a seeded game with this spawner is identical to a
    normal seeded game.
    """

    name = "random"

    def __init__(self):
        self.rng = random.Random()

    def reset(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_spawn(self, board: Board) -> Spawn:
        row, col = self.rng.choice(empty_cells(board))
        value = 2 if self.rng.random() < SPAWN_2_PROBABILITY else 4
        return row, col, value
