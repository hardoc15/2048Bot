"""Run agents through full games and collect results."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from game.game import Game


@dataclass
class GameResult:
    seed: Optional[int]
    score: int
    max_tile: int
    moves: int
    total_time: float  # seconds spent inside the agent's choose_move

    @property
    def reached_2048(self) -> bool:
        return self.max_tile >= 2048

    @property
    def time_per_move(self) -> float:
        return self.total_time / self.moves if self.moves else 0.0


def play_game(agent, seed: Optional[int] = None, max_moves: Optional[int] = None,
              verbose: bool = False) -> GameResult:
    """Let ``agent`` play one full game and return its result.

    ``agent`` must have a ``choose_move(board)`` method (see agents/base.py).
    The game ends when no legal moves remain, or after ``max_moves``.
    If the agent returns an illegal move, the game stops with an error.
    """
    game = Game(seed=seed)
    agent_time = 0.0
    while not game.is_over():
        if max_moves is not None and game.moves >= max_moves:
            break
        start = time.perf_counter()
        direction = agent.choose_move(game.board)
        agent_time += time.perf_counter() - start
        if not game.step(direction).moved:
            raise RuntimeError(f"{type(agent).__name__} chose illegal move {direction}")
        if verbose:
            print(f"\n{direction.value.upper()}\n{game}")
    return GameResult(seed=seed, score=game.score, max_tile=game.max_tile(),
                      moves=game.moves, total_time=agent_time)


def play_games(agent, seeds: List[int], verbose: bool = False) -> List[GameResult]:
    """Play one game per seed. Using the same seeds across agents keeps comparisons fair."""
    return [play_game(agent, seed=seed, verbose=verbose) for seed in seeds]
