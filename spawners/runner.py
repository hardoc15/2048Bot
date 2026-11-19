"""Play full games where a spawner, not the built-in randomness, places the tiles."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from game.board import Board, apply_move, is_game_over, max_tile, place_tile
from game.game import Game
from game.runner import GameResult

from spawners.base import Spawner


@dataclass
class SpawnerGameResult(GameResult):
    spawner: str = ""
    spawner_time: float = 0.0  # seconds spent inside the spawner's choose_spawn
    capped: bool = False       # True if the game hit max_moves before ending
    final_board: Optional[Board] = None


def play_game_with_spawner(agent, spawner: Spawner, seed: Optional[int] = None,
                           max_moves: Optional[int] = None) -> SpawnerGameResult:
    """Let ``agent`` play one game against ``spawner``.

    The two starting tiles are placed randomly from ``seed`` (exactly as in
    ``game.Game``), so every spawner starts from the same position. After
    that, the spawner places every new tile. ``max_moves`` caps very long
    games, which matters for the helper.
    """
    game = Game(seed=seed)
    spawner.reset(game.rng)
    board, score, moves = game.board, 0, 0
    agent_time = spawner_time = 0.0
    capped = False

    while not is_game_over(board):
        if max_moves is not None and moves >= max_moves:
            capped = True
            break
        start = time.perf_counter()
        direction = agent.choose_move(board)
        agent_time += time.perf_counter() - start

        board, gained, changed = apply_move(board, direction)
        if not changed:
            raise RuntimeError(f"{type(agent).__name__} chose illegal move {direction}")
        score += gained
        moves += 1

        start = time.perf_counter()
        row, col, value = spawner.choose_spawn(board)
        spawner_time += time.perf_counter() - start
        if value not in (2, 4):
            raise RuntimeError(f"{spawner.name} spawned invalid value {value}")
        board = place_tile(board, row, col, value)  # raises if the cell is occupied

    return SpawnerGameResult(seed=seed, score=score, max_tile=max_tile(board), moves=moves,
                             total_time=agent_time, spawner=spawner.name,
                             spawner_time=spawner_time, capped=capped, final_board=board)


def play_games_with_spawner(agent, spawner: Spawner, seeds: List[int],
                            max_moves: Optional[int] = None) -> List[SpawnerGameResult]:
    return [play_game_with_spawner(agent, spawner, seed=s, max_moves=max_moves) for s in seeds]
