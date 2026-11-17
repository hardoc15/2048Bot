"""2048 game engine shared by all agents."""

from game.board import (
    ALL_DIRECTIONS,
    Board,
    Direction,
    apply_move,
    empty_cells,
    format_board,
    is_game_over,
    legal_moves,
    max_tile,
    spawn_outcomes,
)
from game.game import Game, MoveResult
from game.runner import GameResult, play_game, play_games

__all__ = [
    "ALL_DIRECTIONS",
    "Board",
    "Direction",
    "Game",
    "GameResult",
    "MoveResult",
    "apply_move",
    "empty_cells",
    "format_board",
    "is_game_over",
    "legal_moves",
    "max_tile",
    "play_game",
    "play_games",
    "spawn_outcomes",
]
