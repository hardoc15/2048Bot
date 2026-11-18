"""Expectimax search using the engine's complete tile spawn distribution."""

from functools import lru_cache
from math import fsum
from typing import Callable, Dict

from agents.base import Agent
from agents.heuristic import DEFAULT_WEIGHTS, HeuristicWeights, evaluate
from game.board import ALL_DIRECTIONS, Board, Direction, apply_move, spawn_outcomes


class ExpectimaxAgent(Agent):
    """Search a fixed number of player moves, including the spawn after each."""

    name = "expectimax"

    def __init__(self, depth: int = 2, weights: HeuristicWeights = DEFAULT_WEIGHTS):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("Depth must be a positive integer")
        self.depth = depth
        self.weights = weights

    def score_moves(self, board: Board) -> Dict[Direction, float]:
        """Return the expected heuristic value of each legal root move."""
        weights = self.weights

        @lru_cache(maxsize=None)
        def player_value(position: Board, remaining_moves: int) -> float:
            if remaining_moves:
                scores = _score_moves(position, remaining_moves, player_value)
                if scores:
                    return max(scores.values())
            return evaluate(position, weights)

        try:
            return _score_moves(board, self.depth, player_value)
        finally:
            player_value.cache_clear()

    def choose_move(self, board: Board) -> Direction:
        scores = self.score_moves(board)
        if not scores:
            raise ValueError("No legal moves available")
        return max(scores, key=scores.__getitem__)


def _score_moves(
    board: Board, depth: int, player_value: Callable[[Board, int], float]
) -> Dict[Direction, float]:
    scores = {}
    for direction in ALL_DIRECTIONS:
        moved, _, changed = apply_move(board, direction)
        if changed:
            scores[direction] = fsum(
                probability * player_value(outcome, depth - 1)
                for probability, outcome in spawn_outcomes(moved)
            )
    return scores
