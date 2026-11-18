"""Worst-case tile-spawn search for 2048 with alpha-beta pruning."""

from math import inf

from agents.base import Agent
from agents.heuristic import DEFAULT_WEIGHTS, HeuristicWeights, evaluate
from game.board import ALL_DIRECTIONS, Board, Direction, apply_move, spawn_outcomes


class MinimaxAgent(Agent):
    """Search a fixed number of player moves, including the spawn after each."""

    name = "minimax"

    def __init__(self, depth: int = 2, weights: HeuristicWeights = DEFAULT_WEIGHTS):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("Depth must be a positive integer")
        self.depth = depth
        self.weights = weights

    def choose_move(self, board: Board) -> Direction:
        best_move = None
        best_value = -inf
        for direction in ALL_DIRECTIONS:
            moved, _, changed = apply_move(board, direction)
            if changed:
                value = self._spawn_value(moved, self.depth - 1, best_value, inf)
                if best_move is None or value > best_value:
                    best_move = direction
                    best_value = value

        if best_move is None:
            raise ValueError("No legal moves available")
        return best_move

    def _player_value(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        if depth == 0:
            return evaluate(board, self.weights)

        best = -inf
        for direction in ALL_DIRECTIONS:
            moved, _, changed = apply_move(board, direction)
            if changed:
                best = max(best, self._spawn_value(moved, depth - 1, alpha, beta))
                if best >= beta:
                    return best
                alpha = max(alpha, best)

        # A terminal board is evaluated with the same heuristic as a depth cutoff.
        return evaluate(board, self.weights) if best == -inf else best

    def _spawn_value(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        worst = inf
        for _, spawned in spawn_outcomes(board):
            worst = min(worst, self._player_value(spawned, depth, alpha, beta))
            if worst <= alpha:
                return worst
            beta = min(beta, worst)
        return worst
