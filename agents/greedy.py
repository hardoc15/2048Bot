"""Greedy agent: picks the move whose immediate result scores best.

For each legal move, the agent slides the board (before any new tile
appears) and scores the result with the shared heuristic. It takes the
highest-scoring move. There is no lookahead: it never considers where the
next tile might spawn or what it could do on the following turn. That is
exactly what separates it from Minimax and Expectimax, which use the same
heuristic but search ahead.
"""

from __future__ import annotations

from game.board import ALL_DIRECTIONS, Board, Direction, apply_move

from agents.base import Agent
from agents.heuristic import DEFAULT_WEIGHTS, HeuristicWeights, evaluate


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(self, weights: HeuristicWeights = DEFAULT_WEIGHTS):
        self.weights = weights

    def score_moves(self, board: Board) -> dict:
        """Return {direction: heuristic score} for every legal move."""
        scores = {}
        for direction in ALL_DIRECTIONS:
            new_board, _, changed = apply_move(board, direction)
            if changed:
                scores[direction] = evaluate(new_board, self.weights)
        return scores

    def choose_move(self, board: Board) -> Direction:
        scores = self.score_moves(board)
        if not scores:
            raise ValueError("No legal moves available")
        # max() keeps the first of any tied moves, in ALL_DIRECTIONS order,
        # so the agent is fully deterministic.
        return max(scores, key=scores.get)
