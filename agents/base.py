"""Common interface every agent implements."""

from abc import ABC, abstractmethod

from game.board import Board, Direction


class Agent(ABC):
    """Base class for Random, Greedy, Minimax, and Expectimax agents.

    Subclasses implement ``choose_move``. It receives the current board and
    must return a LEGAL direction (one that changes the board); use
    ``game.legal_moves(board)`` to check. The runner only calls it while at
    least one legal move exists.
    """

    name = "agent"

    @abstractmethod
    def choose_move(self, board: Board) -> Direction:
        ...
