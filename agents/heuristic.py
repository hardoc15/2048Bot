"""Board evaluation heuristic shared by the Greedy, Minimax, and Expectimax agents.

Higher scores mean better boards. Tile values are compared on a log2 scale
(2 -> 1, 4 -> 2, 2048 -> 11) so that big tiles don't drown out every other
feature.

The heuristic combines four features:

- empty cells:  more free space means more room to maneuver.
- monotonicity: tiles should increase or decrease steadily along each row and
                column, so they can be merged in a chain.
- smoothness:   neighboring tiles with similar values are easier to merge.
- corner:       the largest tile should be kept in a corner.

Using one shared ``evaluate`` with the same weights keeps the agent
comparison fair: differences in results come from the search strategy, not
from different board evaluations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from game.board import Board, max_tile


@dataclass(frozen=True)
class HeuristicWeights:
    """Feature weights.

    Defaults were picked by a grid search with the Greedy agent
    (empty in {1, 2.7, 5}, monotonicity in {0.5, 1, 2}, smoothness in
    {0, 0.1, 0.5}, corner in {0, 1, 3}; 60 games each on seeds 1000-1059),
    then checked on 100 held-out games (seeds 5000-5099). Smoothness had the
    biggest effect. Re-tune if Minimax/Expectimax behave differently.
    """

    empty: float = 1.0
    monotonicity: float = 1.0
    smoothness: float = 0.5
    corner: float = 3.0


DEFAULT_WEIGHTS = HeuristicWeights()


def _log_board(board: Board) -> List[List[float]]:
    return [[math.log2(v) if v else 0.0 for v in row] for row in board]


def _lines(values: List[List[float]]) -> List[Sequence[float]]:
    """Every row and every column."""
    return values + [list(col) for col in zip(*values)]


def empty_cells_feature(board: Board) -> float:
    """Number of empty cells (0 to 16)."""
    return float(sum(1 for row in board for v in row if v == 0))


def monotonicity_feature(board: Board) -> float:
    """How far each row/column is from being monotonic (0 is perfect, more negative is worse).

    For each line we measure how much it breaks an increasing order and how much
    it breaks a decreasing order, and keep the smaller of the two, so a line is
    only penalized for zig-zagging, not for which way it slopes.
    """
    penalty = 0.0
    for line in _lines(_log_board(board)):
        breaks_increasing = 0.0
        breaks_decreasing = 0.0
        for a, b in zip(line, line[1:]):
            if a > b:
                breaks_increasing += a - b
            else:
                breaks_decreasing += b - a
        penalty += min(breaks_increasing, breaks_decreasing)
    return -penalty


def smoothness_feature(board: Board) -> float:
    """Negative sum of log2 differences between adjacent non-empty tiles (0 is perfectly smooth)."""
    values = _log_board(board)
    total = 0.0
    for line in _lines(values):
        for a, b in zip(line, line[1:]):
            if a and b:
                total += abs(a - b)
    return -total


def corner_feature(board: Board) -> float:
    """log2 of the largest tile if it sits in a corner, otherwise 0."""
    largest = max_tile(board)
    if largest == 0:
        return 0.0
    n = len(board) - 1
    corners = (board[0][0], board[0][n], board[n][0], board[n][n])
    return math.log2(largest) if largest in corners else 0.0


def features(board: Board) -> Dict[str, float]:
    """Return each raw feature value, for debugging and for the paper."""
    return {
        "empty": empty_cells_feature(board),
        "monotonicity": monotonicity_feature(board),
        "smoothness": smoothness_feature(board),
        "corner": corner_feature(board),
    }


def evaluate(board: Board, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    """Weighted sum of all features. Higher is better."""
    f = features(board)
    return (
        weights.empty * f["empty"]
        + weights.monotonicity * f["monotonicity"]
        + weights.smoothness * f["smoothness"]
        + weights.corner * f["corner"]
    )
