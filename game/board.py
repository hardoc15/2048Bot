"""Pure 2048 board mechanics.

A board is an immutable tuple of rows, e.g. ``((0, 2, 0, 0), ...)``, where 0
means an empty cell. Every function here returns a new board instead of
mutating, so agents can explore hypothetical moves freely during search.
Boards are hashable, so they can also be used as cache/dictionary keys.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple

Board = Tuple[Tuple[int, ...], ...]

SIZE = 4
SPAWN_2_PROBABILITY = 0.9
SPAWN_4_PROBABILITY = 0.1


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


ALL_DIRECTIONS = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)


def empty_board(size: int = SIZE) -> Board:
    """Return a size x size board with no tiles."""
    return tuple((0,) * size for _ in range(size))


def from_lists(rows: List[List[int]]) -> Board:
    """Convert a list of lists into a Board."""
    return tuple(tuple(row) for row in rows)


def _transpose(board: Board) -> Board:
    return tuple(zip(*board))


def _reverse_rows(board: Board) -> Board:
    return tuple(row[::-1] for row in board)


def slide_row_left(row: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """Slide and merge one row to the left.

    Each tile merges at most once per move, and merges resolve from the
    direction of movement: ``[2, 2, 2, 2] -> [4, 4, 0, 0]`` and
    ``[2, 2, 2, 0] -> [4, 2, 0, 0]``.

    Returns the new row and the score gained (the sum of merged tile values).
    """
    tiles = [value for value in row if value != 0]
    merged: List[int] = []
    gained = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            value = tiles[i] * 2
            merged.append(value)
            gained += value
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged.extend([0] * (len(row) - len(merged)))
    return tuple(merged), gained


def _slide_left(board: Board) -> Tuple[Board, int]:
    rows = []
    total = 0
    for row in board:
        new_row, gained = slide_row_left(row)
        rows.append(new_row)
        total += gained
    return tuple(rows), total


def apply_move(board: Board, direction: Direction) -> Tuple[Board, int, bool]:
    """Apply a move WITHOUT spawning a new tile.

    Returns ``(new_board, score_gained, changed)``. ``changed`` is False when
    the move does nothing, which makes it illegal in 2048.
    """
    if direction is Direction.LEFT:
        new_board, gained = _slide_left(board)
    elif direction is Direction.RIGHT:
        moved, gained = _slide_left(_reverse_rows(board))
        new_board = _reverse_rows(moved)
    elif direction is Direction.UP:
        moved, gained = _slide_left(_transpose(board))
        new_board = _transpose(moved)
    elif direction is Direction.DOWN:
        moved, gained = _slide_left(_reverse_rows(_transpose(board)))
        new_board = _transpose(_reverse_rows(moved))
    else:
        raise ValueError(f"Unknown direction: {direction!r}")
    return new_board, gained, new_board != board


def legal_moves(board: Board) -> List[Direction]:
    """Return every direction that changes the board."""
    return [d for d in ALL_DIRECTIONS if apply_move(board, d)[2]]


def empty_cells(board: Board) -> List[Tuple[int, int]]:
    """Return the (row, col) coordinates of every empty cell."""
    return [
        (r, c)
        for r, row in enumerate(board)
        for c, value in enumerate(row)
        if value == 0
    ]


def place_tile(board: Board, row: int, col: int, value: int) -> Board:
    """Return a copy of ``board`` with ``value`` placed at (row, col)."""
    if board[row][col] != 0:
        raise ValueError(f"Cell ({row}, {col}) is already occupied")
    rows = [list(r) for r in board]
    rows[row][col] = value
    return from_lists(rows)


def spawn_outcomes(board: Board) -> List[Tuple[float, Board]]:
    """Return every possible tile spawn as ``(probability, resulting_board)``.

    The cell is chosen uniformly among empty cells; the tile is 2 with
    probability 0.9 and 4 with probability 0.1. Probabilities sum to 1 when
    the board has at least one empty cell. Expectimax uses these weights;
    Minimax can ignore them and take the worst outcome.
    """
    cells = empty_cells(board)
    if not cells:
        return []
    cell_probability = 1.0 / len(cells)
    outcomes = []
    for r, c in cells:
        outcomes.append((cell_probability * SPAWN_2_PROBABILITY, place_tile(board, r, c, 2)))
        outcomes.append((cell_probability * SPAWN_4_PROBABILITY, place_tile(board, r, c, 4)))
    return outcomes


def is_game_over(board: Board) -> bool:
    """True when no move can change the board."""
    return not legal_moves(board)


def max_tile(board: Board) -> int:
    """Return the largest tile value on the board."""
    return max(max(row) for row in board)


def format_board(board: Board) -> str:
    """Return a human-readable grid for printing."""
    width = max(len(str(max_tile(board))), 4)
    border = "+" + "+".join("-" * (width + 2) for _ in board[0]) + "+"
    lines = [border]
    for row in board:
        cells = [(str(v) if v else ".").center(width) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
        lines.append(border)
    return "\n".join(lines)
