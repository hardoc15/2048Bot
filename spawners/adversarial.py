"""Villain and helper spawners: tile placement chosen by adversarial search.

The spawner and the player take turns: the spawner places a tile, the player
slides, the spawner places another tile, and so on. The spawner searches
this game tree a fixed number of player turns ahead:

- the **villain** picks the tile that leads to the WORST outcome for the player,
- the **helper** picks the tile that leads to the BEST outcome for the player.

At each player turn the search assumes the player picks the move with the
highest heuristic value (the shared heuristic from agents/heuristic.py), and
at the search horizon it scores the board with that same heuristic. A spawn
that leaves the player with no legal move ends the game immediately, so the
villain always takes one if it exists and the helper always avoids it.

Because the search models the player with the shared heuristic, the villain
"knows" exactly how the Greedy agent thinks. Against other players (humans,
Minimax, Expectimax) it is a proxy model of what they value.
"""

from __future__ import annotations

from typing import List, Sequence

from agents.heuristic import DEFAULT_WEIGHTS, HeuristicWeights, evaluate
from game.board import ALL_DIRECTIONS, Board, apply_move, empty_cells, place_tile

from spawners.base import Spawn, Spawner

GAME_OVER = -1e9  # value of a board where the player has no legal move


class SearchSpawner(Spawner):
    """Shared search for the villain and helper.

    ``depth`` is how many player turns the spawner looks ahead. Depth 2 (the
    default) takes a few milliseconds per tile, fast enough for live play.
    Each extra level multiplies the cost by about 2 x (number of empty cells).

    Depth matters much more for the helper than the villain: a depth-1 helper
    is too short-sighted to help (it mostly hands out 4s that look good now
    and jams the board), while a depth-1 villain is already devastating.
    """

    wants_player_to_win = False

    def __init__(self, depth: int = 2, weights: HeuristicWeights = DEFAULT_WEIGHTS,
                 values: Sequence[int] = (2, 4)):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        self.weights = weights
        self.values = tuple(values)
        self._cache = {}

    def _options(self, board: Board) -> List[Spawn]:
        return [(r, c, v) for r, c in empty_cells(board) for v in self.values]

    def _pick(self, values):
        return max(values) if self.wants_player_to_win else min(values)

    def _evaluate(self, board: Board) -> float:
        value = self._cache.get(board)
        if value is None:
            value = self._cache[board] = evaluate(board, self.weights)
        return value

    def _player_value(self, board: Board, depth: int) -> float:
        """Value of ``board`` for the player, who is about to move.

        The player is modeled as greedy: it takes the move whose result has
        the highest heuristic value. Only that move is searched further.
        """
        best_after, best_value = None, None
        for direction in ALL_DIRECTIONS:
            after, _, changed = apply_move(board, direction)
            if changed:
                value = self._evaluate(after)
                if best_value is None or value > best_value:
                    best_after, best_value = after, value
        if best_after is None:
            return GAME_OVER
        if depth == 1:
            return best_value
        return self._spawner_value(best_after, depth - 1)

    def _spawner_value(self, board: Board, depth: int) -> float:
        """Value for the player after the spawner, who is about to place a tile, plays its best."""
        return self._pick(self._player_value(place_tile(board, r, c, v), depth)
                          for r, c, v in self._options(board))

    def score_spawns(self, board: Board) -> dict:
        """Return {(row, col, value): value for the player} for every possible spawn."""
        return {(r, c, v): self._player_value(place_tile(board, r, c, v), self.depth)
                for r, c, v in self._options(board)}

    def choose_spawn(self, board: Board) -> Spawn:
        self._cache = {}  # heuristic values, reused across this one decision
        scores = self.score_spawns(board)
        if not scores:
            raise ValueError("No empty cell to spawn into")
        # min/max keep the first of any ties, so the spawner is deterministic.
        chooser = max if self.wants_player_to_win else min
        return chooser(scores, key=scores.get)


class VillainSpawner(SearchSpawner):
    """Places every tile where it hurts the player most."""

    name = "villain"
    wants_player_to_win = False


class HelperSpawner(SearchSpawner):
    """Places every tile where it helps the player most."""

    name = "helper"
    wants_player_to_win = True
