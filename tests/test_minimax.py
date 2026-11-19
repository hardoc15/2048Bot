import unittest
from unittest.mock import patch

from agents.heuristic import evaluate
from agents.minimax import MinimaxAgent
from game.board import Direction, apply_move, legal_moves, spawn_outcomes
from game.runner import play_game


class MinimaxTests(unittest.TestCase):
    def test_uses_the_worst_spawn_including_a_rare_four(self):
        board = ((2, 0), (0, 0))

        def heuristic(position, _):
            if position[0][1] == 4 and position[1][0] == 2:
                return -10.0
            return 1.0

        # Down allows a harmful 4 in the top-right cell; right occupies that cell.
        with patch("agents.minimax.evaluate", side_effect=heuristic):
            self.assertEqual(MinimaxAgent(depth=1).choose_move(board), Direction.RIGHT)

    def test_pruned_search_matches_exhaustive_search(self):
        def spawn_value(board, depth):
            return min(player_value(outcome, depth - 1) for _, outcome in spawn_outcomes(board))

        def player_value(board, depth):
            if depth:
                values = move_values(board, depth)
                if values:
                    return max(values.values())
            return evaluate(board)

        def move_values(board, depth):
            return {
                direction: spawn_value(apply_move(board, direction)[0], depth)
                for direction in legal_moves(board)
            }

        boards = (
            ((2, 0), (0, 0)),
            ((2, 4), (2, 8)),
            ((0, 2), (8, 16)),
            ((8, 2, 2, 0), (0, 0, 0, 2), (0, 0, 0, 0), (0, 0, 0, 0)),
        )
        for depth in (1, 2, 3):
            small_boards = boards if depth < 3 else boards[:3]
            for board in small_boards:
                with self.subTest(depth=depth, board=board):
                    values = move_values(board, depth)
                    self.assertEqual(
                        MinimaxAgent(depth).choose_move(board), max(values, key=values.get)
                    )

    def test_ties_choose_first_legal_move(self):
        with patch("agents.minimax.evaluate", return_value=0.0):
            self.assertEqual(MinimaxAgent().choose_move(((2, 0), (0, 0))), Direction.DOWN)

    def test_terminal_board_has_no_move(self):
        with self.assertRaisesRegex(ValueError, "No legal moves"):
            MinimaxAgent().choose_move(((2, 4), (4, 2)))

    def test_invalid_depths_raise(self):
        for depth in (0, -1, 1.0, "2", True, None):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                MinimaxAgent(depth)

    def test_plays_a_complete_seeded_game(self):
        result = play_game(MinimaxAgent(depth=1), seed=0)
        self.assertGreater(result.moves, 0)
        self.assertGreater(result.score, 0)
