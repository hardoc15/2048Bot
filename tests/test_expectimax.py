import unittest
from unittest.mock import patch

from agents.expectimax import ExpectimaxAgent
from game.board import Direction, legal_moves


class ExpectimaxTests(unittest.TestCase):
    def test_depth_one_weights_spawn_cells_and_tile_values(self):
        board = ((2, 0), (0, 0))

        with patch("agents.expectimax.evaluate", side_effect=lambda position, _: position[0][0]):
            scores = ExpectimaxAgent(depth=1).score_moves(board)

        # Both legal slides leave three empty cells, including the top left.
        expected = (0.9 * 2 + 0.1 * 4) / 3
        self.assertEqual(set(scores), {Direction.DOWN, Direction.RIGHT})
        for score in scores.values():
            self.assertAlmostEqual(score, expected)

    def test_depth_two_changes_decision(self):
        board = (
            (8, 2, 2, 0),
            (0, 0, 0, 2),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )
        self.assertEqual(ExpectimaxAgent(depth=1).choose_move(board), Direction.LEFT)
        self.assertEqual(ExpectimaxAgent(depth=2).choose_move(board), Direction.UP)

    def test_tied_scores_choose_first_legal_direction(self):
        board = ((2, 0), (0, 0))
        agent = ExpectimaxAgent()

        with patch("agents.expectimax.evaluate", return_value=1.0):
            move = agent.choose_move(board)

        self.assertIn(move, legal_moves(board))
        self.assertEqual(move, Direction.DOWN)

    def test_terminal_root_raises(self):
        board = ((2, 4), (4, 2))
        with self.assertRaisesRegex(ValueError, "No legal moves"):
            ExpectimaxAgent().choose_move(board)

    def test_terminal_spawn_is_evaluated_at_any_depth(self):
        board = ((0, 2), (8, 16))
        terminal = ((2, 4), (8, 16))
        with patch(
            "agents.expectimax.evaluate",
            side_effect=lambda position, _: 10.0 if position == terminal else 0.0,
        ):
            scores = ExpectimaxAgent(depth=2).score_moves(board)

        # Moving left leaves the top right empty. Spawning a 4 ends the game.
        self.assertGreaterEqual(scores[Direction.LEFT], 0.1 * 10.0)

    def test_invalid_depths_raise(self):
        for depth in (0, -1, 1.0, "2", True, None):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                ExpectimaxAgent(depth=depth)
