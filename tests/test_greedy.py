import unittest

from agents.greedy import GreedyAgent
from agents.heuristic import (
    HeuristicWeights,
    corner_feature,
    empty_cells_feature,
    evaluate,
    monotonicity_feature,
    smoothness_feature,
)
from game.board import Direction, from_lists, legal_moves
from game.runner import play_game


class HeuristicFeatureTests(unittest.TestCase):
    def test_empty_cells(self):
        board = from_lists([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 8]])
        self.assertEqual(empty_cells_feature(board), 13)

    def test_monotonic_board_has_no_penalty(self):
        board = from_lists([[16, 8, 4, 2], [8, 4, 2, 0], [4, 2, 0, 0], [2, 0, 0, 0]])
        self.assertEqual(monotonicity_feature(board), 0)

    def test_zigzag_row_is_penalized(self):
        mono = from_lists([[2, 4, 8, 16], [0] * 4, [0] * 4, [0] * 4])
        zigzag = from_lists([[2, 16, 4, 8], [0] * 4, [0] * 4, [0] * 4])
        self.assertLess(monotonicity_feature(zigzag), monotonicity_feature(mono))

    def test_smoothness(self):
        smooth = from_lists([[4, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4])
        rough = from_lists([[4, 64, 0, 0], [0] * 4, [0] * 4, [0] * 4])
        self.assertEqual(smoothness_feature(smooth), 0)
        self.assertEqual(smoothness_feature(rough), -4)  # |log2(4) - log2(64)| = 4

    def test_corner(self):
        cornered = from_lists([[0] * 4, [0] * 4, [0, 2, 0, 0], [0, 0, 0, 128]])
        centered = from_lists([[0] * 4, [0, 128, 0, 0], [0, 2, 0, 0], [0] * 4])
        self.assertEqual(corner_feature(cornered), 7)
        self.assertEqual(corner_feature(centered), 0)

    def test_weights_change_evaluation(self):
        board = from_lists([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])
        only_empty = HeuristicWeights(empty=1, monotonicity=0, smoothness=0, corner=0)
        self.assertEqual(evaluate(board, only_empty), 15)


class GreedyAgentTests(unittest.TestCase):
    def test_takes_the_merge_that_keeps_corner(self):
        # Left merges the pair of 64s into the top-left corner.
        board = from_lists([[64, 64, 0, 0], [2, 0, 0, 0], [0] * 4, [0] * 4])
        self.assertEqual(GreedyAgent().choose_move(board), Direction.LEFT)

    def test_only_scores_legal_moves(self):
        board = from_lists([[2, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4])
        scores = GreedyAgent().score_moves(board)
        self.assertEqual(set(scores), set(legal_moves(board)))

    def test_is_deterministic(self):
        a = play_game(GreedyAgent(), seed=3)
        b = play_game(GreedyAgent(), seed=3)
        self.assertEqual((a.score, a.max_tile, a.moves), (b.score, b.max_tile, b.moves))

    def test_plays_competently(self):
        results = [play_game(GreedyAgent(), seed=s) for s in range(5)]
        self.assertGreaterEqual(max(r.max_tile for r in results), 512)


if __name__ == "__main__":
    unittest.main()
