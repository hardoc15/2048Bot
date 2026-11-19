import random
import unittest

import numpy as np

from game.board import ALL_DIRECTIONS, apply_move, from_lists, legal_moves
from game.runner import play_game
from td.agent import TDAgent
from td.bitboard import apply, from_bitboard, to_bitboard
from td.ntuple import FEATURES, new_weights, seed_rng, train, value


def random_board(rng):
    values = [0, 0, 0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    return from_lists([[rng.choice(values) for _ in range(4)] for _ in range(4)])


class BitboardTests(unittest.TestCase):
    def test_round_trip(self):
        rng = random.Random(1)
        for _ in range(500):
            board = random_board(rng)
            self.assertEqual(from_bitboard(to_bitboard(board)), board)

    def test_moves_match_engine(self):
        rng = random.Random(2)
        for _ in range(2000):
            board = random_board(rng)
            for i, direction in enumerate(ALL_DIRECTIONS):
                expected, gained, _ = apply_move(board, direction)
                moved, score = apply(to_bitboard(board), i)
                self.assertEqual(from_bitboard(moved), expected)
                self.assertEqual(score, gained)


class NTupleTests(unittest.TestCase):
    def test_features_cover_symmetries(self):
        self.assertEqual(FEATURES.shape, (5, 8, 4))
        for t in range(FEATURES.shape[0]):
            for s in range(8):
                self.assertEqual(len(set(FEATURES[t, s])), 4)  # 4 distinct cells

    def test_value_is_symmetric(self):
        rng = np.random.default_rng(0)
        weights = rng.normal(size=new_weights().shape)
        board = from_lists([[2, 4, 8, 16], [0, 2, 0, 4], [0, 0, 2, 0], [0, 0, 0, 0]])
        mirrored = tuple(row[::-1] for row in board)
        rotated = tuple(zip(*board[::-1]))
        v = value(to_bitboard(board), weights, FEATURES)
        self.assertAlmostEqual(v, value(to_bitboard(mirrored), weights, FEATURES))
        self.assertAlmostEqual(v, value(to_bitboard(rotated), weights, FEATURES))

    def test_training_is_reproducible_and_learns(self):
        a, b = new_weights(), new_weights()
        seed_rng(5)
        scores_a, _, _ = train(a, 50)
        seed_rng(5)
        scores_b, _, _ = train(b, 50)
        np.testing.assert_array_equal(scores_a, scores_b)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.any(a != 0))


class TDAgentTests(unittest.TestCase):
    def test_always_picks_legal_moves(self):
        weights = new_weights()
        seed_rng(0)
        train(weights, 20)
        agent = TDAgent(weights)
        rng = random.Random(3)
        for _ in range(300):
            board = random_board(rng)
            moves = legal_moves(board)
            if moves:
                self.assertIn(agent.choose_move(board), moves)

    def test_plays_full_game(self):
        result = play_game(TDAgent(), seed=0)
        self.assertGreater(result.moves, 0)


if __name__ == "__main__":
    unittest.main()
