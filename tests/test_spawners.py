import unittest

from agents.greedy import GreedyAgent
from game.board import empty_cells, from_lists
from game.runner import play_game
from spawners import (
    HelperSpawner,
    RandomSpawner,
    VillainSpawner,
    play_game_with_spawner,
)
from spawners.adversarial import GAME_OVER


class RandomSpawnerTests(unittest.TestCase):
    def test_matches_normal_game_exactly(self):
        for seed in range(3):
            normal = play_game(GreedyAgent(), seed=seed)
            spawned = play_game_with_spawner(GreedyAgent(), RandomSpawner(), seed=seed)
            self.assertEqual((normal.score, normal.max_tile, normal.moves),
                             (spawned.score, spawned.max_tile, spawned.moves))


class SearchSpawnerTests(unittest.TestCase):
    # One empty cell at (3, 3) and no adjacent equal tiles anywhere else.
    # A 4 there leaves no legal move (game over); a 2 there can merge with
    # the 2 above it, so the game continues.
    almost_full = from_lists([
        [2, 4, 2, 4],
        [4, 2, 4, 8],
        [2, 4, 8, 2],
        [4, 2, 16, 0],
    ])

    def test_villain_ends_game_when_it_can(self):
        for depth in (1, 2):
            self.assertEqual(VillainSpawner(depth=depth).choose_spawn(self.almost_full), (3, 3, 4))

    def test_helper_avoids_ending_game(self):
        for depth in (1, 2):
            self.assertEqual(HelperSpawner(depth=depth).choose_spawn(self.almost_full), (3, 3, 2))

    def test_scores_mark_game_over(self):
        scores = VillainSpawner(depth=1).score_spawns(self.almost_full)
        self.assertEqual(scores[(3, 3, 4)], GAME_OVER)
        self.assertGreater(scores[(3, 3, 2)], GAME_OVER)

    def test_only_spawns_into_empty_cells(self):
        board = from_lists([[2, 0, 0, 0], [0, 4, 0, 0], [0] * 4, [0] * 4])
        for spawner in (VillainSpawner(), HelperSpawner()):
            row, col, value = spawner.choose_spawn(board)
            self.assertIn((row, col), empty_cells(board))
            self.assertIn(value, (2, 4))

    def test_respects_allowed_values(self):
        board = from_lists([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])
        self.assertEqual(VillainSpawner(values=(2,)).choose_spawn(board)[2], 2)

    def test_rejects_bad_depth(self):
        with self.assertRaises(ValueError):
            VillainSpawner(depth=0)


class SpawnerGameTests(unittest.TestCase):
    def test_villain_hurts_and_helper_helps(self):
        seeds = range(2)
        normal = sum(play_game_with_spawner(GreedyAgent(), RandomSpawner(), seed=s).score for s in seeds)
        villain = sum(play_game_with_spawner(GreedyAgent(), VillainSpawner(depth=1), seed=s).score
                      for s in seeds)
        self.assertLess(villain, normal)

    def test_helper_beats_villain(self):
        villain = play_game_with_spawner(GreedyAgent(), VillainSpawner(depth=1), seed=0)
        helper = play_game_with_spawner(GreedyAgent(), HelperSpawner(depth=1), seed=0, max_moves=200)
        self.assertGreater(helper.score, villain.score)

    def test_max_moves_caps_game(self):
        result = play_game_with_spawner(GreedyAgent(), HelperSpawner(depth=1), seed=0, max_moves=50)
        self.assertTrue(result.capped)
        self.assertEqual(result.moves, 50)


if __name__ == "__main__":
    unittest.main()
