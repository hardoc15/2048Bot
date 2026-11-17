import unittest

from game.board import (
    Direction,
    apply_move,
    from_lists,
    is_game_over,
    legal_moves,
    max_tile,
    slide_row_left,
    spawn_outcomes,
)
from game.game import Game
from game.runner import play_game


class SlideRowTests(unittest.TestCase):
    def test_slides_without_merge(self):
        self.assertEqual(slide_row_left((0, 2, 0, 4)), ((2, 4, 0, 0), 0))

    def test_merges_pair(self):
        self.assertEqual(slide_row_left((2, 0, 2, 0)), ((4, 0, 0, 0), 4))

    def test_each_tile_merges_once(self):
        self.assertEqual(slide_row_left((2, 2, 2, 2)), ((4, 4, 0, 0), 8))
        self.assertEqual(slide_row_left((4, 4, 8, 0)), ((8, 8, 0, 0), 8))

    def test_merge_prefers_leading_pair(self):
        self.assertEqual(slide_row_left((2, 2, 2, 0)), ((4, 2, 0, 0), 4))


class MoveTests(unittest.TestCase):
    board = from_lists([
        [2, 0, 0, 2],
        [0, 4, 0, 0],
        [2, 0, 0, 0],
        [0, 4, 0, 8],
    ])

    def test_left(self):
        new, gained, changed = apply_move(self.board, Direction.LEFT)
        self.assertEqual(new, from_lists([[4, 0, 0, 0], [4, 0, 0, 0], [2, 0, 0, 0], [4, 8, 0, 0]]))
        self.assertEqual(gained, 4)
        self.assertTrue(changed)

    def test_right(self):
        new, gained, _ = apply_move(self.board, Direction.RIGHT)
        self.assertEqual(new, from_lists([[0, 0, 0, 4], [0, 0, 0, 4], [0, 0, 0, 2], [0, 0, 4, 8]]))
        self.assertEqual(gained, 4)

    def test_up(self):
        new, gained, _ = apply_move(self.board, Direction.UP)
        self.assertEqual(new, from_lists([[4, 8, 0, 2], [0, 0, 0, 8], [0, 0, 0, 0], [0, 0, 0, 0]]))
        self.assertEqual(gained, 12)

    def test_down(self):
        new, gained, _ = apply_move(self.board, Direction.DOWN)
        self.assertEqual(new, from_lists([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [4, 8, 0, 8]]))
        self.assertEqual(gained, 12)

    def test_no_op_move_is_illegal(self):
        board = from_lists([[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertFalse(apply_move(board, Direction.LEFT)[2])
        self.assertFalse(apply_move(board, Direction.UP)[2])
        self.assertEqual(set(legal_moves(board)), {Direction.RIGHT, Direction.DOWN})

    def test_does_not_mutate_input(self):
        before = self.board
        apply_move(self.board, Direction.LEFT)
        self.assertIs(self.board, before)


class GameStateTests(unittest.TestCase):
    def test_game_over(self):
        stuck = from_lists([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
        self.assertTrue(is_game_over(stuck))
        mergeable = from_lists([[2, 2, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
        self.assertFalse(is_game_over(mergeable))

    def test_max_tile(self):
        self.assertEqual(max_tile(from_lists([[2, 0, 0, 0], [0, 128, 0, 0], [0] * 4, [0] * 4])), 128)

    def test_spawn_outcomes_probabilities(self):
        board = from_lists([[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 0, 0]])
        outcomes = spawn_outcomes(board)
        self.assertEqual(len(outcomes), 4)  # 2 empty cells x {2, 4}
        self.assertAlmostEqual(sum(p for p, _ in outcomes), 1.0)
        self.assertAlmostEqual(outcomes[0][0], 0.45)
        self.assertAlmostEqual(outcomes[1][0], 0.05)
        self.assertEqual(spawn_outcomes(from_lists([[2] * 4] * 4)), [])


class GameTests(unittest.TestCase):
    def test_starts_with_two_tiles(self):
        game = Game(seed=1)
        tiles = [v for row in game.board for v in row if v]
        self.assertEqual(len(tiles), 2)
        self.assertTrue(all(v in (2, 4) for v in tiles))

    def test_seed_is_reproducible(self):
        a, b = Game(seed=42), Game(seed=42)
        for direction in [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN] * 5:
            a.step(direction)
            b.step(direction)
        self.assertEqual(a.board, b.board)
        self.assertEqual(a.score, b.score)

    def test_illegal_move_changes_nothing(self):
        game = Game(seed=0)
        game.board = from_lists([[2, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4])
        result = game.step(Direction.LEFT)
        self.assertFalse(result.moved)
        self.assertEqual(game.moves, 0)
        self.assertEqual(game.board, from_lists([[2, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4]))

    def test_legal_move_spawns_one_tile(self):
        game = Game(seed=0)
        game.board = from_lists([[2, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4])
        result = game.step(Direction.RIGHT)
        self.assertTrue(result.moved)
        self.assertEqual(sum(1 for row in game.board for v in row if v), 3)
        self.assertEqual(game.moves, 1)


class RunnerTests(unittest.TestCase):
    def test_play_game_to_completion(self):
        class FirstLegalAgent:
            def choose_move(self, board):
                return legal_moves(board)[0]

        result = play_game(FirstLegalAgent(), seed=7)
        self.assertGreater(result.moves, 0)
        self.assertGreaterEqual(result.max_tile, 4)


if __name__ == "__main__":
    unittest.main()
