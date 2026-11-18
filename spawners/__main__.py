"""Play 2048 against a tile spawner: python -m spawners --spawner villain

Spawners: random (normal 2048), villain (places tiles to make you lose),
helper (places tiles to help you win).
"""

import argparse

from game.board import Direction, apply_move, format_board, is_game_over, max_tile, place_tile
from game.game import Game

from spawners import SPAWNERS

KEYS = {"w": Direction.UP, "a": Direction.LEFT, "s": Direction.DOWN, "d": Direction.RIGHT}


def main() -> None:
    parser = argparse.ArgumentParser(description="Play 2048 against a tile spawner.")
    parser.add_argument("--spawner", choices=sorted(SPAWNERS), default="villain")
    parser.add_argument("--depth", type=int, default=2, help="search depth for villain/helper")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    spawner = SPAWNERS[args.spawner]() if args.spawner == "random" else SPAWNERS[args.spawner](depth=args.depth)
    game = Game(seed=args.seed)
    spawner.reset(game.rng)
    board, score, moves = game.board, 0, 0

    print(f"Spawner: {args.spawner}. Controls: w/a/s/d then Enter, q to quit.")
    while not is_game_over(board):
        print(f"\nScore: {score}   Moves: {moves}\n{format_board(board)}")
        key = input("Move: ").strip().lower()
        if key == "q":
            break
        if key not in KEYS:
            print("Use w, a, s, d, or q.")
            continue
        new_board, gained, changed = apply_move(board, KEYS[key])
        if not changed:
            print("That move doesn't change the board. Try another direction.")
            continue
        score += gained
        moves += 1
        row, col, value = spawner.choose_spawn(new_board)
        board = place_tile(new_board, row, col, value)
        print(f"The {args.spawner} placed a {value} at row {row + 1}, column {col + 1}.")

    ending = "Game over!" if is_game_over(board) else "Quit."
    print(f"\nScore: {score}   Moves: {moves}\n{format_board(board)}\n"
          f"{ending} Final score: {score}, highest tile: {max_tile(board)}")


if __name__ == "__main__":
    main()
