"""Play 2048 in the terminal: python -m game [--seed N]"""

import argparse

from game.board import Direction
from game.game import Game

KEYS = {
    "w": Direction.UP,
    "a": Direction.LEFT,
    "s": Direction.DOWN,
    "d": Direction.RIGHT,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Play 2048 in the terminal.")
    parser.add_argument("--seed", type=int, default=None, help="random seed for tile spawns")
    args = parser.parse_args()

    game = Game(seed=args.seed)
    print("Controls: w = up, a = left, s = down, d = right, q = quit (press Enter after each)")
    while not game.is_over():
        print(f"\n{game}")
        key = input("Move: ").strip().lower()
        if key == "q":
            break
        if key not in KEYS:
            print("Use w, a, s, d, or q.")
            continue
        if not game.step(KEYS[key]).moved:
            print("That move doesn't change the board. Try another direction.")
    ending = "Game over!" if game.is_over() else "Quit."
    print(f"\n{game}\n{ending} Final score: {game.score}, highest tile: {game.max_tile()}")


if __name__ == "__main__":
    main()
