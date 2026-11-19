"""Check the browser demo's JavaScript engine and villain against the Python versions.

Skipped automatically if Node.js is not installed.
"""

import json
import os
import random
import shutil
import subprocess
import unittest

from agents.heuristic import evaluate
from game.board import ALL_DIRECTIONS, apply_move, from_lists
from spawners import HelperSpawner, VillainSpawner

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_JS = os.path.join(REPO, "slides", "src", "game.js")

NODE_SCRIPT = r"""
const G = require(process.argv[1]);
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = cases.map(c => {
  const b = c.board;
  const villain = G.makeSearch(false, c.depth)(b);
  const helper = G.makeSearch(true, c.depth)(b);
  return {
    moves: [0, 1, 2, 3].map(d => { const m = G.move(b, d); return [m.board, m.score, m.changed]; }),
    traced: [0, 1, 2, 3].map(d => {
      const t = G.traceMove(b, d), out = new Array(16).fill(0);
      t.moves.forEach(m => { if (!m.merged) out[m.to] = m.value; });
      t.merges.forEach(m => { out[m.at] = m.value; });
      return out;
    }),
    evaluate: G.evaluate(b),
    villain, helper,
  };
});
process.stdout.write(JSON.stringify(out));
"""


def random_board(rng, empties):
    values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    cells = [rng.choice(values) for _ in range(16)]
    for i in rng.sample(range(16), empties):
        cells[i] = 0
    return from_lists([cells[4 * r:4 * r + 4] for r in range(4)])


@unittest.skipIf(shutil.which("node") is None, "Node.js not installed")
class JavaScriptParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(7)
        cls.boards, cases = [], []
        for k in range(120):
            board = random_board(rng, empties=rng.randint(1, 9))
            depth = 1 if k % 3 == 0 else 2
            cls.boards.append((board, depth))
            cases.append({"board": [v for row in board for v in row], "depth": depth})
        proc = subprocess.run(["node", "-e", NODE_SCRIPT, GAME_JS], input=json.dumps(cases),
                              capture_output=True, text=True, check=True)
        cls.js = json.loads(proc.stdout)

    def test_moves_match(self):
        for (board, _), js in zip(self.boards, self.js):
            for i, direction in enumerate(ALL_DIRECTIONS):
                new, gained, changed = apply_move(board, direction)
                self.assertEqual(js["moves"][i], [[v for row in new for v in row], gained, changed])

    def test_animation_trace_matches_moves(self):
        for (board, _), js in zip(self.boards, self.js):
            for i, direction in enumerate(ALL_DIRECTIONS):
                new, _, _ = apply_move(board, direction)
                self.assertEqual(js["traced"][i], [v for row in new for v in row])

    def test_heuristic_matches(self):
        for (board, _), js in zip(self.boards, self.js):
            self.assertAlmostEqual(js["evaluate"], evaluate(board), places=9)

    def test_villain_and_helper_match(self):
        for (board, depth), js in zip(self.boards, self.js):
            r, c, v = VillainSpawner(depth=depth).choose_spawn(board)
            self.assertEqual(js["villain"], [4 * r + c, v])
            r, c, v = HelperSpawner(depth=depth).choose_spawn(board)
            self.assertEqual(js["helper"], [4 * r + c, v])


if __name__ == "__main__":
    unittest.main()
