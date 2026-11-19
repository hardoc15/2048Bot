// 2048 engine, shared heuristic, and villain spawner for the browser demo.
// A JavaScript port of game/board.py, agents/heuristic.py and
// spawners/adversarial.py. tests/test_slides_js.py checks it against the
// Python versions on the same boards.
//
// A board is an array of 16 tile values (0 = empty), row-major: index 4*row + col.
// Directions follow game.board.ALL_DIRECTIONS: 0 = up, 1 = down, 2 = left, 3 = right.

(function (root) {
  "use strict";

  const UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3;
  const WEIGHTS = { empty: 1.0, monotonicity: 1.0, smoothness: 0.5, corner: 3.0 };
  const GAME_OVER = -1e9;

  function slideRowLeft(row) {
    const tiles = row.filter(v => v !== 0);
    const out = [];
    let score = 0;
    for (let i = 0; i < tiles.length; i++) {
      if (i + 1 < tiles.length && tiles[i] === tiles[i + 1]) {
        out.push(tiles[i] * 2);
        score += tiles[i] * 2;
        i++;
      } else {
        out.push(tiles[i]);
      }
    }
    while (out.length < 4) out.push(0);
    return { row: out, score };
  }

  // Indices of each line, ordered in the direction tiles slide toward.
  const LINES = [[], [], [], []];
  for (let k = 0; k < 4; k++) {
    LINES[UP].push([0, 1, 2, 3].map(r => 4 * r + k));
    LINES[DOWN].push([3, 2, 1, 0].map(r => 4 * r + k));
    LINES[LEFT].push([0, 1, 2, 3].map(c => 4 * k + c));
    LINES[RIGHT].push([3, 2, 1, 0].map(c => 4 * k + c));
  }

  function move(board, dir) {
    const out = board.slice();
    let score = 0;
    for (const idx of LINES[dir]) {
      const res = slideRowLeft(idx.map(i => board[i]));
      score += res.score;
      idx.forEach((i, j) => { out[i] = res.row[j]; });
    }
    const changed = out.some((v, i) => v !== board[i]);
    return { board: out, score, changed };
  }

  // Same result as move(), plus where every tile travels (for animation).
  // moves: [{from, to, value, merged}], merges: [{at, value}] (value after merging).
  function traceMove(board, dir) {
    const moves = [], merges = [];
    for (const idx of LINES[dir]) {
      const tiles = idx.filter(i => board[i] !== 0);
      let k = 0;
      for (let j = 0; j < tiles.length; j++) {
        const a = tiles[j], to = idx[k];
        if (j + 1 < tiles.length && board[a] === board[tiles[j + 1]]) {
          const b = tiles[j + 1];
          moves.push({ from: a, to, value: board[a], merged: true });
          moves.push({ from: b, to, value: board[b], merged: true });
          merges.push({ at: to, value: board[a] * 2 });
          j++;
        } else {
          moves.push({ from: a, to, value: board[a], merged: false });
        }
        k++;
      }
    }
    return { moves, merges };
  }

  function legalMoves(board) {
    const moves = [];
    for (let d = 0; d < 4; d++) if (move(board, d).changed) moves.push(d);
    return moves;
  }

  function emptyCells(board) {
    const cells = [];
    for (let i = 0; i < 16; i++) if (board[i] === 0) cells.push(i);
    return cells;
  }

  function maxTile(board) { return Math.max(...board); }

  // ---------------------------------------------------------------- heuristic
  const EVAL_LINES = [];
  for (let r = 0; r < 4; r++) EVAL_LINES.push([0, 1, 2, 3].map(c => 4 * r + c));
  for (let c = 0; c < 4; c++) EVAL_LINES.push([0, 1, 2, 3].map(r => 4 * r + c));

  function evaluate(board) {
    const lg = board.map(v => (v ? Math.log2(v) : 0));
    let empty = 0;
    for (const v of board) if (v === 0) empty++;
    let monoPenalty = 0, smooth = 0;
    for (const line of EVAL_LINES) {
      let inc = 0, dec = 0;
      for (let j = 0; j < 3; j++) {
        const a = lg[line[j]], b = lg[line[j + 1]];
        if (a > b) inc += a - b; else dec += b - a;
        if (a && b) smooth += Math.abs(a - b);
      }
      monoPenalty += Math.min(inc, dec);
    }
    const largest = maxTile(board);
    let corner = 0;
    if (largest > 0 && [board[0], board[3], board[12], board[15]].includes(largest)) corner = Math.log2(largest);
    return WEIGHTS.empty * empty + WEIGHTS.monotonicity * -monoPenalty
      + WEIGHTS.smoothness * -smooth + WEIGHTS.corner * corner;
  }

  // ---------------------------------------------------------------- villain
  function place(board, i, v) { const b = board.slice(); b[i] = v; return b; }

  function makeSearch(wantsPlayerToWin, depth) {
    let cache = new Map();
    const ev = b => {
      const key = b.join(",");
      let v = cache.get(key);
      if (v === undefined) { v = evaluate(b); cache.set(key, v); }
      return v;
    };
    const better = wantsPlayerToWin ? (a, b) => a > b : (a, b) => a < b;

    function playerValue(board, d) {
      let bestAfter = null, bestValue = null;
      for (let dir = 0; dir < 4; dir++) {
        const m = move(board, dir);
        if (!m.changed) continue;
        const v = ev(m.board);
        if (bestValue === null || v > bestValue) { bestAfter = m.board; bestValue = v; }
      }
      if (bestAfter === null) return GAME_OVER;
      if (d === 1) return bestValue;
      return spawnerValue(bestAfter, d - 1);
    }

    function spawnerValue(board, d) {
      let best = null;
      for (const i of emptyCells(board)) {
        for (const v of [2, 4]) {
          const val = playerValue(place(board, i, v), d);
          if (best === null || better(val, best)) best = val;
        }
      }
      return best;
    }

    return function chooseSpawn(board) {
      cache = new Map();
      let bestSpawn = null, bestVal = null;
      for (const i of emptyCells(board)) {
        for (const v of [2, 4]) {
          const val = playerValue(place(board, i, v), depth);
          if (bestVal === null || better(val, bestVal)) { bestVal = val; bestSpawn = [i, v]; }
        }
      }
      return bestSpawn;
    };
  }

  function randomSpawn(board, rand) {
    const cells = emptyCells(board);
    const i = cells[Math.floor(rand() * cells.length)];
    return [i, rand() < 0.9 ? 2 : 4];
  }

  // Gentle villain: plays the worst tile with probability `strength`, otherwise a normal random tile.
  function makeVillain(depth, strength, rand) {
    const search = makeSearch(false, depth);
    return function (board) {
      return rand() < strength ? search(board) : randomSpawn(board, rand);
    };
  }

  const api = { UP, DOWN, LEFT, RIGHT, GAME_OVER, slideRowLeft, move, traceMove, legalMoves, emptyCells,
                maxTile, evaluate, makeSearch, makeVillain, randomSpawn };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Game2048 = api;
})(this);
