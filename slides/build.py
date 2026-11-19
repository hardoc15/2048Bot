"""Build slides/presentation.html: one self-contained file (no internet needed).

Usage:
    python3 -m slides.build
    python3 -m slides.build --team "Name 1, Name 2, Name 3, Name 4" --gentle 0.8

Inlines the browser game (slides/src/game.js), the figures from
experiments/figures/, and every number on the slides, read from the result
CSVs in experiments/results/. Re-run it after re-running experiments so the
slides always match the data.
"""

from __future__ import annotations

import argparse
import base64
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "experiments", "results")
FIGURES = os.path.join(REPO, "experiments", "figures")
TD_LOG = os.path.join(REPO, "td", "checkpoints", "training_log.csv")


def fmt(x) -> str:
    return f"{x:,.0f}"


def data_uri(name: str) -> str:
    path = os.path.join(FIGURES, name)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def most_common(series) -> str:
    return fmt(series.mode().max())


def line_svg(points, color, baseline=None, width=500, height=290) -> str:
    """Small log-x line chart (training games vs. average score) styled for the deck."""
    import math
    left, right, top, bottom = 62, 18, 22, 40
    xs = [math.log10(g) for g, _ in points]
    x0, x1 = 2.0, 6.0  # 100 .. 1,000,000 games
    ymax = max(v for _, v in points) * 1.18
    step = 10 ** math.floor(math.log10(ymax / 4))
    step *= next(m for m in (1, 2, 2.5, 5, 10) if ymax / (step * m) <= 5)
    sx = lambda x: left + (x - x0) / (x1 - x0) * (width - left - right)
    sy = lambda y: height - bottom - y / ymax * (height - top - bottom)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    y = 0
    while y <= ymax:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="#eadfce" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{sy(y) + 5:.1f}" fill="#8f8174" font-size="14" text-anchor="end">{y:,.0f}</text>')
        y += step
    for e, lab in [(2, "100"), (3, "1k"), (4, "10k"), (5, "100k"), (6, "1M")]:
        parts.append(f'<text x="{sx(e):.1f}" y="{height - 14}" fill="#8f8174" font-size="14" text-anchor="middle">{lab}</text>')
    if baseline is not None:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(baseline):.1f}" y2="{sy(baseline):.1f}" '
                     f'stroke="#8f8174" stroke-width="1.5" stroke-dasharray="5 5"/>')
        parts.append(f'<text x="{width - right}" y="{sy(baseline) - 7:.1f}" fill="#8f8174" font-size="13" text-anchor="end">untrained</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(v):.1f}" for i, (x, (_, v)) in enumerate(zip(xs, points)))
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linejoin="round"/>')
    for x, (_, v) in zip(xs, points):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="6" fill="{color}" stroke="#fffdf9" stroke-width="2.5"/>')
    gx, (_, gv) = xs[-1], points[-1]
    parts.append(f'<text x="{sx(gx) - 4:.1f}" y="{sy(gv) - 14:.1f}" fill="#241c16" font-size="17" font-weight="800" '
                 f'text-anchor="end">{gv:,.0f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def greedy_numbers() -> dict:
    df = pd.read_csv(os.path.join(RESULTS, "greedy_vs_spawners.csv"))
    pick = lambda s, d: df[(df.spawner == s) & (df.depth == d)]
    villain, helper, normal = pick("villain", 2), pick("helper", 2), pick("random", 0)
    bar = lambda v: max(12, round(760 * v / helper.score.mean()))
    return {
        "BAR_VILLAIN": bar(villain.score.mean()),
        "BAR_RANDOM": bar(normal.score.mean()),
        "BAR_HELPER": bar(helper.score.mean()),
        "GREEDY_VILLAIN": fmt(villain.score.mean()),
        "GREEDY_VILLAIN_TILE": most_common(villain.max_tile),
        "GREEDY_RANDOM": fmt(normal.score.mean()),
        "GREEDY_HELPER": fmt(helper.score.mean()),
        "GREEDY_VILLAIN_D1": fmt(pick("villain", 1).score.mean()),
        "GREEDY_HELPER_D1": fmt(pick("helper", 1).score.mean()),
        "GREEDY_2048": fmt((normal.max_tile >= 2048).mean() * 100),
        "SWING": fmt(helper.score.mean() / villain.score.mean()),
    }


def td_numbers() -> dict:
    path = os.path.join(RESULTS, "td_learning_curve.csv")
    if not os.path.exists(path) or not os.path.exists(TD_LOG):
        missing = "(pending)"
        return {k: missing for k in ["SVG_NORMAL", "SVG_VILLAIN", "TD_TAKEAWAY_SHORT", "TD_DISTINCT", "TD_GAMES", "TD_RANDOM", "TD_RANDOM_2048", "TD_RANDOM_TILE", "TD_VILLAIN",
                                     "TD_VILLAIN_TILE", "TD_TRAIN_MIN", "TD_VILLAIN_SENTENCE", "TD_TAKEAWAY"]}
    df = pd.read_csv(path)
    log = pd.read_csv(TD_LOG)
    final = df.games_trained.max()
    trained = df[df.games_trained == final]
    first = df[df.games_trained == df[df.games_trained > 0].games_trained.min()]
    normal, villain = trained[trained.spawner == "random"], trained[trained.spawner == "villain"]
    early_normal = first[first.spawner == "random"].score.mean()
    early_villain = first[first.spawner == "villain"].score.mean()
    n_first = int(first.games_trained.iloc[0])
    normal_gain = normal.score.mean() / early_normal
    villain_gain = villain.score.mean() / early_villain
    sentence = (f"From {n_first:,} to {final:,} training games, the bot's normal-game score grew "
                f"{normal_gain:.1f}× ({fmt(early_normal)} → {fmt(normal.score.mean())}). Against the villain it went "
                f"from {fmt(early_villain)} to {fmt(villain.score.mean())} ({villain_gain:.1f}×).")
    takeaway = (f"The TD bot's normal-game score grew {normal_gain:.1f}× with training, "
                f"but against the villain only {villain_gain:.1f}×.")
    distinct = df[(df.spawner == "villain") & (df.games_trained > 0)].groupby("games_trained").score.nunique()
    curve = df.groupby(["spawner", "games_trained"]).score.mean()
    pts = lambda s: [(g, v) for g, v in curve[s].items() if g > 0]
    return {
        "SVG_NORMAL": line_svg(pts("random"), "#2a78d6", baseline=curve["random"].get(0)),
        "SVG_VILLAIN": line_svg(pts("villain"), "#d63a3a", baseline=curve["villain"].get(0)),
        "TD_TAKEAWAY_SHORT": (f"training grew the bot's normal score {normal_gain:.1f}×, "
                              f"its score vs. the villain only {villain_gain:.1f}×."),
        "TD_DISTINCT": f"{distinct.min()}–{distinct.max()}",
        "TD_GAMES": fmt(final),
        "TD_RANDOM": fmt(normal.score.mean()),
        "TD_RANDOM_2048": fmt((normal.max_tile >= 2048).mean() * 100),
        "TD_RANDOM_TILE": most_common(normal.max_tile),
        "TD_VILLAIN": fmt(villain.score.mean()),
        "TD_VILLAIN_TILE": most_common(villain.max_tile),
        "TD_TRAIN_MIN": fmt(log.elapsed_s.max() / 60),
        "TD_VILLAIN_SENTENCE": sentence,
        "TD_TAKEAWAY": takeaway,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the self-contained presentation.")
    parser.add_argument("--team", default="Team names TBD")
    parser.add_argument("--gentle", type=float, default=0.8, help="demo villain strength: share of tiles it places (0-1)")
    parser.add_argument("--out", default=os.path.join(HERE, "presentation.html"))
    args = parser.parse_args()

    with open(os.path.join(HERE, "src", "template.html")) as f:
        html = f.read()
    with open(os.path.join(HERE, "src", "game.js")) as f:
        game_js = f.read()

    values = {
        "TEAM": args.team,
        "GENTLE": repr(args.gentle),
        "GENTLE_PCT": fmt(args.gentle * 100),
        "GAME_JS": game_js,
        "FIG_SPAWNERS": data_uri("fig7_spawner_scores.png"),
        "FIG_BOARDS": data_uri("fig9_spawner_final_boards.png"),
        "FIG_TRAINING": data_uri("fig10_td_training_curve.png"),
        "FIG_TD_CURVE": data_uri("fig11_td_vs_villain_curve.png"),
        **greedy_numbers(),
        **td_numbers(),
    }
    html = re.sub(r"\{\{(\w+)\}\}", lambda m: str(values[m.group(1)]), html)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote {os.path.relpath(args.out, REPO)} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
