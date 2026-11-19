"""Build slides/presentation.html: one self-contained file (no internet needed).

Usage:
    python3 -m slides.build
    python3 -m slides.build --team "Dev, Harrison, Maruthi, Tarun"

Inlines the font, the browser game (slides/src/game.js), and every number
on the slides, read from the result CSVs in experiments/results/ and the
Tuplet training log. Re-run it after re-running experiments so the
slides always match the data.
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "experiments", "results")
TD_LOG = os.path.join(REPO, "td", "checkpoints", "training_log.csv")


def font_uri(name: str) -> str:
    with open(os.path.join(HERE, "src", "fonts", name), "rb") as f:
        return "data:font/woff2;base64," + base64.b64encode(f.read()).decode()


def fmt(x) -> str:
    return f"{x:,.0f}"


def line_svg(points, color, baseline=None, width=500, height=385) -> str:
    """Small log-x line chart (training games vs. average score) styled for the deck."""
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
        parts.append(f'<text x="{left - 8}" y="{sy(y) + 5:.1f}" fill="#4a3a2f" font-size="15" font-weight="700" text-anchor="end">{y:,.0f}</text>')
        y += step
    for e, lab in [(2, "100"), (3, "1k"), (4, "10k"), (5, "100k"), (6, "1M")]:
        parts.append(f'<text x="{sx(e):.1f}" y="{height - 14}" fill="#4a3a2f" font-size="15" font-weight="700" text-anchor="middle">{lab}</text>')
    if baseline is not None:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(baseline):.1f}" y2="{sy(baseline):.1f}" '
                     f'stroke="#6a5242" stroke-width="1.5" stroke-dasharray="5 5"/>')
        parts.append(f'<text x="{width - right}" y="{sy(baseline) - 7:.1f}" fill="#4a3a2f" font-size="14" font-weight="700" text-anchor="end">untrained</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(v):.1f}" for i, (x, (_, v)) in enumerate(zip(xs, points)))
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linejoin="round"/>')
    for x, (_, v) in zip(xs, points):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="6" fill="{color}" stroke="#fffdf9" stroke-width="2.5"/>')
    gx, (_, gv) = xs[-1], points[-1]
    parts.append(f'<text x="{sx(gx) - 4:.1f}" y="{sy(gv) - 14:.1f}" fill="#241c16" font-size="17" font-weight="800" '
                 f'text-anchor="end">{gv:,.0f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def td_numbers() -> dict:
    path = os.path.join(RESULTS, "td_learning_curve.csv")
    keys = ["SVG_NORMAL", "SVG_VILLAIN", "TD_TAKEAWAY_SHORT", "TD_GAMES", "TD_GAMES_SHORT", "TD_NORMAL_TO",
            "TD_VILLAIN_TO", "TD_RANDOM_2048", "TD_VILLAIN_2048", "TD_TRAIN_MIN"]
    if not os.path.exists(path) or not os.path.exists(TD_LOG):
        return {k: "(pending)" for k in keys}
    df = pd.read_csv(path)
    log = pd.read_csv(TD_LOG)
    final = df.games_trained.max()
    trained = df[df.games_trained == final]
    normal, villain = trained[trained.spawner == "random"], trained[trained.spawner == "villain"]
    normal_2048 = fmt((normal.max_tile >= 2048).mean() * 100)
    villain_2048 = fmt((villain.max_tile >= 2048).mean() * 100)
    curve = df.groupby(["spawner", "games_trained"]).score.mean()
    pts = lambda s: [(g, v) for g, v in curve[s].items() if g > 0]
    return {
        "SVG_NORMAL": line_svg(pts("random"), "#8a5a3c", baseline=curve["random"].get(0)),
        "SVG_VILLAIN": line_svg(pts("villain"), "#d63a3a", baseline=curve["villain"].get(0)),
        "TD_TAKEAWAY_SHORT": (f"Tuplet reaches 2048 in <b>{normal_2048}%</b> of normal games, "
                              f"<b>{villain_2048}%</b> against the villain."),
        "TD_GAMES": fmt(final),
        "TD_GAMES_SHORT": f"{final / 1e6:g}M" if final >= 1e6 else f"{final / 1e3:g}k",
        "TD_NORMAL_TO": fmt(normal.score.mean()),
        "TD_VILLAIN_TO": fmt(villain.score.mean()),
        "TD_RANDOM_2048": normal_2048,
        "TD_VILLAIN_2048": villain_2048,
        "TD_TRAIN_MIN": fmt(log.elapsed_s.max() / 60),
    }

AGENT_ORDER = ["random", "greedy", "minimax", "expectimax", "td"]
AGENT_NAME = {"random": "Random", "greedy": "Greedy", "minimax": "Minimax", "expectimax": "Expectimax", "td": "Tuplet (trained)"}
SPAWNER_COLOR = {"villain": "#d63a3a", "random": "#3d3a36", "helper": "#2a78d6"}
AGENT_DOT = {"random": "#6b5d52", "greedy": "#2a78d6", "minimax": "#eb6834", "expectimax": "#199e70", "td": "#6250d6"}

# Placeholder numbers shown ONLY until experiments/results/all_agents.csv exists.
# Slides built from these carry a "SAMPLE DATA" banner.
SAMPLE = {
    "random":     {"random": 1100,  "villain": 300,  "villain_moves": 40,  "villain_tile": 32,  "r2048": 0},
    "greedy":     {"random": 11900, "villain": 360,  "villain_moves": 45,  "villain_tile": 32,  "r2048": 12},
    "minimax":    {"random": 17600, "villain": 2100, "villain_moves": 140, "villain_tile": 128, "r2048": 27},
    "expectimax": {"random": 20000, "villain": 400,  "villain_moves": 45,  "villain_tile": 64,  "r2048": 35},
    "td":         {"random": 40600, "villain": 690,  "villain_moves": 68,  "villain_tile": 64,  "r2048": 83},
}


def agent_table():
    """{agent: {...}} from all_agents.csv, or (SAMPLE, True) if results are not ready."""
    path = os.path.join(RESULTS, "all_agents.csv")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        have = set(zip(df.agent, df.spawner))
        if all((a, s) in have for a in AGENT_ORDER for s in ("random", "villain")):
            out = {}
            for a in AGENT_ORDER:
                n, v = df[(df.agent == a) & (df.spawner == "random")], df[(df.agent == a) & (df.spawner == "villain")]
                h = df[(df.agent == a) & (df.spawner == "helper")]
                out[a] = {"random": n.score.mean(), "villain": v.score.mean(), "villain_moves": v.moves.mean(),
                          "villain_tile": v.max_tile.mode().max(), "r2048": (n.max_tile >= 2048).mean() * 100}
                if len(h):
                    out[a]["helper"] = h.score.mean()
            return out, False
    return SAMPLE, True


def dumbbell_svg(table, width=1048, height=362) -> str:
    """Log-scale dot plot: each agent's score against the villain, in the normal game, and (if run) with the helper."""
    left, right, top, bottom = 190, 24, 6, 34
    vals = [v[k] for v in table.values() for k in ("random", "villain", "helper") if k in v]
    lo, hi = math.log10(min(vals) / 1.45), math.log10(max(vals) * 2.05)
    sx = lambda v: left + (math.log10(v) - lo) / (hi - lo) * (width - left - right)
    rows = [a for a in AGENT_ORDER if a in table]
    step = (height - top - bottom) / len(rows)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for e in range(int(math.ceil(lo)), int(math.floor(hi)) + 1):
        x = sx(10 ** e)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height - bottom}" stroke="#eadfce" stroke-width="1.2"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" fill="#4a3a2f" font-size="16" font-weight="700" '
                     f'text-anchor="middle">{10 ** e:,}</text>')
    for i, a in enumerate(rows):
        y = top + step * (i + 0.5)
        d = table[a]
        keys = [k for k in ("villain", "random", "helper") if k in d]
        xs = [sx(d[k]) for k in keys]
        parts.append(f'<text x="0" y="{y + 7:.1f}" fill="#4a3a2f" font-size="23" font-weight="600" '
                     f'text-anchor="start">{AGENT_NAME[a]}</text>')
        parts.append(f'<line x1="{min(xs):.1f}" x2="{max(xs):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e7d8c3" '
                     f'stroke-width="11" stroke-linecap="round"/>')
        for k, x in zip(keys, xs):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12.5" fill="{SPAWNER_COLOR[k]}" stroke="#fffdf9" stroke-width="3"/>')
        parts.append(f'<text x="{sx(d["villain"]) - 19:.1f}" y="{y + 6:.1f}" fill="{SPAWNER_COLOR["villain"]}" '
                     f'font-size="18" font-weight="900" text-anchor="end">{d["villain"]:,.0f}</text>')
        top_key = "helper" if "helper" in d else "random"
        parts.append(f'<text x="{sx(d[top_key]) + 19:.1f}" y="{y + 6:.1f}" fill="#241c16" font-size="18" '
                     f'font-weight="900">{d[top_key]:,.0f}</text>')
        if top_key == "helper":
            parts.append(f'<text x="{sx(d["random"]):.1f}" y="{y - 19:.1f}" fill="#3d3a36" font-size="15" '
                         f'font-weight="800" text-anchor="middle">{d["random"]:,.0f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def agents_numbers() -> dict:
    table, sample = agent_table()
    best_n = max(table, key=lambda a: table[a]["random"])
    best_v = max(table, key=lambda a: table[a]["villain_moves"])
    drops = [1 - table[a]["villain"] / table[a]["random"] for a in table if a != "random"]
    legend = "".join(f'<span><i style="background:{SPAWNER_COLOR[k]}"></i>{lab}</span>'
                     for k, lab in [("villain", "Against the villain"), ("random", "Normal game"), ("helper", "With the helper (20 games)")]
                     if k != "helper" or any("helper" in v for v in table.values()))
    rows = "\n".join(
        f'        <tr><td><span class="who"><i style="background:{AGENT_DOT[a]}"></i>{AGENT_NAME[a]}'
        f'{"<small>1M training games</small>" if a == "td" else ""}</span></td>'
        f'<td class="n">{fmt(table[a]["villain_moves"])}</td>'
        f'<td class="n"><span class="t tchip t{int(table[a]["villain_tile"])}">{int(table[a]["villain_tile"])}</span></td>'
        f'<td class="n">{fmt(table[a]["villain"])}</td></tr>'
        for a in AGENT_ORDER if a in table)
    short = {**AGENT_NAME, "td": "Tuplet"}
    return {
        "SAMPLE_BANNER": '<div class="sample-banner">SAMPLE DATA: final numbers pending</div>' if sample else "",
        "SVG_AGENTS": dumbbell_svg(table),
        "AGENT_LEGEND": legend,
        "AGENT_ROWS": rows,
        "BEST_NORMAL_SHORT": short[best_n],
        "BEST_NORMAL_SCORE": fmt(table[best_n]["random"]),
        "BEST_NORMAL_VILLAIN": fmt(table[best_n]["villain"]),
        "BEST_VILLAIN_NAME": AGENT_NAME[best_v],
        "BEST_VILLAIN_MOVES": fmt(table[best_v]["villain_moves"]),
        "TAKE_COLLAPSE": f"Every smart agent loses <b>{min(drops) * 100:.0f}%+</b> of its score to the villain.",
        "TAKE_SURVIVOR": f"<b>{short[best_v]}</b>, which plans for the worst, survives longest.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the self-contained presentation.")
    parser.add_argument("--team", default="Dev, Harrison, Maruthi, Tarun")
    parser.add_argument("--out", default=os.path.join(HERE, "presentation.html"))
    args = parser.parse_args()

    with open(os.path.join(HERE, "src", "template.html")) as f:
        html = f.read()
    with open(os.path.join(HERE, "src", "game.js")) as f:
        game_js = f.read()

    values = {
        "TEAM": args.team,
        "GAME_JS": game_js,
        "FONT_NUNITO": font_uri("Nunito-latin.woff2"),
        **agents_numbers(),
        **td_numbers(),
    }
    html = re.sub(r"\{\{(\w+)\}\}", lambda m: str(values[m.group(1)]), html)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote {os.path.relpath(args.out, REPO)} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
