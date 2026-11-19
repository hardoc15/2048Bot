"""Build slides/presentation.html: one self-contained file (no internet needed).

Usage:
    python3 -m slides.build
    python3 -m slides.build --team "Dev, Harrison, Maruthi, Tarun" --gentle 1.0

Inlines the browser game (slides/src/game.js), the figures from
experiments/figures/, and every number on the slides, read from the result
CSVs in experiments/results/. Re-run it after re-running experiments so the
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
FIGURES = os.path.join(REPO, "experiments", "figures")
TD_LOG = os.path.join(REPO, "td", "checkpoints", "training_log.csv")


def font_uri(name: str) -> str:
    with open(os.path.join(HERE, "src", "fonts", name), "rb") as f:
        return "data:font/woff2;base64," + base64.b64encode(f.read()).decode()


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


def line_svg(points, color, baseline=None, width=500, height=330) -> str:
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
        "GREEDY_D1_DROP": fmt((1 - pick("villain", 1).score.mean() / normal.score.mean()) * 100),
        "GREEDY_HELPER_D1": fmt(pick("helper", 1).score.mean()),
        "GREEDY_2048": fmt((normal.max_tile >= 2048).mean() * 100),
        "SWING": fmt(helper.score.mean() / villain.score.mean()),
    }


def td_numbers() -> dict:
    path = os.path.join(RESULTS, "td_learning_curve.csv")
    if not os.path.exists(path) or not os.path.exists(TD_LOG):
        missing = "(pending)"
        return {k: missing for k in ["TD_GAMES_SHORT", "TD_NORMAL_GAIN", "TD_VILLAIN_GAIN", "TD_NORMAL_FROM", "TD_NORMAL_TO", "TD_VILLAIN_FROM", "TD_VILLAIN_TO", "SVG_NORMAL", "SVG_VILLAIN", "TD_TAKEAWAY_SHORT", "TD_DISTINCT", "TD_GAMES", "TD_RANDOM", "TD_RANDOM_2048", "TD_RANDOM_TILE", "TD_VILLAIN",
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
                f"{normal_gain:.1f}x ({fmt(early_normal)} to {fmt(normal.score.mean())}). Against the villain it went "
                f"from {fmt(early_villain)} to {fmt(villain.score.mean())} ({villain_gain:.1f}x).")
    takeaway = (f"Tuplet's normal-game score grew {normal_gain:.1f}x with training, "
                f"but against the villain only {villain_gain:.1f}x.")
    distinct = df[(df.spawner == "villain") & (df.games_trained > 0)].groupby("games_trained").score.nunique()
    curve = df.groupby(["spawner", "games_trained"]).score.mean()
    pts = lambda s: [(g, v) for g, v in curve[s].items() if g > 0]
    return {
        "SVG_NORMAL": line_svg(pts("random"), "#8a5a3c", baseline=curve["random"].get(0)),
        "SVG_VILLAIN": line_svg(pts("villain"), "#d63a3a", baseline=curve["villain"].get(0)),
        "TD_TAKEAWAY_SHORT": f"Training made Tuplet <b>{normal_gain:.0f}x</b> better, but not against the villain.",
        "TD_DISTINCT": f"{distinct.min()} to {distinct.max()}",
        "TD_GAMES": fmt(final),
        "TD_GAMES_SHORT": f"{final / 1e6:g}M" if final >= 1e6 else f"{final / 1e3:g}k",
        "TD_NORMAL_GAIN": f"{normal_gain:.1f}",
        "TD_VILLAIN_GAIN": f"{villain_gain:.1f}",
        "TD_NORMAL_FROM": fmt(early_normal), "TD_NORMAL_TO": fmt(normal.score.mean()),
        "TD_VILLAIN_FROM": fmt(early_villain), "TD_VILLAIN_TO": fmt(villain.score.mean()),
        "TD_RANDOM": fmt(normal.score.mean()),
        "TD_RANDOM_2048": fmt((normal.max_tile >= 2048).mean() * 100),
        "TD_RANDOM_TILE": most_common(normal.max_tile),
        "TD_VILLAIN": fmt(villain.score.mean()),
        "TD_VILLAIN_TILE": most_common(villain.max_tile),
        "TD_TRAIN_MIN": fmt(log.elapsed_s.max() / 60),
        "TD_VILLAIN_SENTENCE": sentence,
        "TD_TAKEAWAY": takeaway,
    }



AGENT_ORDER = ["random", "greedy", "minimax", "expectimax", "td"]
AGENT_NAME = {"random": "Random", "greedy": "Greedy", "minimax": "Minimax", "expectimax": "Expectimax", "td": "Tuplet (trained)"}
SPAWNER_COLOR = {"villain": "#d63a3a", "random": "#3d3a36", "helper": "#2a78d6"}

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


def dumbbell_svg(table, width=1040, height=340) -> str:
    """Log-scale dot plot: each agent's score against the villain, in the normal game, and (if run) with the helper."""
    left, right, top, bottom = 190, 30, 22, 40
    vals = [v[k] for v in table.values() for k in ("random", "villain", "helper") if k in v]
    lo, hi = math.log10(min(vals) / 1.45), math.log10(max(vals) * 1.7)
    sx = lambda v: left + (math.log10(v) - lo) / (hi - lo) * (width - left - right)
    rows = [a for a in AGENT_ORDER if a in table]
    step = (height - top - bottom) / len(rows)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for e in range(int(math.ceil(lo)), int(math.floor(hi)) + 1):
        x = sx(10 ** e)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height - bottom}" stroke="#eadfce" stroke-width="1.2"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 14}" fill="#4a3a2f" font-size="15" font-weight="700" '
                     f'text-anchor="middle">{10 ** e:,}</text>')
    for i, a in enumerate(rows):
        y = top + step * (i + 0.5)
        d = table[a]
        keys = [k for k in ("villain", "random", "helper") if k in d]
        xs = [sx(d[k]) for k in keys]
        parts.append(f'<text x="0" y="{y + 7:.1f}" fill="#4a3a2f" font-size="20" font-weight="600" '
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
    others = [table[a]["villain_moves"] for a in table if a != best_v]
    drops = [1 - table[a]["villain"] / table[a]["random"] for a in table if a != "random"]
    legend = "".join(f'<span><i style="background:{SPAWNER_COLOR[k]}"></i>{lab}</span>'
                     for k, lab in [("villain", "Against the villain"), ("random", "Normal game"), ("helper", "With the helper (20 games)")]
                     if k != "helper" or any("helper" in v for v in table.values()))
    agent_dot = {"random": "#6b5d52", "greedy": "#2a78d6", "minimax": "#eb6834", "expectimax": "#199e70", "td": "#6250d6"}
    rows = "\n".join(
        f'        <tr><td><span class="who"><i style="background:{agent_dot[a]}"></i>{AGENT_NAME[a]}'
        f'{"<small>1M training games</small>" if a == "td" else ""}</span></td>'
        f'<td class="n">{fmt(table[a]["villain_moves"])}</td>'
        f'<td class="n"><span class="t tchip t{int(table[a]["villain_tile"])}">{int(table[a]["villain_tile"])}</span></td>'
        f'<td class="n">{fmt(table[a]["villain"])}</td></tr>'
        for a in AGENT_ORDER if a in table)
    dot = {"random": "#6b5d52", "greedy": "#2a78d6", "minimax": "#eb6834", "expectimax": "#199e70", "td": "#6250d6"}
    short = {**AGENT_NAME, "td": "Tuplet"}

    def ranking(key):
        order = sorted(table, key=lambda a: -table[a][key])
        return "".join(f'<span class="rchip{" first" if i == 0 else ""}"><span class="n">{i + 1}</span>'
                       f'<i style="background:{dot[a]}"></i>{short[a]}</span>' for i, a in enumerate(order))

    def slope_svg(width=1060, height=250):
        """Rank in the normal game (left) vs. against the villain (right), one line per agent."""
        n_rank = {a: i for i, a in enumerate(sorted(table, key=lambda a: -table[a]["random"]))}
        v_rank = {a: i for i, a in enumerate(sorted(table, key=lambda a: -table[a]["villain"]))}
        movers = {a for a in table if n_rank[a] != v_rank[a]}
        top, step, xl, xr = 52, 38, 330, 730
        y = lambda r: top + r * step
        parts = [f'<svg viewBox="0 0 {width} {height}" role="img">',
                 f'<text x="{xl}" y="20" fill="#241c16" font-size="20" font-weight="900" text-anchor="end">Normal game</text>',
                 f'<text x="{xr}" y="20" fill="#d63a3a" font-size="20" font-weight="900">Vs. the villain</text>',
                 f'<text x="{(xl + xr) / 2}" y="20" fill="#3f322a" font-size="15" font-weight="800" text-anchor="middle">RANK BY AVERAGE SCORE</text>']
        for a in sorted(table, key=lambda a: a in movers):  # draw highlighted lines last
            hot = a in movers
            col = dot[a] if hot else "#d9c9b3"
            parts.append(f'<line x1="{xl + 22}" y1="{y(n_rank[a])}" x2="{xr - 22}" y2="{y(v_rank[a])}" stroke="{col}" '
                         f'stroke-width="{6 if hot else 3}" stroke-linecap="round"/>')
        for a in table:
            hot = a in movers
            col = dot[a] if hot else "#b9a993"
            wt = 900 if hot else 700
            ink = "#241c16" if hot else "#5a4636"
            for side, r in (("l", n_rank[a]), ("r", v_rank[a])):
                cx = xl + 10 if side == "l" else xr - 10
                parts.append(f'<circle cx="{cx}" cy="{y(r)}" r="9" fill="{col}" stroke="#fffdf9" stroke-width="3"/>')
                label = f"{r + 1}. {short[a]}"
                if side == "l":
                    parts.append(f'<text x="{xl - 10}" y="{y(r) + 7}" fill="{ink}" font-size="20" font-weight="{wt}" text-anchor="end">{label}</text>')
                else:
                    parts.append(f'<text x="{xr + 10}" y="{y(r) + 7}" fill="{ink}" font-size="20" font-weight="{wt}">{label}</text>')
        parts.append("</svg>")
        return "".join(parts)

    return {
        "SVG_SLOPE": slope_svg(),
        "BEST_NORMAL_SHORT": short[best_n],
        "BEST_NORMAL_VILLAIN": fmt(table[best_n]["villain"]),
        "RANK_NORMAL": ranking("random"),
        "RANK_VILLAIN": ranking("villain"),
        "SAMPLE_BANNER": '<div class="sample-banner">SAMPLE DATA: final numbers pending</div>' if sample else "",
        "SVG_AGENTS": dumbbell_svg(table),
        "AGENT_LEGEND": legend,
        "AGENT_ROWS": rows,
        "BEST_NORMAL_NAME": AGENT_NAME[best_n],
        "BEST_NORMAL_SCORE": fmt(table[best_n]["random"]),
        "BEST_NORMAL_2048": fmt(table[best_n]["r2048"]),
        "BEST_VILLAIN_NAME": AGENT_NAME[best_v],
        "BEST_VILLAIN_MOVES": fmt(table[best_v]["villain_moves"]),
        "BEST_VILLAIN_SCORE": fmt(table[best_v]["villain"]),
        "TAKE_COLLAPSE": f"Every smart agent loses <b>{min(drops) * 100:.0f}%+</b> of its score to the villain.",
        "TAKE_SURVIVOR": f"<b>{short[best_v]}</b>, which plans for the worst, survives longest.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the self-contained presentation.")
    parser.add_argument("--team", default="Dev, Harrison, Maruthi, Tarun")
    parser.add_argument("--gentle", type=float, default=1.0, help="demo villain strength: share of tiles it places (0-1); 1.0 = full force")
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
        "FONT_NUNITO": font_uri("Nunito-latin.woff2"),
        "FIG_SPAWNERS": data_uri("fig7_spawner_scores.png"),
        "FIG_BOARDS": data_uri("fig9_spawner_final_boards.png"),
        "FIG_TRAINING": data_uri("fig10_td_training_curve.png"),
        "FIG_TD_CURVE": data_uri("fig11_td_vs_villain_curve.png"),
        **greedy_numbers(),
        **agents_numbers(),
        **td_numbers(),
    }
    html = re.sub(r"\{\{(\w+)\}\}", lambda m: str(values[m.group(1)]), html)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote {os.path.relpath(args.out, REPO)} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
