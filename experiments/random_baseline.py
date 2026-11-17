"""Run the seeded RandomAgent baseline and generate a two-panel SVG figure."""

from collections import Counter
from pathlib import Path
from statistics import mean

from agents.random_agent import RandomAgent
from game import play_games


def draw_histogram(x, title, axis_label, labels, counts, tick_step):
    width = 490
    top, bottom = 180, 395
    maximum = max(tick_step, ((max(counts) + tick_step - 1) // tick_step) * tick_step)
    step = width / len(counts)
    parts = [
        f'<text x="{x}" y="133" class="panel">{title}</text>',
        f'<text x="{x}" y="157" class="label">Games</text>',
    ]
    for tick in range(0, maximum + 1, tick_step):
        y = bottom - (bottom - top) * tick / maximum
        parts.extend([
            f'<line x1="{x}" x2="{x + width}" y1="{y}" y2="{y}" stroke="#cbd5e1"/>',
            f'<text x="{x - 9}" y="{y + 5}" text-anchor="end" class="label">{tick}</text>',
        ])
    for index, (label, count) in enumerate(zip(labels, counts)):
        center = x + (index + 0.5) * step
        height = (bottom - top) * count / maximum
        y = bottom - height
        parts.extend([
            f'<rect x="{center - step * 0.32}" y="{y}" width="{step * 0.64}" '
            f'height="{height}" fill="#146b70"/>',
            f'<text x="{center}" y="{y - 7}" text-anchor="middle" class="value">{count}</text>',
            f'<text x="{center}" y="{bottom + 24}" text-anchor="middle" class="label">{label}</text>',
        ])
    parts.append(
        f'<text x="{x + width / 2}" y="449" text-anchor="middle" class="label">{axis_label}</text>'
    )
    return parts


def main():
    games = play_games(RandomAgent(seed=0), seeds=list(range(100)))
    average_score = mean(game.score for game in games)
    successes = sum(game.reached_2048 for game in games)
    score_counts = Counter(game.score // 400 for game in games)
    bins = range(max(score_counts) + 1)
    tile_counts = Counter(game.max_tile for game in games)
    tiles = sorted(tile_counts)

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="480" '
        'viewBox="0 0 1200 480" role="img" aria-labelledby="title description">',
        '<title id="title">2048 random baseline</title>',
        '<desc id="description">Histograms of final scores and highest tiles over 100 games.</desc>',
        '<style>text{font-family:Arial,sans-serif;fill:#14253b}'
        '.title{font-size:34px;font-weight:bold}.summary{font-size:18px}'
        '.panel{font-size:22px;font-weight:bold}.label{font-size:15px}'
        '.value{font-size:15px;font-weight:bold}</style>',
        '<rect width="1200" height="480" fill="white"/>',
        '<text x="56" y="49" class="title">2048 random baseline</text>',
        f'<text x="56" y="80" class="summary">100 games · Mean score {average_score:,.2f}'
        f' · {successes}/100 reached 2048 · Agent seed 0, game seeds 0–99</text>',
    ]
    svg.extend(draw_histogram(
        56, "Final score distribution", "Final score (400-point bins)",
        [f"{index * 400}–{index * 400 + 399}" for index in bins],
        [score_counts[index] for index in bins], 10,
    ))
    svg.extend(draw_histogram(
        650, "Highest tile distribution", "Highest tile reached in each game",
        tiles, [tile_counts[tile] for tile in tiles], 10,
    ))
    svg.append("</svg>")
    output = Path(__file__).resolve().parents[1] / "slides" / "random-baseline.svg"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"Score bins: {sorted(score_counts.items())}")
    print(f"Highest tiles: {sorted(tile_counts.items())}")
    print(f"Figure: {output}")


if __name__ == "__main__":
    main()
