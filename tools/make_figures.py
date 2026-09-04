"""Render the static result figures in results/figures/ from results/tables/.

Run with:  python tools/make_figures.py

Every figure is written twice: SVG (for the docs site) and PNG at 200 dpi
(for the README and the slide deck).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

# Categorical slots 1-4 of the validated default palette (light surface).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL = "#c9c8c2"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#84837c"
GRID = "#e8e7e2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "svg.fonttype": "none",
})

CORNER_PX = 4.0  # rounded data-end radius, in device pixels


def read_table(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _data_radius(ax, axis: str) -> float:
    """CORNER_PX expressed in data units along *axis*, so corners read as circular."""
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    if axis == "x":
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        return CORNER_PX / max(bbox.width, 1.0) * span
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    return CORNER_PX / max(bbox.height, 1.0) * span


def rounded_bar(ax, x, height, width, color, *, horizontal=False, base=0.0):
    """A bar with rounded corners on the data end and a square end on the baseline."""
    if horizontal:
        rx, ry = _data_radius(ax, "x"), _data_radius(ax, "y")
        length = height
        r = min(rx, ry, abs(length) / 2, width / 2)
        y0, y1 = x - width / 2, x + width / 2
        x0, x1 = base, base + length
        verts = [
            (x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r),
            (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0),
        ]
        codes = [
            MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY,
        ]
    else:
        rx, ry = _data_radius(ax, "x"), _data_radius(ax, "y")
        r_x = min(rx, width / 2)
        r_y = min(ry, abs(height) / 2) if height else 0.0
        x0, x1 = x - width / 2, x + width / 2
        y0, y1 = base, base + height
        verts = [
            (x0, y0), (x0, y1 - r_y), (x0, y1), (x0 + r_x, y1),
            (x1 - r_x, y1), (x1, y1), (x1, y1 - r_y), (x1, y0), (x0, y0),
        ]
        codes = [
            MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY,
        ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, linewidth=0))


def headline(ax, title: str, subtitle: str) -> None:
    """Bold title above a muted subtitle, both flush-left over the plot area."""
    ax.set_title(title, fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=34)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=INK_MUTED, va="bottom", ha="left")


def style_axes(ax, *, ymax: float = 1.0) -> None:
    """Recessive axes: horizontal grid only, no top/right/left spines."""
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelsize=9)


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {FIGURES / stem}.svg / .png")


# ── 1. retrieval metrics on the full PAR4PC validation split ──────────────────
def figure_retrieval() -> None:
    rows = read_table("retrieval_validation_full.csv")
    metrics = [("hit@1", "Hit@1"), ("hit@3", "Hit@3"), ("recall@3", "Recall@3"), ("exact@gold", "Exact@|gold|")]
    colors = [BLUE, ORANGE, AQUA]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    style_axes(ax, ymax=1.08)
    ax.set_xlim(-0.55, len(metrics) - 0.45)
    group_width = 0.72
    bar_width = group_width / 3 - 0.035  # surface gap between adjacent bars

    for series_index, (row, color) in enumerate(zip(rows, colors)):
        offset = -group_width / 2 + group_width / 3 * (series_index + 0.5)
        for metric_index, (key, _) in enumerate(metrics):
            x = metric_index + offset
            height = float(row[key])
            rounded_bar(ax, x, height, bar_width, color)
            # relief for the sub-3:1 slots: every bar carries its own value
            ax.text(x, height + 0.016, f"{height:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK_SOFT)
        ax.add_patch(Rectangle((0, 0), 0, 0, color=color,
                               label=f"{row['method']}  ·  {row['label']}"))

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], fontsize=10.5, color=INK)
    headline(ax, "Prior-art ranking on PANORAMA PAR4PC",
             "Full validation split · 3,029 cases · 8 candidates (A–H) per case")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=3, handlelength=1.0, handleheight=1.0, borderpad=0, columnspacing=1.6)
    save(fig, "retrieval_par4pc_validation")


# ── 2. human product-QA study: overall score ──────────────────────────────────
def figure_human_overall() -> None:
    rows = sorted(read_table("human_eval_summary.csv"), key=lambda r: float(r["overall"]))
    labels = [r["system_name"] for r in rows]
    values = [float(r["overall"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlim(0, 1.06)
    for index, (label, value) in enumerate(zip(labels, values)):
        is_ours = label == "Our Agent"
        rounded_bar(ax, index, value, 0.5, BLUE if is_ours else NEUTRAL, horizontal=True)
        ax.text(value + 0.012, index, f"{value:.3f}", va="center", ha="left", fontsize=10,
                color=INK, fontweight="bold" if is_ours else "normal")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelsize=9)
    headline(ax, "Overall human-rated answer quality",
             "20 product-QA prompts per system · mean of groundedness, helpfulness, "
             "hallucination-freedom, context reuse")
    save(fig, "human_eval_overall")


# ── 3. human product-QA study: per-aspect breakdown ───────────────────────────
def figure_human_aspects() -> None:
    by_system = {r["system_name"]: r for r in read_table("human_eval_summary.csv")}
    order = ["Our Agent", "RAG only", "ChatGPT Auto", "Gemini Fast"]
    aspects = [
        ("grounded", "Groundedness"),
        ("helpful", "Helpfulness"),
        ("hallucination_reliability", "Hallucination-free"),
        ("context_reuse", "Context reuse"),
    ]
    colors = [BLUE, ORANGE, AQUA, YELLOW]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    style_axes(ax, ymax=1.10)
    ax.set_xlim(-0.55, len(aspects) - 0.45)
    group_width = 0.76
    bar_width = group_width / 4 - 0.03

    for series_index, (name, color) in enumerate(zip(order, colors)):
        offset = -group_width / 2 + group_width / 4 * (series_index + 0.5)
        for aspect_index, (key, _) in enumerate(aspects):
            x = aspect_index + offset
            height = float(by_system[name][key])
            rounded_bar(ax, x, height, bar_width, color)
            ax.text(x, height + 0.016, f"{height:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK_SOFT)
        ax.add_patch(Rectangle((0, 0), 0, 0, color=color, label=name))

    ax.set_xticks(range(len(aspects)))
    ax.set_xticklabels([label for _, label in aspects], fontsize=10.5, color=INK)
    headline(ax, "Where the agent wins, and where it does not",
             "Human-rated score per aspect · 1.0 = all 20 answers earned the top label")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=4, handlelength=1.0, handleheight=1.0, borderpad=0, columnspacing=1.8)
    save(fig, "human_eval_aspects")


# ── 4. label distribution (ordinal: top / middle / bottom) ────────────────────
def figure_label_distribution() -> None:
    rows = read_table("human_eval_label_counts.csv")
    aspects = ["grounded", "helpful", "hallucination", "context reuse"]
    aspect_titles = ["Groundedness", "Helpfulness", "Hallucination-free", "Context reuse"]
    order = ["Our Agent", "RAG only", "ChatGPT Auto", "Gemini Fast"]
    lookup = {(r["system_name"], r["aspect"]): r for r in rows}

    # Ordinal ramp: one hue, best → worst; lightest step clears the 2:1 ordinal floor.
    ramp = ["#184f95", "#3987e5", "#9ec5f4"]
    level_names = ["Top label", "Middle", "Bottom label"]

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5), sharey=True)
    for ax, aspect, title in zip(axes, aspects, aspect_titles):
        ax.set_xlim(0, 20)
        ax.set_ylim(len(order) - 0.5, -0.5)
        left = [0.0] * len(order)
        for level_index, (key, color) in enumerate(zip(("top", "middle", "bottom"), ramp)):
            widths = [float(lookup[(name, aspect)][key]) for name in order]
            for row_index, (width, start) in enumerate(zip(widths, left)):
                if width <= 0:
                    continue
                # 2px surface gap between stacked segments
                gap = 0.16 if start > 0 else 0.0
                ax.barh(row_index, width - gap, left=start + gap, height=0.54,
                        color=color, linewidth=0)
                if width >= 3:  # selective direct labels only
                    ax.text(start + width / 2, row_index, f"{int(width)}", ha="center",
                            va="center", fontsize=8.5,
                            color="#ffffff" if level_index < 2 else INK)
            left = [a + b for a, b in zip(left, widths)]
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
        ax.set_xticks([0, 10, 20])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=9)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=10, color=INK)
    handles = [Rectangle((0, 0), 1, 1, color=color) for color in ramp]
    fig.legend(handles, level_names, frameon=False, fontsize=9, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.08), handlelength=1.0, handleheight=1.0)
    fig.text(0.005, 1.14, "Label distribution across the 20 prompts", fontsize=13.5,
             fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.005, 1.05, "Each row sums to 20 answers, ordered best label to worst label",
             fontsize=9.5, color=INK_MUTED, ha="left", va="top")
    save(fig, "human_eval_label_distribution")


def main() -> None:
    figure_retrieval()
    figure_human_overall()
    figure_human_aspects()
    figure_label_distribution()


if __name__ == "__main__":
    main()
