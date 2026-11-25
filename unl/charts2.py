"""Charts for the analytics pages (second layer).  Same look as charts.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from unl.pitch import draw_pitch, finish_pitch
from unl.style import AWAY, GOLD, GREY, HOME, INK

PALETTE = ["#0E7C86", "#B5326E", "#D99A00", "#4C5FB0", "#3F9D4B", "#C2410C", "#6B7280", "#7C3AED"]


def _clean(ax, grid="y"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#E6EBF0", lw=0.6, zorder=0)
    ax.tick_params(labelsize=8)


def role_map(df, colors, highlight=None):
    """df: player, depth, mean_y, role. Pitch with one dot per player, coloured by role."""
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    draw_pitch(ax)
    for role, g in df.groupby("role"):
        ax.scatter(g["depth"], g["mean_y"], s=34, color=colors.get(role, GREY), alpha=0.85, edgecolor="white", lw=0.5,
                   label=f"{role} ({len(g)})", zorder=3)
    if highlight is not None and len(highlight):
        ax.scatter(highlight["depth"], highlight["mean_y"], s=150, facecolor="none", edgecolor=INK, lw=1.8, zorder=4)
        for r in highlight.itertuples():
            ax.annotate(r.player, (r.depth, r.mean_y), xytext=(5, 6), textcoords="offset points", fontsize=8, color=INK,
                        fontweight="bold")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4)
    finish_pitch(ax)
    ax.annotate("", xy=(0.97, -0.005), xytext=(0.03, -0.005), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    fig.tight_layout()
    return fig


def line_chart(x, series, ylabel="", xlabel="", title=None, zero=True, figsize=(7.2, 3.2), vline=None, ci=None):
    """series: [(name, values, colour)]; ci: optional {name: (lo, hi)}"""
    fig, ax = plt.subplots(figsize=figsize)
    for name, v, c in series:
        ax.plot(x, v, color=c, lw=2.2, label=name, zorder=3)
        if ci and name in ci:
            ax.fill_between(x, ci[name][0], ci[name][1], color=c, alpha=0.15, zorder=2)
    if zero:
        ax.axhline(0, color=GREY, lw=0.8)
    if vline is not None:
        ax.axvline(vline, color=GREY, lw=0.8, ls=":")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(series))
    _clean(ax, "both")
    fig.tight_layout()
    return fig


def grouped_bars(cats, series, ylabel="", title=None, figsize=(7.2, 3.2), fmt=None, xlabel=None):
    """series: [(name, values, colour)]"""
    fig, ax = plt.subplots(figsize=figsize)
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(cats))
    for i, (name, v, c) in enumerate(series):
        b = ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.92, color=c, label=name, zorder=3)
        if fmt:
            for rect, val in zip(b, v):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), fmt.format(val), ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=n, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    _clean(ax)
    fig.tight_layout()
    return fig


def stacked_share(labels, parts, colors, title=None, figsize=None):
    """Horizontal 100% stacked bars. parts: dict name -> array of % (columns sum ~100)."""
    fig, ax = plt.subplots(figsize=figsize or (7.2, 0.3 * len(labels) + 0.9))
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for (name, v), c in zip(parts.items(), colors):
        v = np.asarray(v, dtype=float)
        ax.barh(y, v, left=left, color=c, height=0.66, label=name, zorder=3)
        for yi, l, w_ in zip(y, left, v):
            if w_ >= 9:
                ax.text(l + w_ / 2, yi, f"{w_:.0f}", ha="center", va="center", fontsize=7, color="white")
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8, frameon=False, ncol=len(parts), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK, pad=22)
    fig.tight_layout()
    return fig


def forest(labels, est, lo, hi, title=None, xlabel="", zero=0.0, color=HOME, highlight=None):
    """Point estimates with intervals, one row per label."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(6.8, 0.3 * n + 0.9))
    y = np.arange(n)[::-1]
    for yi, e, l, h, name in zip(y, est, lo, hi, labels):
        c = AWAY if highlight and name == highlight else color
        ax.plot([l, h], [yi, yi], color=c, lw=2, alpha=0.55, zorder=3)
        ax.scatter([e], [yi], color=c, s=34, zorder=4)
    ax.axvline(zero, color=GREY, lw=0.9, ls="--")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=color, lw=2, alpha=0.8, label="Estimate and range"),
                       Line2D([], [], color=GREY, ls="--", label="No effect / average")], fontsize=7, frameon=False,
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    _clean(ax, "x")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig


def scatter_trend(x, y, labels, xlabel, ylabel, highlight=None, label_n=8, trend=True, title=None, color=HOME,
                  figsize=(7.2, 4.8)):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x[ok], y[ok], s=36, color=color, alpha=0.75, edgecolor="white", lw=0.5, zorder=3, label="Team / player")
    if trend and ok.sum() > 3:
        b = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=AWAY, lw=1.8, zorder=2, label="Linear trend")
    ax.legend(fontsize=8, frameon=False, loc="best")
    labels = list(labels)
    score = np.hypot((x - np.nanmean(x)) / (np.nanstd(x) or 1), (y - np.nanmean(y)) / (np.nanstd(y) or 1))
    score = np.where(ok, score, -1)
    idx = set(np.argsort(-score)[:label_n])
    for i in idx | {i for i, n in enumerate(labels) if highlight and n == highlight}:
        if not ok[i]:
            continue
        hl = bool(highlight) and labels[i] == highlight
        ax.annotate(labels[i], (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=10 if hl else 8,
                    color=AWAY if hl else INK, fontweight="bold" if hl else "normal")
        if hl:
            ax.scatter([x[i]], [y[i]], s=130, facecolor="none", edgecolor=AWAY, lw=2, zorder=4)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    _clean(ax, "both")
    fig.tight_layout()
    return fig


def reliability(cal):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    ax.plot(cal["pred"], cal["actual"], color=HOME, lw=2, marker="o", ms=5)
    ax.set_xlabel("Predicted completion probability", fontsize=9)
    ax.set_ylabel("Actual completion rate", fontsize=9)
    ax.set_title("Model check: predicted versus actual (deciles)", fontsize=10, loc="left", color=INK)
    _clean(ax, "both")
    fig.tight_layout()
    return fig


def goal_zone_grid(tbl, value="Save %", title=None):
    """3 x 3 goal frame: each cell shows the value, shots faced, and goals."""
    from unl.finishing import GOAL_ZONES
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    vals = [tbl.loc[z, value] for z in GOAL_ZONES if z in tbl.index and tbl.loc[z, "n"] >= 1]
    lo, hi = (min(vals), max(vals)) if vals else (0, 100)
    for z, (cx, cy) in GOAL_ZONES.items():
        ax.add_patch(Rectangle((cx, cy), 1, 1, facecolor="#F2F5F8", edgecolor="white", lw=2))
        if z in tbl.index:
            r = tbl.loc[z]
            t = 0 if hi == lo else (r[value] - lo) / (hi - lo)
            ax.add_patch(Rectangle((cx, cy), 1, 1, facecolor=plt.cm.GnBu(0.15 + 0.7 * t), edgecolor="white", lw=2))
            ax.text(cx + .5, cy + .56, f"{r[value]:.0f}%", ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
            ax.text(cx + .5, cy + .24, f"{int(r['n'])} shots, {int(r['goals'])} goals", ha="center", va="center", fontsize=7, color=INK)
    ax.add_patch(Rectangle((1, 1), 0, 0))
    ax.plot([0, 0, 3, 3], [0, 3, 3, 0], color=INK, lw=3)
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(-0.1, 3.15)
    ax.set_aspect("equal")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.GnBu, norm=plt.Normalize(lo, hi))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.06, pad=0.02)
    cb.set_label(f"{value}: darker = higher", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    return fig


def avg_positions(ax, pts, line_height=None, hull=True, color=HOME, label=True):
    """pts: DataFrame with average_x, average_y and a name column."""
    draw_pitch(ax)
    ax.scatter(pts["average_x"], pts["average_y"], s=110, color=color, edgecolor="white", lw=1.2, zorder=4)
    if label:
        for r in pts.itertuples():
            ax.annotate(str(r.name_short), (r.average_x, r.average_y), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
    if line_height is not None:
        ax.axvline(line_height, color=AWAY, lw=2, ls="--", zorder=3)
        ax.text(line_height + 1, 97, f"back line {line_height:.0f}", color=AWAY, fontsize=8, va="top")
    finish_pitch(ax)

    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", ms=9, label="Outfield player (average position)")]
    if line_height is not None:
        h.append(Line2D([], [], color=AWAY, ls="--", lw=2, label="Back line (4 deepest players)"))
    ax.legend(handles=h, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2)
