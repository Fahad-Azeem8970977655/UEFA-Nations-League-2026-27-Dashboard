"""Non-pitch charts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unl.style import AWAY, GOLD, GREY, HOME, INK


def compare_bars(rows, h_color=HOME, a_color=AWAY, names=None):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.52 * n + 0.4))
    for i, (label, hv, av, dec) in enumerate(rows):
        y = n - 1 - i
        total = hv + av
        share = 0.5 if total <= 0 else hv / total
        ax.barh(y, share, height=0.28, color=h_color)
        ax.barh(y, 1 - share, left=share, height=0.28, color=a_color)
        ax.text(0.5, y + 0.34, label, ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(-0.02, y, f"{hv:.{dec}f}", ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
        ax.text(1.02, y, f"{av:.{dec}f}", ha="left", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.6, n + 0.5)
    ax.axis("off")
    if names:
        ax.text(0.0, n + 0.2, names[0], color=h_color, fontsize=10, fontweight="bold", ha="left", va="center")
        ax.text(1.0, n + 0.2, names[1], color=a_color, fontsize=10, fontweight="bold", ha="right", va="center")
    fig.tight_layout()
    return fig


def momentum_chart(mom, goals, home_team, away_team):
    fig, ax = plt.subplots(figsize=(8, 2.6))
    mom = mom.sort_values("minute")
    ax.axhline(0, color=GREY, lw=0.8)
    v = mom["value"].astype(float).values
    mins = mom["minute"].astype(float).values
    ax.fill_between(mins, 0, v, where=v >= 0, color=HOME, alpha=0.85, step="mid", lw=0)
    ax.fill_between(mins, 0, v, where=v < 0, color=AWAY, alpha=0.85, step="mid", lw=0)
    top = max(np.abs(v).max() if len(v) else 1, 1) * 1.15
    for _, g in goals.iterrows():
        c = HOME if g["team"] == home_team else AWAY
        ax.scatter([g["minute"]], [top * (0.92 if c == HOME else -0.92)], marker="*", s=150, color=GOLD,
                   edgecolor=INK, lw=0.7, zorder=5)
    ax.set_ylim(-top, top)
    ax.set_xlabel("Minute", fontsize=9)
    ax.set_ylabel("Momentum", fontsize=9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=HOME, lw=6, label=f"{home_team} pressure"), Line2D([], [], color=AWAY, lw=6, label=f"{away_team} pressure"),
                       Line2D([], [], marker="*", color="none", markerfacecolor=GOLD, markeredgecolor=INK, ms=11, label="Goal")],
              fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(0.005, 0.95, home_team, transform=ax.transAxes, color=HOME, fontsize=9, fontweight="bold", va="top")
    ax.text(0.005, 0.05, away_team, transform=ax.transAxes, color=AWAY, fontsize=9, fontweight="bold", va="bottom")
    fig.tight_layout()
    return fig


def radar_chart(series_list):
    """series_list: [(name, Series of 0-100 values indexed by label, colour), ...]"""
    labels = list(series_list[0][1].index)
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(5.4, 5.0))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25", "50", "75"], fontsize=7, color=GREY)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=8, color=INK)
    ax.grid(color="#D5DCE3", lw=0.8)
    ax.spines["polar"].set_color("#D5DCE3")
    for name, s, c in series_list:
        vals = np.concatenate([s.values, s.values[:1]])
        ax.plot(ang, vals, color=c, lw=2, label=name)
        ax.fill(ang, vals, color=c, alpha=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False, ncol=2, title="Percentile among peers (100 = best)", title_fontsize=7)
    fig.tight_layout()
    return fig


def bar_chart(labels, values, title, color=HOME, fmt="{:.1f}", xlabel=None):
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(labels) + 0.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    for yi, v in zip(y, values):
        ax.text(v, yi, (" " if v >= 0 else "") + fmt.format(v) + ("" if v >= 0 else " "), va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel(xlabel or title, fontsize=8, color=GREY)
    if np.min(values) < 0:
        ax.axvline(0, color=GREY, lw=0.8)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    return fig


CLUSTER_COLORS = ["#0E7C86", "#B5326E", "#D99A00", "#4C5FB0", "#3F9D4B", "#C2410C"]


def xg_race(shots, home, away):
    """Cumulative xG for both teams through the match, with goals as stars."""
    fig, ax = plt.subplots(figsize=(8, 3.0))
    s = shots.copy()
    s["xg"] = np.asarray(s["xg"], dtype=float)
    s["t"] = s["minute"].astype(float) + s["added_time"].fillna(0).astype(float)
    end = max(90.0, float(s["t"].max()) if len(s) else 90.0)
    for team, c, dy in ((home, HOME, 0.02), (away, AWAY, -0.02)):
        d = s[s["team"] == team].sort_values("t")
        t = np.concatenate([[0], d["t"].values, [end]])
        cum = np.concatenate([[0], d["xg"].cumsum().values])
        ax.step(t, np.concatenate([cum, [cum[-1]]]), where="post", color=c, lw=2.4)
        g = d["shot_type"].values == "goal"
        if g.any():
            ax.scatter(d["t"].values[g], cum[1:][g], marker="*", s=190, color=GOLD, edgecolor=INK, lw=0.8, zorder=5)
        ax.text(end + 0.8, cum[-1], f"{team} {cum[-1]:.2f}", color=c, fontsize=9, fontweight="bold", va="center")
    ax.axvline(45, color=GREY, lw=0.8, ls=":", label="Half-time")
    from matplotlib.lines import Line2D
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=[Line2D([], [], color=HOME, lw=3, label=home), Line2D([], [], color=AWAY, lw=3, label=away),
                       Line2D([], [], marker="*", color="none", markerfacecolor=GOLD, markeredgecolor=INK, ms=11, label="Goal")] + h,
              fontsize=7, frameon=False, loc="upper left")
    ax.set_xlim(0, end + 14)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Minute", fontsize=9)
    ax.set_ylabel("Cumulative xG", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def scatter_labeled(x, y, labels, xlabel, ylabel, diag=False, groups=None, highlight=None, label="auto",
                    label_n=10, title=None, xlog=False):
    """Scatter with team labels. diag draws y = x. groups (Series aligned with x) colours by category."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    labels = list(labels)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if diag:
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--", zorder=1)
    if groups is None:
        ax.scatter(x, y, s=46, color=HOME, alpha=0.8, edgecolor="white", lw=0.6, zorder=3, label="Team")
        if diag:
            ax.legend(handles=[plt.Line2D([], [], marker="o", color="none", markerfacecolor=HOME, label="Team"),
                               plt.Line2D([], [], color=GREY, ls="--", label="Actual = expected")], fontsize=8, frameon=False, loc="upper left")
    else:
        g = np.asarray(groups)
        for i, name in enumerate(sorted(set(g))):
            m = g == name
            ax.scatter(x[m], y[m], s=52, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], alpha=0.85,
                       edgecolor="white", lw=0.6, label=name, zorder=3)
        ax.legend(fontsize=7, frameon=False, loc="best", title="Style group", title_fontsize=7)
    if label == "all":
        idx = set(range(len(labels)))
    else:
        score = np.abs(y - x) if diag else np.hypot((x - x.mean()) / (x.std() or 1), (y - y.mean()) / (y.std() or 1))
        idx = set(np.argsort(-score)[:label_n])
    for i in idx | {i for i, n in enumerate(labels) if highlight and n == highlight}:
        hl = bool(highlight) and labels[i] == highlight
        ax.annotate(labels[i], (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=8 if not hl else 10,
                    color=AWAY if hl else INK, fontweight="bold" if hl else "normal")
        if hl:
            ax.scatter([x[i]], [y[i]], s=130, facecolor="none", edgecolor=AWAY, lw=2, zorder=4)
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#E6EBF0", lw=0.6, zorder=0)
    fig.tight_layout()
    return fig
