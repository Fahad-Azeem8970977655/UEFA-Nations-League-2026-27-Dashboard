"""Pitch drawing and pass filters. Coordinates are 0-100 on both axes.

 - passes / defensive / carries / heatmaps / positions: own goal at x=0, team attacks towards x=100.
 - shots come in a frame rotated 180 degrees (goal at x=0); shot_xy() converts them.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Ellipse, Rectangle

from unl.style import AMBER, AWAY, GOLD, GREY, HOME, INK, LINE, PITCH

PITCH_ASPECT = 68 / 105


def draw_pitch(ax):
    ax.add_patch(Rectangle((0, 0), 100, 100, facecolor=PITCH, edgecolor=LINE, lw=1.2, zorder=0))
    kw = dict(fill=False, edgecolor=LINE, lw=1.2, zorder=1)
    ax.plot([50, 50], [0, 100], color=LINE, lw=1.2, zorder=1)
    ax.add_patch(Ellipse((50, 50), 2 * 8.714, 2 * 13.46, **kw))
    ax.scatter([50], [50], s=6, color=LINE, zorder=1)
    for side in (0, 1):
        ax.add_patch(Rectangle((0 if side == 0 else 100 - 15.71, 20.35), 15.71, 59.29, **kw))
        ax.add_patch(Rectangle((0 if side == 0 else 100 - 5.24, 36.53), 5.24, 26.94, **kw))
        spot = 10.48 if side == 0 else 89.52
        ax.scatter([spot], [50], s=6, color=LINE, zorder=1)
        th = (-53.13, 53.13) if side == 0 else (180 - 53.13, 180 + 53.13)
        ax.add_patch(Arc((spot, 50), 2 * 8.714, 2 * 13.46, theta1=th[0], theta2=th[1], color=LINE, lw=1.2, zorder=1))
        ax.add_patch(Rectangle((-1.9 if side == 0 else 100, 44.62), 1.9, 10.76, **kw))


def finish_pitch(ax, half=False):
    ax.set_xlim(48 if half else -3, 102 if half else 103)
    ax.set_ylim(-3, 103)
    ax.set_aspect(PITCH_ASPECT, adjustable="box")
    ax.axis("off")
    if not half:
        ax.annotate("", xy=(66, -2.2), xytext=(34, -2.2), arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
        ax.text(50, -2.2, "attack", ha="center", va="bottom", fontsize=6, color=GREY)


def new_figure(n=1, w=6.2, half=False):
    width = w * (0.62 if half else 1) * n
    fig, axes = plt.subplots(1, n, figsize=(width, w * 0.68 + 0.2))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    return fig, axes


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def shot_xy(df, attack_right=True):
    if attack_right:
        return 100 - df["x"], 100 - df["y"]
    return df["x"], df["y"]


def draw_shots(ax, df, color, attack_right=True, half=False):
    draw_pitch(ax)
    if len(df):
        px, py = shot_xy(df, attack_right)
        xg = pd.to_numeric(df["xg"], errors="coerce").fillna(0.03)
        goal = (df["shot_type"] == "goal").values
        size = 45 + xg.values * 900
        ax.scatter(px[~goal], py[~goal], s=size[~goal], facecolor=color, edgecolor=color, alpha=0.35, lw=1.2, zorder=3)
        ax.scatter(px[goal], py[goal], s=130 + xg.values[goal] * 380, marker="*", facecolor=GOLD, edgecolor=INK, lw=0.8, zorder=4)
    finish_pitch(ax, half)


def draw_passes(ax, df, color, max_n=500):
    draw_pitch(ax)
    d = df.dropna(subset=["x", "y", "end_x", "end_y"])
    if len(d) > max_n:
        d = d.sample(max_n, random_state=1)
    for sub, c, a in ((d[~d["outcome"]], GREY, 0.55), (d[d["outcome"]], color, 0.75)):
        if len(sub):
            ax.quiver(sub["x"], sub["y"], sub["end_x"] - sub["x"], sub["end_y"] - sub["y"], angles="xy",
                      scale_units="xy", scale=1, color=c, alpha=a, width=0.0024, headwidth=4, headlength=5, zorder=3)
    finish_pitch(ax)


DEF_STYLE = {"tackle": ("^", HOME), "interception": ("D", AWAY), "clearance": ("s", AMBER),
             "block": ("o", INK), "ball-recovery": ("o", GREY)}


def draw_defensive(ax, df):
    draw_pitch(ax)
    for kind, (mk, c) in DEF_STYLE.items():
        sub = df[df["event_action_type"] == kind]
        if len(sub):
            ax.scatter(sub["x"], sub["y"], marker=mk, s=34, facecolor=c, edgecolor="white", lw=0.5, alpha=0.85,
                       label=kind.replace("-", " "), zorder=3)
    ax.legend(loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    finish_pitch(ax)


def draw_carries(ax, carries, dribbles, color, max_n=500):
    draw_pitch(ax)
    c = carries.dropna(subset=["x", "y", "end_x", "end_y"])
    if len(c) > max_n:
        c = c.sample(max_n, random_state=1)
    if len(c):
        ax.quiver(c["x"], c["y"], c["end_x"] - c["x"], c["end_y"] - c["y"], angles="xy", scale_units="xy", scale=1,
                  color=color, alpha=0.7, width=0.0024, headwidth=4, headlength=5, zorder=3)
    if len(dribbles):
        ok = dribbles["outcome"]
        ax.scatter(dribbles.loc[ok, "x"], dribbles.loc[ok, "y"], s=46, marker="o", facecolor=AMBER, edgecolor="white",
                   zorder=4, label="dribble won")
        ax.scatter(dribbles.loc[~ok, "x"], dribbles.loc[~ok, "y"], s=46, marker="o", facecolor="white",
                   edgecolor=AMBER, zorder=4, label="dribble lost")
        ax.legend(loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    finish_pitch(ax)


def blur(a, sigma=1.4):
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


HEAT_CMAP = LinearSegmentedColormap.from_list("unl_heat", ["#D6ECEA", HOME, "#083A40"])


def draw_heat(ax, df):
    draw_pitch(ax)
    if len(df):
        h, _, _ = np.histogram2d(df["x"].astype(float), df["y"].astype(float), bins=[40, 30], range=[[0, 100], [0, 100]])
        h = blur(h, 1.8)
        h = np.ma.masked_less(h, h.max() * 0.06)
        ax.imshow(h.T, extent=[0, 100, 0, 100], origin="lower", cmap=HEAT_CMAP, alpha=0.92, aspect="auto",
                  interpolation="bilinear", zorder=0.5)
    ax.text(1, 99, "Darker = more touches", fontsize=7, color=GREY, va="top", zorder=5)
    finish_pitch(ax)


# ---- pass filters
PASS_FILTERS = ["All passes", "Key passes", "Assists", "Into final third", "Into penalty area", "Progressive",
                "Long balls", "Crosses"]


def progressive_mask(d):
    sx, sy = (100 - d["x"]) * 1.05, (50 - d["y"]) * 0.68
    ex, ey = (100 - d["end_x"]) * 1.05, (50 - d["end_y"]) * 0.68
    return (np.hypot(ex, ey) <= 0.75 * np.hypot(sx, sy)) & d["outcome"] & (d["end_x"] >= 40)


def apply_pass_filter(d, kind, success_only):
    d = d.copy()
    if kind == "Key passes":
        d = d[d["keypass"]]
    elif kind == "Assists":
        d = d[d["is_assist"]]
    elif kind == "Into final third":
        d = d[(d["end_x"] >= 66.7) & (d["x"] < 66.7)]
    elif kind == "Into penalty area":
        d = d[(d["end_x"] >= 84.29) & d["end_y"].between(20.35, 79.65) & ~((d["x"] >= 84.29) & d["y"].between(20.35, 79.65))]
    elif kind == "Progressive":
        d = d[progressive_mask(d)]
    elif kind == "Long balls":
        d = d[d["is_long_ball"]]
    elif kind == "Crosses":
        d = d[d["event_action_type"] == "cross"]
    if success_only:
        d = d[d["outcome"]]
    return d
