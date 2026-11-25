"""Pitch drawings for the expected-threat page. Same frame as pitch.py: team attacks to the right, 0-100."""
import numpy as np
from matplotlib.colors import PowerNorm

from unl.advanced import NX, NY
from unl.pitch import HEAT_CMAP, blur, draw_pitch, finish_pitch


def draw_xt_surface(ax, grid):
    """The fitted xT surface: how likely a possession starting in each zone is to end in a goal."""
    draw_pitch(ax)
    ax.imshow(grid.T, extent=[0, 100, 0, 100], origin="lower", cmap=HEAT_CMAP, alpha=0.92, aspect="auto",
              interpolation="bilinear", norm=PowerNorm(0.5), zorder=0.5)
    finish_pitch(ax)


def draw_xt_origin(ax, moves):
    """Where a team's threat is created: xT gained by completed passes and carries, summed by start zone."""
    draw_pitch(ax)
    gain = np.clip(moves["xt_gain"].values, 0, None)
    if len(moves) and gain.sum() > 0:
        h, _, _ = np.histogram2d(moves["x"].astype(float), moves["y"].astype(float), bins=[NX, NY],
                                 range=[[0, 100], [0, 100]], weights=gain)
        h = blur(h, 1.2)
        h = np.ma.masked_less(h, h.max() * 0.04)
        ax.imshow(h.T, extent=[0, 100, 0, 100], origin="lower", cmap=HEAT_CMAP, alpha=0.92, aspect="auto",
                  interpolation="bilinear", zorder=0.5)
    finish_pitch(ax)
