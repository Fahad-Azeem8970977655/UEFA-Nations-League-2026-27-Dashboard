"""Extra drawing + metrics for the Shot Map, Pressing and Passing pages.

Frames (same as pitch.py): passes / defensive actions use 0-100 with the team attacking to the right.
Shots arrive rotated 180 degrees (goal at x=0) and are converted with shot_xy().
"""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from unl.pitch import HEAT_CMAP, blur, draw_heat, draw_pitch, finish_pitch, progressive_mask, shot_xy
from unl.style import AMBER, AWAY, GOLD, GREY, HOME, INK, LINE

# ------------------------------------------------------------------ shots
SHOT_STYLE = {  # kind: (marker, colour, label)
    "goal": ("*", GOLD, "Goal"),
    "save": ("o", HOME, "Saved"),
    "post": ("D", AMBER, "Hit post"),
    "miss": ("X", GREY, "Off target"),
    "block": ("s", AWAY, "Blocked"),
}
ON_TARGET = ["goal", "save", "post"]
MANY_SHOTS = 350  # above this many shots, markers turn into an xG density map so the picture stays readable


def shot_kind(s):
    return s.replace({"blocked-off-line": "block"})


def draw_shot_map(ax, df, lines=False):
    """Half pitch, goal on the right. Size = xG. Gold star = goal."""
    draw_pitch(ax)
    if len(df):
        px, py = shot_xy(df)
        kind = shot_kind(df["shot_type"]).values
        xg = pd.to_numeric(df["xg"], errors="coerce").fillna(0.03).values
        if len(df) > MANY_SHOTS:
            h, _, _ = np.histogram2d(px, py, bins=[40, 30], range=[[0, 100], [0, 100]], weights=xg)
            h = np.ma.masked_less(blur(h, 1.4), 1e-9)
            h = np.ma.masked_less(h, h.max() * 0.05)
            ax.imshow(h.T, extent=[0, 100, 0, 100], origin="lower", cmap=HEAT_CMAP, alpha=0.92, aspect="auto",
                      interpolation="bilinear", zorder=0.5)
            g = kind == "goal"
            ax.scatter(px[g], py[g], s=34, marker="*", facecolor=GOLD, edgecolor=INK, lw=0.5, zorder=5)
            finish_pitch(ax, half=True)
            handles = [Line2D([], [], marker="*", color="w", markerfacecolor=GOLD, markeredgecolor=INK, markersize=11,
                              label="Goal"), Line2D([], [], color=HEAT_CMAP(0.7), lw=6, label="Shot density (xG-weighted)")]
            ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))
            return
        if lines:
            g = kind == "goal"
            for a, b in zip(px[g], py[g]):
                ax.plot([a, 100], [b, 50], color=GOLD, lw=0.9, alpha=0.55, zorder=2)
        for k, (mk, c, _) in SHOT_STYLE.items():
            sel = kind == k
            if not sel.any():
                continue
            if k == "goal":
                ax.scatter(px[sel], py[sel], s=140 + xg[sel] * 420, marker=mk, facecolor=c, edgecolor=INK, lw=0.8, zorder=5)
            else:
                ax.scatter(px[sel], py[sel], s=40 + xg[sel] * 800, marker=mk, facecolor=c, edgecolor="white",
                           lw=0.6, alpha=0.7, zorder=3)
    finish_pitch(ax, half=True)
    handles = [Line2D([], [], marker=mk, color="w", markerfacecolor=c, markeredgecolor=INK if k == "goal" else "white",
                      markersize=11 if k == "goal" else 7, label=lab) for k, (mk, c, lab) in SHOT_STYLE.items()]
    ax.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))


# goal frame: posts at y 45.2-54.8 (all goals/saves fall in 45.3-54.7); z = height, crossbar assumed ~38
GOAL_Y = (45.2, 54.8)
GOAL_Z = 38.0


def draw_goal_frame(ax, df):
    """Where the shots ended up, seen from the front. Misses outside the frame are shown too."""
    d = df.dropna(subset=["goal_mouth_y", "goal_mouth_z"])
    ax.add_patch(Rectangle((GOAL_Y[0], 0), GOAL_Y[1] - GOAL_Y[0], GOAL_Z, facecolor="#F3F7F4", edgecolor="none", zorder=0))
    ax.plot([GOAL_Y[0], GOAL_Y[0], GOAL_Y[1], GOAL_Y[1]], [0, GOAL_Z, GOAL_Z, 0], color=INK, lw=3, zorder=2,
            solid_capstyle="round")
    ax.axhline(0, color=LINE, lw=1.5, zorder=1)
    kind = shot_kind(d["shot_type"]).values
    f = 0.3 if len(d) > MANY_SHOTS else 1.0  # smaller markers when there are many shots
    for k, (mk, c, _) in SHOT_STYLE.items():
        if k == "block":
            continue
        sel = kind == k
        if sel.any():
            ax.scatter(d["goal_mouth_y"].values[sel], d["goal_mouth_z"].values[sel],
                       s=(120 if k == "goal" else 42) * f, marker=mk, facecolor=c,
                       edgecolor=INK if k == "goal" else "white", lw=0.7 * f, alpha=0.9 if k == "goal" else 0.5 * (1.4 if f == 1 else 1),
                       zorder=4 if k == "goal" else 3)
    ax.set_xlim(GOAL_Y[0] - 5, GOAL_Y[1] + 5)
    ax.set_ylim(-2, GOAL_Z + 10)
    ax.axis("off")


def goals_table(shots, inc):
    """Goals with assist provider (from incidents when the minute matches)."""
    g = shots[shots["shot_type"] == "goal"].copy()
    gi = inc[inc["type"] == "goal"][["match_id", "player", "minute", "assist1"]].drop_duplicates(["match_id", "player", "minute"])
    g = g.merge(gi, on=["match_id", "player", "minute"], how="left")
    g["opponent"] = np.where(g["team"] == g["home_team"], g["away_team"], g["home_team"])
    return g


# ------------------------------------------------------------------ pressing
PRESS_TYPES = ["tackle", "interception", "ball-recovery"]


def pressing_table(D):
    """One row per team. PPDA = opponent passes in their own 60% / our tackles+interceptions in that zone.
    In the team's own frame the zone is x >= 40 (we attack to the right)."""
    p = D.passes[D.passes["event_action_type"].isin(["pass", "cross"])]
    opp_p = p[p["x"] <= 60].groupby(["match_id", "team"]).size().rename("opp_passes").reset_index()
    pairs = D.tm[["match_id", "team", "opponent"]]
    opp_p = opp_p.rename(columns={"team": "opponent"}).merge(pairs, on=["match_id", "opponent"], how="right")
    opp_p["opp_passes"] = opp_p["opp_passes"].fillna(0)
    d = D.defensive
    pres = d[d["event_action_type"].isin(["tackle", "interception"]) & (d["x"] >= 40)]
    pres = pres.groupby(["match_id", "team"]).size().rename("press_actions").reset_index()
    mt = opp_p.merge(pres, on=["match_id", "team"], how="left").fillna({"press_actions": 0})
    reg = d[d["event_action_type"].isin(PRESS_TYPES)]
    hi = reg[reg["x"] >= 66.7].groupby(["match_id", "team"]).size().rename("high_regains").reset_index()
    tot = reg.groupby(["match_id", "team"]).size().rename("regains").reset_index()
    mt = mt.merge(hi, on=["match_id", "team"], how="left").merge(tot, on=["match_id", "team"], how="left").fillna(0)
    g = mt.groupby("team")
    out = pd.DataFrame({"Matches": g.size(), "opp_passes": g["opp_passes"].sum(), "press_actions": g["press_actions"].sum(),
                        "High regains / match": g["high_regains"].mean(), "Regains / match": g["regains"].mean()})
    out["PPDA"] = out["opp_passes"] / out["press_actions"].replace(0, np.nan)
    out["Press actions / match"] = out["press_actions"] / out["Matches"]
    rg = reg.groupby("team")["x"].mean()
    out["Avg regain height"] = rg
    out["Regains in opp half %"] = reg.assign(h=reg["x"] >= 50).groupby("team")["h"].mean() * 100
    return out.drop(columns=["opp_passes", "press_actions"]).sort_values("PPDA")


def draw_press_map(ax, df, color=HOME):
    """Heat of ball-winning actions, with the PPDA zone (x >= 40) and the final third marked."""
    draw_heat(ax, df)
    ax.axvline(40, color=INK, lw=1, ls="--", alpha=0.6, zorder=2)
    ax.axvline(66.7, color=INK, lw=1, ls=":", alpha=0.6, zorder=2)
    ax.text(40.8, 100.5, "press zone", fontsize=7, color=INK, va="bottom")
    ax.text(67.5, 100.5, "final third", fontsize=7, color=INK, va="bottom")
    ax.annotate("", xy=(97, -1.6), xytext=(80, -1.6), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(79, -1.6, "attack", fontsize=7, color=GREY, ha="right", va="center")


# ------------------------------------------------------------------ passing
PASS_CATS = ["Assist", "Key pass", "Cross", "Progressive", "Long ball", "Completed", "Incomplete"]
PASS_COLORS = {"Assist": GOLD, "Key pass": AWAY, "Cross": AMBER, "Progressive": HOME, "Long ball": "#4C5FB0",
               "Completed": "#9DB4B8", "Incomplete": "#C7CDD4"}


def classify_passes(d):
    """First matching category wins, in the order of PASS_CATS."""
    out = np.full(len(d), "Completed", dtype=object)
    out[~d["outcome"].values] = "Incomplete"
    out[d["is_long_ball"].values & d["outcome"].values] = "Long ball"
    out[progressive_mask(d).values] = "Progressive"
    out[(d["event_action_type"] == "cross").values] = "Cross"
    out[d["keypass"].values] = "Key pass"
    out[d["is_assist"].values] = "Assist"
    return pd.Series(out, index=d.index)


def _quiver(ax, sub, color, alpha, width=0.0026, z=3):
    ax.quiver(sub["x"], sub["y"], sub["end_x"] - sub["x"], sub["end_y"] - sub["y"], angles="xy", scale_units="xy",
              scale=1, color=color, alpha=alpha, width=width, headwidth=4, headlength=5, zorder=z)


def draw_pass_arrows(ax, d, cats, by="type", max_n=450, players=None):
    """d needs a 'cat' column. by='type' colours by category, by='player' colours by player."""
    draw_pitch(ax)
    d = d[d["cat"].isin(cats)].dropna(subset=["x", "y", "end_x", "end_y"])
    rare = d[d["cat"].isin(["Assist", "Key pass"])]
    rest = d.drop(rare.index)
    if len(rest) > max_n:
        rest = rest.sample(max_n, random_state=1)
    d = pd.concat([rest, rare])
    handles = []
    if by == "player":
        pal = ["#0E7C86", "#B5326E", "#D99A00", "#4C5FB0", "#3F9D4B", "#C2410C", "#7A4FA0", "#16202E"]
        names = players or list(d["player"].value_counts().index[:8])
        for i, nm in enumerate(names):
            sub = d[d["player"] == nm]
            if len(sub):
                _quiver(ax, sub, pal[i % len(pal)], 0.8)
            handles.append(Line2D([], [], color=pal[i % len(pal)], lw=2.5, label=nm))
    else:
        for cat in PASS_CATS[::-1]:
            sub = d[d["cat"] == cat]
            if len(sub):
                _quiver(ax, sub, PASS_COLORS[cat], 0.45 if cat in ("Incomplete", "Completed") else 0.9,
                        z=3 if cat in ("Incomplete", "Completed") else 4)
        handles = [Line2D([], [], color=PASS_COLORS[c], lw=2.5, label=c) for c in PASS_CATS if c in cats]
    finish_pitch(ax)
    ax.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), fontsize=7, frameon=False,
              bbox_to_anchor=(0.5, -0.04))


FLOW_CMAP = LinearSegmentedColormap.from_list("flow", [AWAY, "#C9B8C0", HOME])


def draw_pass_flow(ax, d, nx=6, ny=4, min_n=8):
    """One arrow per start zone: average start -> average end. Width = volume, colour = completion rate."""
    draw_pitch(ax)
    d = d.dropna(subset=["x", "y", "end_x", "end_y"]).copy()
    d["zx"] = np.clip((d["x"] / 100 * nx).astype(int), 0, nx - 1)
    d["zy"] = np.clip((d["y"] / 100 * ny).astype(int), 0, ny - 1)
    g = d.groupby(["zx", "zy"])
    agg = g.agg(n=("x", "size"), x=("x", "mean"), y=("y", "mean"), ex=("end_x", "mean"), ey=("end_y", "mean"),
                acc=("outcome", "mean")).reset_index()
    agg = agg[agg["n"] >= min_n]
    if len(agg):
        top = agg["n"].max()
        for r in agg.itertuples():
            ax.annotate("", xy=(r.ex, r.ey), xytext=(r.x, r.y), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=FLOW_CMAP(np.clip((r.acc - 0.6) / 0.4, 0, 1)),
                                        lw=1.2 + 5 * r.n / top, alpha=0.9, mutation_scale=11, shrinkA=0, shrinkB=0))
    for i in range(1, nx):
        ax.plot([i * 100 / nx] * 2, [0, 100], color=LINE, lw=0.5, ls=":", zorder=1)
    for j in range(1, ny):
        ax.plot([0, 100], [j * 100 / ny] * 2, color=LINE, lw=0.5, ls=":", zorder=1)
    finish_pitch(ax)
