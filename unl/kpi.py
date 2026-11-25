"""KPI strip used at the top of every page.

    kpi_row([("Matches", 156), ("xG per match", "1.32", "league average 1.28", "blue"), ...])

Each item is (label, value, note=None, tone=None, help=None).  Tones borrow the colours of the Nations League flag.
"""
import html

import streamlit as st

TONES = {"slate": "#3F5466", "blue": "#1D5FAF", "red": "#D61F3D", "green": "#14A04B", "gold": "#E5A400",
         "teal": "#0E7C86", "magenta": "#B5326E"}
ICONS = [("goal", "⚽"), ("xg", "🎯"), ("xt", "⚡"), ("match", "📅"), ("team", "👕"), ("player", "👤"), ("scorer", "👟"), ("assist", "🅰️"),
         ("possession", "🔵"), ("pass", "📨"), ("red", "🟥"), ("rating", "📈"), ("shot", "🥅"), ("minute", "⏱️"), ("sub", "🔁"),
         ("keeper", "🧤"), ("save", "🧤"), ("line", "🧱"), ("set", "🚩"), ("role", "🎭"), ("win", "🏅"), ("press", "🔥"), ("ppda", "🔥"),
         ("carry", "🚀"), ("progress", "🚀"), ("channel", "🛣️"), ("point", "🏆"), ("elo", "📈"), ("model", "🧪"), ("spread", "↔️"),
         ("late", "⏳"), ("defence", "🛡️"), ("attack", "🗡️"), ("variance", "📐"), ("correlation", "📐"), ("style", "🧩"), ("touch", "👆")]


def icon_for(label):
    l = str(label).lower()
    for key, ic in ICONS:
        if key in l:
            return ic
    return "📌"


_CYCLE = ["blue", "red", "green", "gold", "teal", "magenta"]


def kpi_row(items, min_width=150):
    cards = []
    for i, it in enumerate(items):
        label, value = it[0], it[1]
        note = it[2] if len(it) > 2 else None
        tone = (it[3] if len(it) > 3 else None) or _CYCLE[i % len(_CYCLE)]
        tip = it[4] if len(it) > 4 else None
        title = f' title="{html.escape(str(tip))}"' if tip else ""
        n = f'<div class="kpi-note">{html.escape(str(note))}</div>' if note else ""
        cards.append(
            f'<div class="kpi" style="--tone:{TONES.get(tone, tone)}"{title}>'
            f'<div class="kpi-label"><span class="kpi-icon">{icon_for(label)}</span>{html.escape(str(label))}</div>'
            f'<div class="kpi-value">{html.escape(str(value))}</div>{n}</div>')
    st.markdown(f'<div class="kpi-grid" style="--kmin:{min_width}px">{"".join(cards)}</div>', unsafe_allow_html=True)


def fmt(v, d=0, suffix=""):
    try:
        return f"{float(v):,.{d}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"
