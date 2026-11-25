"""Colours and page CSS. Teal / magenta are the two team colours in every chart."""
import streamlit as st

INK = "#16202E"
HOME = "#0E7C86"
AWAY = "#B5326E"
GREY = "#8A97A6"
PITCH = "#E9F1EA"
LINE = "#7C9A89"
GOLD = "#D99A00"
AMBER = "#C27C0E"

NAV_ORDER = ["🏠", "🏆", "👕", "⚽", "👤", "🆚", "🎯", "🔥", "📨", "📊", "⚡", "🧩", "🎭", "🧱", "🤖", "🚀", "⏳", "🧤", "📈", "🔎", "🔮", "🧬", "🟨", "📉", "📋", "🧪", "📝"]
# icons are drawn with CSS so file names stay plain ASCII (emoji in file names break on some Windows setups)
NAV_CSS = "\n".join(
    '[data-testid="stSidebarNav"] li:nth-child(%d) a span:not(:has(span))::before { content: "%s"; display: inline-block; flex: none; margin-right: .6rem; font-size: 1.05rem; line-height: 1; }'
    % (i, "".join("\\%X" % ord(c) for c in e)) for i, e in enumerate(NAV_ORDER, 1))


def inject_css():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&display=swap');
    h1, h2, h3 { font-family: 'Barlow Semi Condensed', sans-serif; letter-spacing: 0; }
    .scoreline { display: flex; align-items: center; justify-content: center; gap: 2.2rem; margin: .6rem 0 .2rem 0; }
    .scoreline .side { font-family: 'Barlow Semi Condensed', sans-serif; font-size: 1.9rem; font-weight: 600;
                       flex: 1; padding-bottom: .25rem; }
    .scoreline .home { text-align: right; border-bottom: 5px solid #0E7C86; }
    .scoreline .away { text-align: left; border-bottom: 5px solid #B5326E; }
    .scoreline .score { font-family: 'Barlow Semi Condensed', sans-serif; font-size: 3.6rem; font-weight: 700;
                        line-height: 1; white-space: nowrap; }
    .matchmeta { text-align: center; color: #5B6878; margin-bottom: 1rem; }

    /* App shell: dark navy sidebar, soft page background */
    [data-testid="stAppViewContainer"] { background: #F3F6FB; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"], .block-container { padding-top: 2.2rem !important; }
    [data-testid="stSidebar"] { background: #0B1F3A; }
    [data-testid="stSidebar"] * { color: #E8EEF7; }
    [data-testid="stSidebarNav"] a { border-radius: 8px; margin: 1px 6px; padding: .28rem .6rem; }
    [data-testid="stSidebarNav"] ul { gap: .15rem; padding-top: .4rem; }
    [data-testid="stSidebarNav"] li a { display: flex !important; flex-direction: row !important; align-items: center !important;
                                         min-height: 0 !important; padding: .32rem .7rem !important; gap: 0 !important; }
    [data-testid="stSidebarNav"] li a span:not(:has(span)) { display: flex !important; flex-direction: row !important;
                                         align-items: center !important; margin: 0 !important; line-height: 1.25 !important;
                                         font-size: .95rem; }
    /*NAVICONS*/
    [data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,.08); }
    [data-testid="stSidebarNav"] a[aria-current="page"] { background: #1D4E5F; font-weight: 600; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #9FB2CC; }
    h1 { color: #0B1F3A; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #1D5FAF; }
    [data-testid="stDataFrame"] { border: 1px solid #DDE3EA; border-radius: 8px; }

    /* KPI strip */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(var(--kmin, 150px), 1fr));
                gap: .8rem; margin: .4rem 0 1.4rem 0; }
    .kpi { background: #fff; border: 1px solid #E1E7EF; border-top: 5px solid var(--tone, #3F5466);
           border-radius: 12px; padding: .65rem .9rem .7rem .9rem; box-shadow: 0 1px 3px rgba(11,31,58,.06); }
    .kpi-label { font-size: .8rem; color: #5B6878; line-height: 1.2; display: flex; align-items: center; gap: .35rem; }
    .kpi-icon { font-size: 1.05rem; }
    .kpi-value { font-family: 'Barlow Semi Condensed', sans-serif; font-size: 2.05rem; font-weight: 700;
                 line-height: 1.15; color: var(--tone, #16202E); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .kpi-note { font-size: .76rem; color: #5B6878; margin-top: .05rem; }

    /* Home hero */
    .hero { display: flex; align-items: center; justify-content: space-between; gap: 1.2rem; padding: 1.1rem 1.4rem;
            margin: 0 0 .8rem 0; border-radius: 16px; color: #fff;
            background: linear-gradient(120deg, #0A2A6B 0%, #0D47A1 55%, #1976D2 100%); }
    .hero-left { flex: 1 1 260px; min-width: 220px; }
    .hero-logo { background: #fff; border-radius: 14px; display: inline-block; padding: .3rem .6rem; margin-bottom: .6rem; }
    .hero-logo img { height: 64px; width: auto; display: block; }
    .hero-title { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 800; line-height: .9; color: #fff;
                  font-size: clamp(1.8rem, 3.4vw, 3rem); text-transform: uppercase; margin: 0; }
    .hero-kicker { font-weight: 700; letter-spacing: .14em; font-size: .62rem; color: #BFD4F6; margin: .35rem 0 .4rem 0; }
    .hero-season { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 400; font-size: clamp(1.1rem, 1.8vw, 1.5rem); }
    .hero-season b { font-weight: 800; }
    .hero-pill { display: inline-block; background: #F2B705; color: #0B1F3A; font-weight: 700; padding: .25rem .7rem; font-size: .8rem;
                 border-radius: 6px; margin: .5rem 0 .4rem 0; }
    .hero-sub { color: #CFE0FA; font-size: .75rem; }
    .hex-wrap { flex: 0 0 auto; display: flex; flex-direction: column; align-items: center; padding-bottom: 26px; }
    .hex-row { display: flex; margin-bottom: -25px; }
    .hex-row.off { margin-left: 0; }
    .hex { width: 84px; height: 97px; margin: 0 2px; clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
           background: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
           padding: 0 6px; }
    .hex.light { background: #DCE9FB; }
    .hex.gold { background: #F2B705; }
    .hex.green { background: #2E9E6B; }
    .hex.green *, .hex.green { color: #fff !important; }
    .hex-ico { font-size: .8rem; line-height: 1; }
    .hex-val { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 800; font-size: 1.1rem; color: #0A2A6B; line-height: 1.1; }
    .hex-lab { font-size: .46rem; font-weight: 700; color: #35507F; line-height: 1.1; }
    .hex.gold .hex-val, .hex.gold .hex-lab { color: #0B1F3A; }
    @media (max-width: 900px) { .hero { flex-direction: column; text-align: center; } .hex-wrap { transform: scale(.85); } }
    .group-label { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 600; font-size: 1.25rem; color: #3F5466;
                   border-bottom: 3px solid #1D5FAF; display: inline-block; margin: .6rem 0 .4rem 0; }
    </style>
    """.replace("/*NAVICONS*/", NAV_CSS),
        unsafe_allow_html=True,
    )