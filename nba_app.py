# nba_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from rapidfuzz import process

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NBA Player Prop Tools", page_icon="🏀", layout="wide")
st.title("🏀 NBA Player Prop Tools")

# =========================
# THEME / CSS
# =========================
st.markdown("""
<style>
:root {
  --bg: #0e0f11;
  --text: #f9fafb;
  --muted: #9ca3af;
  --card: #1a1b1e;
  --border: #32353b;
  --accent: #00c896;
}
body, .block-container { background: var(--bg); color: var(--text); }

.stButton > button {
  background: var(--accent) !important;
  color: #0b1220 !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  padding: 8px 14px !important;
  font-size: 0.9rem !important;
  box-shadow: 0 6px 18px rgba(0,200,150,0.25) !important;
}

input, select, textarea {
  background: #202225 !important;
  color: var(--text) !important;
  border: 1px solid #3b3f45 !important;
  border-radius: 8px !important;
  font-size: 0.9rem !important;
  padding: 6px !important;
}

.card {
  --pad-x: 20px;
  --pad-y: 18px;
  padding: var(--pad-y) var(--pad-x);
  border-radius: 14px;
  margin: 10px 0 20px 0;
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  box-shadow: 0 0 14px rgba(0,0,0,0.25);
  width: 100%;
}
.neutral { --card-bg:#222; --card-border:#777; }
.pos { --card-bg:#0b3d23; --card-border:#00FF99; }
.neg { --card-bg:#3d0b0b; --card-border:#FF5555; }
.card h2 { color:#fff; margin:0 0 6px 0; font-weight:800; font-size:1.05rem; }
.cond { color:#a9b1bb; font-size:0.9rem; margin: 2px 0 10px 0; }
.row { display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end; justify-content:space-between; margin: 6px 0 4px 0; }
.m { min-width:120px; flex:1; }
.lab { color:#cbd5e1; font-size:0.8rem; margin-bottom:2px; }
.val { color:#fff; font-size:1.1rem; font-weight:800; line-height:1.1; }
.chip {
  display:inline-block;
  margin-top:10px;
  padding:6px 12px;
  border-radius:999px;
  font-size:0.8rem;
  color:#a7f3d0;
  border:1px solid #16a34a33;
  background: transparent;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS (abbreviated)
# =========================
def american_to_implied(odds):
    try:
        x = float(odds)
    except Exception:
        return None
    if -99 < x < 100:
        return None
    if x > 0:
        return 100.0/(x+100.0)
    return abs(x)/(abs(x)+100.0)

def prob_to_american(p):
    if p <= 0 or p >= 1:
        return "N/A"
    dec = 1.0/p
    return f"+{int(round((dec-1)*100))}" if dec >= 2.0 else f"-{int(round(100/(dec-1)))}"

# =========================
# TABS
# =========================
tab_builder, tab_breakeven = st.tabs(["🧮 Parlay Builder", "🧷 Breakeven"])

# =========================
# TAB 1: PARLAY BUILDER
# =========================
with tab_builder:
    if "legs" not in st.session_state:
        st.session_state.legs = []

    st.markdown("### ⚙️ Filters")
    c1, c2 = st.columns([1, 1])
    with c1:
        seasons = st.multiselect("Seasons", ["2024-25","2023-24","2022-23"], default=["2024-25"])
    with c2:
        min_minutes = st.slider("Minimum Minutes", 0, 40, 20, 1)

    st.markdown("### 🏀 Input Bet")
    st.write("Add your props below in natural text form:")

    bet_text = st.text_input(
        "Input bet",
        placeholder="Maxey O 24.5 P Away -110 OR Embiid PRA U 35.5 -130",
        label_visibility="collapsed"
    )

    # (You can keep your existing logic for parsing and displaying legs here)

# =========================
# TAB 2: BREAKEVEN
# =========================
with tab_breakeven:
    st.subheader("🔎 Breakeven Finder")

    cA, cB, cC, cD, cE = st.columns([2,1,1,1,1])
    with cA:
        player_query = st.text_input("Player", placeholder="e.g., Stephen Curry")
    with cB:
        last_n = st.slider("Last N Games", 5, 100, 20, 1)
    with cC:
        min_min_b = st.slider("Min Minutes", 0, 40, 20, 1)
    with cD:
        loc_choice = st.selectbox("Location", ["All","Home Only","Away"], index=0)
    with cE:
        seasons_b = st.multiselect("Seasons", ["2024-25","2023-24","2022-23"], default=["2024-25"])

    do_search = st.button("Search")

    # (Keep rest of Breakeven logic the same — this only changes layout)
