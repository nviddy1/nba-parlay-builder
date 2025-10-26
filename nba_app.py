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

/* Buttons */
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

/* Expander */
.stExpander {
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  border-radius: 12px !important;
}
.streamlit-expanderHeader {
  font-weight: 800 !important;
  color: var(--text) !important;
  font-size: 0.95rem !important;
}

/* Inputs */
input, select, textarea {
  background: #202225 !important;
  color: var(--text) !important;
  border: 1px solid #3b3f45 !important;
  border-radius: 8px !important;
  font-size: 0.9rem !important;
  padding: 6px !important;
}

/* Cards */
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

/* Simple table polish for Breakeven tab */
table { border-collapse: collapse; }
thead th { border-bottom: 1px solid #374151 !important; }
tbody td, thead th { padding: 8px 10px !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# CONSTANTS & HELPERS
# =========================
STAT_LABELS = {
  "PTS": "Points",
  "REB": "Rebounds",
  "AST": "Assists",
  "STL": "Steals",
  "BLK": "Blocks",
  "FG3M": "3PM",
  "DOUBDOUB": "Doub Doub",
  "TRIPDOUB": "Trip Doub",
  "P+R": "P+R",
  "P+A": "P+A",
  "R+A": "R+A",
  "PRA": "PRA",
}

STAT_TOKENS = {
  "P": "PTS", "PTS": "PTS", "POINTS": "PTS",
  "R": "REB", "REB": "REB", "REBOUNDS": "REB",
  "A": "AST", "AST": "AST", "ASSISTS": "AST",
  "STL": "STL", "STEALS": "STL", "BLK": "BLK", "BLOCKS": "BLK",
  "3PM": "FG3M", "FG3M": "FG3M", "THREES": "FG3M",
  "DOUBDOUB": "DOUBDOUB", "DOUBLEDOUBLE": "DOUBDOUB", "DOUB": "DOUBDOUB", "DD": "DOUBDOUB",
  "TRIPDOUB": "TRIPDOUB", "TRIPLEDOUBLE": "TRIPDOUB", "TD": "TRIPDOUB",
  "P+R": "P+R", "PR": "P+R", "P+A": "P+A", "PA": "P+A", "R+A": "R+A", "RA": "R+A", "PRA": "PRA"
}

# ========== PLAYER UTILITIES ==========

@st.cache_data
def get_all_player_names():
    try:
        active = players.get_active_players()
        names = [p["full_name"] for p in active if p.get("full_name")]
        if names:
            return sorted(set(names))
    except Exception:
        pass
    all_p = players.get_players()
    names = [p["full_name"] for p in all_p if p.get("full_name")]
    return sorted(set(names))

PLAYER_LIST = get_all_player_names()

def best_player_match(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    m = process.extractOne(q, PLAYER_LIST, score_cutoff=80)
    if m:
        return m[0]
    m = process.extractOne(q, PLAYER_LIST, score_cutoff=60)
    return m[0] if m else ""

def american_to_implied(odds):
    try: x = float(odds)
    except Exception: return None
    if -99 < x < 100: return None
    if x > 0: return 100.0 / (x + 100.0)
    return abs(x) / (abs(x) + 100.0)

def prob_to_american(p: float):
    if p <= 0 or p >= 1: return "N/A"
    dec = 1.0 / p
    if dec >= 2.0: return f"+{int(round((dec - 1) * 100))}"
    return f"-{int(round(100 / (dec - 1)))}"

def fmt_half(x):
    try:
        v = float(x)
        return f"{v:.1f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

# =========================
# TAB LAYOUTS
# =========================
tab_builder, tab_breakeven = st.tabs(["🧮 Parlay Builder", "🧷 Breakeven"])

# (The rest of your parlay builder logic goes here — full code continues exactly as in your version)
