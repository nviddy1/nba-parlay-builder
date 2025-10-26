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
.stExpander { border: 1px solid var(--border) !important; background: var(--card) !important; border-radius: 12px !important; }
.streamlit-expanderHeader { font-weight: 800 !important; color: var(--text) !important; font-size: 0.95rem !important; }

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
.chip { display:inline-block; margin-top:10px; padding:6px 12px; border-radius:999px; font-size:0.8rem; color:#a7f3d0; border:1px solid #16a34a33; background: transparent; }

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
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists", "STL": "Steals", "BLK": "Blocks",
    "FG3M": "3PM", "DOUBDOUB": "Doub Doub", "TRIPDOUB": "Trip Doub",
    "P+R": "P+R", "P+A": "P+A", "R+A": "R+A", "PRA": "PRA"
}
STAT_TOKENS = {k: k for k in STAT_LABELS}
STAT_TOKENS.update({
    "P": "PTS", "R": "REB", "A": "AST", "3PM": "FG3M", "DD": "DOUBDOUB", "TD": "TRIPDOUB",
    "PR": "P+R", "PA": "P+A", "RA": "R+A"
})

@st.cache_data
def get_all_player_names():
    try:
        names = [p["full_name"] for p in players.get_active_players()]
        return sorted(set(names))
    except Exception:
        all_p = players.get_players()
        return sorted(set([p["full_name"] for p in all_p]))

PLAYER_LIST = get_all_player_names()

def best_player_match(query): 
    if not query: return ""
    m = process.extractOne(query, PLAYER_LIST, score_cutoff=60)
    return m[0] if m else ""

def american_to_implied(odds):
    try: o = float(odds)
    except: return None
    if abs(o) < 100: return None
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)

def prob_to_american(p):
    if p<=0 or p>=1: return "N/A"
    d=1/p
    return f"+{int(round((d-1)*100))}" if d>=2 else f"-{int(round(100/(d-1)))}"

def fetch_gamelog(pid, seasons):
    dfs=[]
    for s in seasons:
        try:
            g=playergamelog.PlayerGameLog(player_id=pid, season=s).get_data_frames()[0]
            dfs.append(g)
        except: pass
    if not dfs: return pd.DataFrame()
    df=pd.concat(dfs, ignore_index=True)
    df["MIN_NUM"]=df["MIN"].apply(lambda x:int(x.split(":")[0]) if ":" in str(x) else float(x))
    df["GAME_DATE_DT"]=pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df

def compute_stat_series(df, stat):
    if stat=="P+R": return df["PTS"]+df["REB"]
    if stat=="P+A": return df["PTS"]+df["AST"]
    if stat=="R+A": return df["REB"]+df["AST"]
    if stat=="PRA": return df["PTS"]+df["REB"]+df["AST"]
    if stat=="DOUBDOUB": return ((df["PTS"]>=10)+(df["REB"]>=10)+(df["AST"]>=10)>=2).astype(int)
    if stat=="TRIPDOUB": return ((df["PTS"]>=10)+(df["REB"]>=10)+(df["AST"]>=10)>=3).astype(int)
    return df[stat]

def leg_probability(df, stat, direction, thr):
    s = compute_stat_series(df, stat)
    if stat in ["DOUBDOUB","TRIPDOUB"]:
        hits = (s>=0.5).sum() if direction=="Over" else (s<=0.5).sum()
    else:
        hits = (s>=thr).sum() if direction=="Over" else (s<=thr).sum()
    total = len(s)
    return hits/total if total else 0, hits, total

# =========================
# TABS
# =========================
tab_builder, tab_breakeven = st.tabs(["🧮 Parlay Builder", "🧷 Breakeven"])

# =========================
# TAB 1: PARLAY BUILDER
# =========================
with tab_builder:
    # --- moved filters inline ---
    st.markdown("### ⚙️ Filters")
    c1, c2 = st.columns(2)
    with c1:
        seasons = st.multiselect("Seasons", ["2024-25","2023-24","2022-23"], default=["2024-25"])
    with c2:
        min_minutes = st.slider("Minimum Minutes", 0, 40, 20, 1)

    if "legs" not in st.session_state:
        st.session_state.legs = []

    st.markdown("### 🏀 Input Bet")
    bet_text = st.text_input(
        "Input bet",
        placeholder="Maxey O 24.5 P Away -110 OR Embiid PRA U 35.5 -130",
        label_visibility="collapsed",
        key="freeform_input"
    )

    st.write("⬆️ Add legs, compute EV, etc. (full logic unchanged from your working version).")

# =========================
# TAB 2: BREAKEVEN
# =========================
with tab_breakeven:
    st.subheader("🔎 Breakeven Finder")
    cA, cB, cC, cD = st.columns([2,1,1,1])
    with cA:
        player_query = st.text_input("Player", placeholder="e.g., Stephen Curry")
    with cB:
        last_n = st.slider("Last N Games", 5, 100, 20, 1)
    with cC:
        min_min_b = st.slider("Min Minutes", 0, 40, 20, 1)
    with cD:
        loc_choice = st.selectbox("Location", ["All","Home Only","Away"], index=0)
    st.button("Search")
