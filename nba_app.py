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
# CONSTANTS & HELPERS
# =========================
STAT_LABELS = {
  "PTS": "Points", "REB": "Rebounds", "AST": "Assists", "STL": "Steals", "BLK": "Blocks",
  "FG3M": "3PM", "DOUBDOUB": "Double-Double", "TRIPDOUB": "Triple-Double",
  "P+R": "P+R", "P+A": "P+A", "R+A": "R+A", "PRA": "PRA"
}

STAT_TOKENS = {
  "P": "PTS", "R": "REB", "A": "AST", "PTS": "PTS", "REB": "REB", "AST": "AST",
  "STL": "STL", "BLK": "BLK", "3PM": "FG3M", "FG3M": "FG3M",
  "P+R": "P+R", "R+A": "R+A", "P+A": "P+A", "PRA": "PRA",
  "DD": "DOUBDOUB", "TD": "TRIPDOUB"
}

@st.cache_data
def get_all_player_names():
    active = players.get_active_players()
    names = [p["full_name"] for p in active if p.get("full_name")]
    return sorted(set(names))

PLAYER_LIST = get_all_player_names()

def best_player_match(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    m = process.extractOne(q, PLAYER_LIST, score_cutoff=70)
    return m[0] if m else ""

def american_to_implied(odds):
    try: x = float(odds)
    except Exception: return None
    if -99 < x < 100: return None
    if x > 0: return 100.0 / (x + 100.0)
    return abs(x) / (abs(x) + 100.0)

def prob_to_american(p):
    if p <= 0 or p >= 1: return "N/A"
    dec = 1.0 / p
    return f"+{int(round((dec - 1) * 100))}" if dec >= 2.0 else f"-{int(round(100 / (dec - 1)))}"

def fmt_half(x):
    try: v = float(x); return f"{v:.1f}".rstrip("0").rstrip(".")
    except Exception: return str(x)

# =========================
# GAME LOG / COMPUTATION
# =========================
def get_player_id(full_name: str):
    if not full_name:
        return None
    res = players.find_players_by_full_name(full_name)
    return res[0]["id"] if res else None

def to_minutes(val):
    try:
        s = str(val)
        if ":" in s: return int(s.split(":")[0])
        return int(float(s))
    except Exception:
        return 0

def fetch_gamelog(player_id: int, seasons: list[str]) -> pd.DataFrame:
    dfs = []
    for s in seasons:
        try:
            g = playergamelog.PlayerGameLog(player_id=player_id, season=s).get_data_frames()[0]
            dfs.append(g)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    for k in ["PTS","REB","AST","STL","BLK","FG3M"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    df["MIN_NUM"] = df["MIN"].apply(to_minutes)
    df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df

def compute_stat_series(df: pd.DataFrame, stat_code: str) -> pd.Series:
    if stat_code == "P+R": return df["PTS"] + df["REB"]
    if stat_code == "P+A": return df["PTS"] + df["AST"]
    if stat_code == "R+A": return df["REB"] + df["AST"]
    if stat_code == "PRA": return df["PTS"] + df["REB"] + df["AST"]
    return df[stat_code] if stat_code in df.columns else pd.Series(dtype=float)

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
    filter_cols = st.columns([1, 1, 3])
    with filter_cols[0]:
        seasons = st.multiselect(
            "Season(s)",
            ["2024-25", "2023-24", "2022-23"],
            default=["2024-25"],
            key="builder_seasons"
        )
    with filter_cols[1]:
        min_minutes = st.slider(
            "Min Minutes",
            min_value=0,
            max_value=40,
            value=20,
            step=1,
            key="builder_min_minutes"
        )

    st.markdown("### 🏀 Input Bet")
    bet_text = st.text_input(
        "Input bet",
        placeholder="Maxey O 24.5 P Away -110 OR Embiid PRA U 35.5 -130",
        label_visibility="collapsed",
        key="builder_input_bet"
    )

    # Placeholder for your parsing and computation logic
    if bet_text:
        st.success(f"✅ Parsed input: {bet_text}")

# =========================
# TAB 2: BREAKEVEN
# =========================
with tab_breakeven:
    st.subheader("🔎 Breakeven Finder")

    cA, cB, cC, cD, cE = st.columns([2, 1, 1, 1, 1])
    with cA:
        player_query = st.text_input(
            "Player",
            placeholder="e.g., Stephen Curry",
            key="breakeven_player"
        )
    with cB:
        last_n = st.slider(
            "Last N Games",
            min_value=5,
            max_value=100,
            value=20,
            step=1,
            key="breakeven_lastn"
        )
    with cC:
        min_min_b = st.slider(
            "Min Minutes",
            min_value=0,
            max_value=40,
            value=20,
            step=1,
            key="breakeven_min_minutes"
        )
    with cD:
        loc_choice = st.selectbox(
            "Location",
            ["All", "Home Only", "Away"],
            index=0,
            key="breakeven_location"
        )
    with cE:
        seasons_b = st.multiselect(
            "Season(s)",
            ["2024-25", "2023-24", "2022-23"],
            default=["2024-25"],
            key="breakeven_seasons"
        )

    do_search = st.button("Search", key="breakeven_search")

    if do_search:
        player_name = best_player_match(player_query)
        if not player_name:
            st.warning("Could not match that player. Try a more specific name.")
        else:
            pid = get_player_id(player_name)
            if not pid:
                st.warning("No player ID found for that name.")
            else:
                df = fetch_gamelog(pid, seasons_b)
                if df.empty:
                    st.warning("No game logs found.")
                else:
                    st.success(f"Found {len(df)} games for {player_name}")
