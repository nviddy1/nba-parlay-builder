import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NBA Player Prop Parlay Builder", page_icon="🏀", layout="wide")
st.title("🏀 NBA Player Prop Parlay Builder")

# =========================
# DARK THEME + CLEAN INPUT FIX
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
  padding: 10px 16px !important;
  box-shadow: 0 6px 18px rgba(0,200,150,0.25) !important;
}

/* Expander styling */
.stExpander {
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  border-radius: 12px !important;
}
.streamlit-expanderHeader {
  font-weight: 800 !important;
  color: var(--text) !important;
}

/* Inputs: matte look */
input, select, textarea {
  background: #202225 !important;
  color: var(--text) !important;
  border: 1px solid #3b3f45 !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}

/* Remove dropdown / input focus bubbles */
.stSelectbox [data-baseweb="select"] > div:focus,
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stTextInput input:focus,
.stNumberInput input:focus {
    outline: none !important;
    box-shadow: none !important;
    border-color: #3b3f45 !important;
}
.stSelectbox [data-baseweb="select"]:focus-within {
    border-color: #3b3f45 !important;
    box-shadow: none !important;
}

/* Cards */
.card {
  --pad-x: 28px; --pad-y: 22px;
  padding: var(--pad-y) var(--pad-x);
  border-radius: 18px;
  margin: 14px 0 26px 0;
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  box-shadow: 0 0 14px rgba(0,0,0,0.25);
  width: 100%;
}
.neutral { --card-bg:#222; --card-border:#777; }
.pos     { --card-bg:#0b3d23; --card-border:#00FF99; }
.neg     { --card-bg:#3d0b0b; --card-border:#FF5555; }

.card h2 { color:#fff; margin:0 0 6px 0; font-weight:800; }
.cond { color:#a9b1bb; font-size:15px; margin: 2px 0 12px 0; }
.row { display:flex; flex-wrap:wrap; gap:16px; align-items:flex-end; justify-content:space-between; margin: 8px 0 6px 0; }
.m { min-width:140px; flex:1; }
.lab { color:#cbd5e1; font-size:14px; margin-bottom:4px; }
.val { color:#fff; font-size:28px; font-weight:800; line-height:1.1; }

.chip {
  display:inline-block; margin-top:12px;
  padding:8px 16px; border-radius:999px;
  font-size:14px; color:#a7f3d0; border:1px solid #16a34a33; background: transparent;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
STAT_LABELS = {"PTS":"Points","REB":"Rebounds","AST":"Assists","STL":"Steals","BLK":"Blocks","FG3M":"3PM"}

def get_player_id(name: str):
    res = players.find_players_by_full_name(name)
    return res[0]["id"] if res else None

def to_minutes(val):
    try:
        s = str(val)
        if ":" in s:
            return int(s.split(":")[0])
        return int(float(s))
    except:
        return 0

def fetch_gamelog(player_id: int, seasons: list[str]) -> pd.DataFrame:
    dfs = []
    for s in seasons:
        try:
            g = playergamelog.PlayerGameLog(player_id=player_id, season=s).get_data_frames()[0]
            dfs.append(g)
        except Exception:
            pass
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    for k in ["PTS","REB","AST","STL","BLK","FG3M"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    df["MIN_NUM"] = df["MIN"].apply(to_minutes) if "MIN" in df.columns else 0
    if "GAME_DATE" in df.columns:
        df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    else:
        df["GAME_DATE_DT"] = pd.Timestamp.now()
    return df

def american_to_implied(odds):
    if odds in (None, "", "0", 0): return None
    try: x = float(odds)
    except: return None
    return 100/(x+100) if x>0 else abs(x)/(abs(x)+100)

def prob_to_american(p: float):
    if p<=0 or p>=1: return "N/A"
    return f"{int(round((-100*p/(1-p)) if p>0.5 else (100*(1-p)/p))):+}"

def calc_prob(df: pd.DataFrame, stat: str, thr: int, min_minutes: int, loc_filter: str, range_key: str):
    if df.empty: return 0.0,0,0,df
    d = df.copy()
    d = d[d["MIN_NUM"] >= min_minutes]
    if loc_filter == "Home Only":
        d = d[d["MATCHUP"].astype(str).str.contains("vs", regex=False)]
    elif loc_filter == "Away Only":
        d = d[d["MATCHUP"].astype(str).str.contains("@", regex=False)]
    d = d.sort_values("GAME_DATE_DT", ascending=False)
    if range_key == "L10": d = d.head(10)
    elif range_key == "L20": d = d.head(20)
    total = len(d)
    if total==0 or stat not in d.columns: return 0.0,0,total,d
    hits = (d[stat] >= thr).sum()
    return hits/total, int(hits), int(total), d

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.subheader("⚙️ Filters")
    season_opts = ["2024-25","2023-24","2022-23"]
    seasons = st.multiselect("Seasons", season_opts, default=["2024-25"])
    min_minutes = st.slider("Minimum Minutes", 0, 40, 20, 1)

# =========================
# LEGS
# =========================
if "legs" not in st.session_state:
    st.session_state.legs = [{
        "player":"", "stat":"PTS", "thr":10, "odds":-110, "loc":"All", "range":"FULL"
    }]

pad_l, add_c, rem_c, pad_r = st.columns([1,1,1,1])
with add_c:
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({"player":"", "stat":"PTS", "thr":10, "odds":-110, "loc":"All", "range":"FULL"})
with rem_c:
    if st.button("➖ Remove Leg") and len(st.session_state.legs)>1:
        st.session_state.legs.pop()

stat_keys = list(STAT_LABELS.keys())
loc_opts = ["All","Home Only","Away Only"]
range_opts = ["FULL","L10","L20"]

def render_leg(idx: int, leg: dict, container):
    with container.expander(f"Leg {idx+1}", expanded=True):
        left, right = st.columns(2)
        with left:
            leg["player"] = st.text_input("Player", value=leg.get("player",""), key=f"p_{idx}")
            leg["loc"]    = st.selectbox("Home/Away", loc_opts,
                                         index=loc_opts.index(leg.get("loc","All")), key=f"l_{idx}")
            leg["range"]  = st.selectbox("Game Range", range_opts,
                                         index=range_opts.index(leg.get("range","FULL")), key=f"r_{idx}")
        with right:
            stat_default = leg.get("stat","PTS")
            leg["stat"]  = st.selectbox("Stat", stat_keys,
                                        index=stat_keys.index(stat_default) if stat_default in stat_keys else 0,
                                        format_func=lambda k: STAT_LABELS[k],
                                        key=f"s_{idx}")
            leg["thr"]   = st.number_input("Threshold (≥)", min_value=0, max_value=100,
                                           value=int(leg.get("thr",10)), key=f"t_{idx}")
            leg["odds"]  = st.number_input("FanDuel Odds", min_value=-10000, max_value=10000,
                                           value=int(leg.get("odds",-110)), step=5, key=f"o_{idx}")

for i in range(0, len(st.session_state.legs), 3):
    cols = st.columns(3)
    for j in range(3):
        k = i + j
        if k < len(st.session_state.legs):
            render_leg(k, st.session_state.legs[k], cols[j])

# =========================
# COMPUTE
# =========================
if st.button("Compute"):
    st.markdown("---")
    rows = []
    probs_for_parlay = []

    plt.rcParams.update({
        "axes.facecolor": "#1e1f22",
        "figure.facecolor": "#1e1f22",
        "text.color": "#ffffff",
        "axes.labelcolor": "#e5e7eb",
        "xtick.color": "#e5e7eb",
        "ytick.color": "#e5e7eb",
        "grid.color": "#374151",
    })

    for leg in st.session_state.legs:
        name = leg["player"].strip()
        stat = leg["stat"]; thr = int(leg["thr"]); book = int(leg["odds"])
        loc_key = leg.get("loc","All")
        range_key = leg.get("range","FULL")

        pid = get_player_id(name) if name else None
        if not pid:
            rows.append({"ok":False,"name":name or "Unknown","stat":stat,"thr":thr,"book":book})
            continue

        df = fetch_gamelog(pid, seasons)
        p, hits, total, df_filt = calc_prob(df, stat, thr, min_minutes, loc_key, range_key)
        fair = prob_to_american(p)
        book_prob = american_to_implied(book)
        ev = None if book_prob is None else (p - book_prob) * 100.0
        if p>0: probs_for_parlay.append(p)

        rows.append({
            "ok":True, "name":name, "stat":stat, "thr":thr, "book":book,
            "p":p, "hits":hits, "total":total, "fair":fair, "book_prob":book_prob,
            "ev":ev, "df":df_filt, "loc":loc_key, "range":range_key
        })

    combined_p = float(np.prod(probs_for_parlay)) if probs_for_parlay else 0.0
    combined_fair = prob_to_american(combined_p) if combined_p>0 else "N/A"
    entered_prob = american_to_implied(0)
    parlay_ev = None if entered_prob is None else (combined_p - entered_prob) * 100.0
    cls = "neutral"
    if parlay_ev is not None: cls = "pos" if parlay_ev >= 0 else "neg"

    st.markdown(f"""
<div class="card {cls}">
  <h2>💥 Combined Parlay — {', '.join(seasons) if seasons else '—'}</h2>
  <div class="cond">Includes all selected legs with their individual filters</div>
</div>
""", unsafe_allow_html=True)
