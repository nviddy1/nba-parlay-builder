import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

st.set_page_config(page_title="NBA Player Prop Parlay Builder", page_icon="🏀", layout="wide")
st.title("🏀 NBA Player Prop Parlay Builder")

# =========================
# STYLES
# =========================
st.markdown("""
<style>
:root {
  --bg: #0e0f11;
  --text: #f9fafb;
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

/* Expanders */
.stExpander {
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  border-radius: 12px !important;
}
.streamlit-expanderHeader { font-weight: 800 !important; color: var(--text) !important; }

/* Inputs */
input, select, textarea {
  background: #202225 !important;
  color: var(--text) !important;
  border: 1px solid #3b3f45 !important;
  border-radius: 8px !important;
  font-size: 15px !important;
  padding: 8px !important;
}

/* Remove focus */
.block-container *:focus,
.block-container *:focus-visible {
  outline: none !important;
  box-shadow: none !important;
}

/* Dropdown style */
.stSelectbox div[data-baseweb="select"] {
  min-width: 110px !important;
  min-height: 44px !important;
  font-size: 15px !important;
}

/* Smaller threshold box */
.stNumberInput input {
  min-height: 40px !important;
  width: 80px !important;
  font-size: 15px !important;
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
  font-size:14px; color:#a7f3d0; border:1px solid #16a34a33;
  background: transparent;
}
</style>

<!-- Disable typing in select boxes -->
<script>
window.addEventListener('load', function() {
  const observer = new MutationObserver(() => {
    document.querySelectorAll('input[type="text"]').forEach(inp => {
      const ph = inp.getAttribute('placeholder') || '';
      if (['Points','Rebounds','Assists','Steals','Blocks','3PM','All','Home Only','Away Only','FULL','L10','L20','O','U'].some(v => ph.includes(v))) {
        inp.setAttribute('readonly', true);
        inp.style.pointerEvents = 'none';
        inp.style.userSelect = 'none';
        inp.style.caretColor = 'transparent';
      }
    });
  });
  observer.observe(document.body, { childList: true, subtree: true });
});
</script>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
STAT_LABELS = {"PTS":"Points","REB":"Rebounds","AST":"Assists","STL":"Steals","BLK":"Blocks","FG3M":"3PM"}

def get_player_id(name):
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

def fetch_gamelog(player_id, seasons):
    dfs = []
    for s in seasons:
        try:
            g = playergamelog.PlayerGameLog(player_id=player_id, season=s).get_data_frames()[0]
            dfs.append(g)
        except:
            pass
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    for k in ["PTS","REB","AST","STL","BLK","FG3M"]:
        if k in df.columns: df[k] = pd.to_numeric(df[k], errors="coerce")
    df["MIN_NUM"] = df["MIN"].apply(to_minutes) if "MIN" in df.columns else 0
    df["GAME_DATE_DT"] = pd.to_datetime(df.get("GAME_DATE", pd.Timestamp.now()), errors="coerce")
    return df

def american_to_implied(odds):
    if odds in (None, "", "0", 0): return None
    try: x = float(odds)
    except: return None
    return 100/(x+100) if x>0 else abs(x)/(abs(x)+100)

def prob_to_american(p):
    if p<=0 or p>=1: return "N/A"
    return f"{int(round((-100*p/(1-p)) if p>0.5 else (100*(1-p)/p))):+}"

def calc_prob(df, stat, thr, min_minutes, loc_filter, range_key, direction):
    if df.empty: return 0.0,0,0,df
    d = df[df["MIN_NUM"] >= min_minutes]
    if loc_filter == "Home Only":
        d = d[d["MATCHUP"].astype(str).str.contains("vs")]
    elif loc_filter == "Away Only":
        d = d[d["MATCHUP"].astype(str).str.contains("@")]
    d = d.sort_values("GAME_DATE_DT", ascending=False)
    if range_key == "L10": d = d.head(10)
    elif range_key == "L20": d = d.head(20)
    total = len(d)
    if total==0 or stat not in d.columns: return 0.0,0,total,d
    hits = (d[stat] >= thr).sum() if direction=="O" else (d[stat] <= thr).sum()
    return hits/total, hits, total, d

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.subheader("⚙️ Filters")
    seasons = st.multiselect("Seasons", ["2024-25","2023-24","2022-23"], default=["2024-25"])
    min_minutes = st.slider("Minimum Minutes", 0, 40, 20, 1)

# =========================
# STATE
# =========================
if "legs" not in st.session_state:
    st.session_state.legs = [{"player":"","stat":"PTS","dir":"O","thr":10,"odds":-110,"loc":"All","range":"FULL"}]

c1, c2 = st.columns(2)
with c1:
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({"player":"","stat":"PTS","dir":"O","thr":10,"odds":-110,"loc":"All","range":"FULL"})
with c2:
    if st.button("➖ Remove Leg") and len(st.session_state.legs)>1:
        st.session_state.legs.pop()

# =========================
# RENDER LEGS
# =========================
def render_leg(i, leg, col):
    with col.expander(f"Leg {i+1}", expanded=True):
        left, right = st.columns(2)
        with left:
            leg["player"] = st.text_input("Player", leg.get("player",""), key=f"p{i}")
            leg["loc"] = st.selectbox("Home/Away", ["All","Home Only","Away Only"], key=f"l{i}")
            leg["range"] = st.selectbox("Game Range", ["FULL","L10","L20"], key=f"r{i}")
        with right:
            leg["stat"] = st.selectbox("Stat", list(STAT_LABELS.keys()), format_func=lambda k: STAT_LABELS[k], key=f"s{i}")
            c_dir, c_thr = st.columns([1,2])
            with c_dir:
                leg["dir"] = st.selectbox("O/U", ["O","U"], key=f"d{i}")
            with c_thr:
                leg["thr"] = st.number_input("Threshold", min_value=0, max_value=100, value=int(leg.get("thr",10)), key=f"t{i}")
            leg["odds"] = st.number_input("FanDuel Odds", min_value=-10000, max_value=10000, value=int(leg.get("odds",-110)), step=5, key=f"o{i}")

for i in range(0, len(st.session_state.legs), 3):
    cols = st.columns(3)
    for j in range(3):
        if i+j < len(st.session_state.legs):
            render_leg(i+j, st.session_state.legs[i+j], cols[j])

# =========================
# COMPUTE (placeholder)
# =========================
if st.button("Compute"):
    st.success("✅ Ready — compute logic will run here.")
