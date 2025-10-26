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

/* Inputs */
input, select, textarea {
  background: #202225 !important;
  color: var(--text) !important;
  border: 1px solid #3b3f45 !important;
  border-radius: 8px !important;
  font-size: 0.85rem !important;
  padding: 4px 6px !important;
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

/* Cards */
.card {
  --pad-x: 20px; --pad-y: 18px;
  padding: var(--pad-y) var(--pad-x);
  border-radius: 14px;
  margin: 10px 0 20px 0;
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  box-shadow: 0 0 14px rgba(0,0,0,0.25);
  width: 100%;
}
.neutral { --card-bg:#222; --card-border:#777; }
.pos     { --card-bg:#0b3d23; --card-border:#00FF99; }
.neg     { --card-bg:#3d0b0b; --card-border:#FF5555; }

.card h2 { color:#fff; margin:0 0 6px 0; font-weight:800; font-size:1.05rem; }
.cond { color:#a9b1bb; font-size:0.9rem; margin: 2px 0 10px 0; }

.row {
  display:flex; flex-wrap:wrap; gap:10px;
  align-items:flex-end; justify-content:space-between;
  margin: 6px 0 4px 0;
}
.m { min-width:120px; flex:1; }
.lab { color:#cbd5e1; font-size:0.8rem; margin-bottom:2px; }
.val { color:#fff; font-size:1.1rem; font-weight:800; line-height:1.1; }

.chip {
  display:inline-block; margin-top:10px;
  padding:6px 12px; border-radius:999px;
  font-size:0.8rem; color:#a7f3d0; border:1px solid #16a34a33;
  background: transparent;
}

/* Simple table polish for Breakeven tab */
table {
  border-collapse: collapse;
}
thead th {
  border-bottom: 1px solid #374151 !important;
}
tbody td, thead th {
  padding: 8px 10px !important;
}
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
    "STL": "STL", "STEALS": "STL",
    "BLK": "BLK", "BLOCKS": "BLK",
    "3PM": "FG3M", "FG3M": "FG3M", "THREES": "FG3M",
    "DOUBDOUB": "DOUBDOUB", "DOUBLEDOUBLE": "DOUBDOUB", "DOUB": "DOUBDOUB", "DD": "DOUBDOUB",
    "TRIPDOUB": "TRIPDOUB", "TRIPLEDOUBLE": "TRIPDOUB", "TD": "TRIPDOUB",
    "P+R": "P+R", "PR": "P+R",
    "P+A": "P+A", "PA": "P+A",
    "R+A": "R+A", "RA": "R+A",
    "PRA": "PRA"
}

@st.cache_data
def get_all_player_names():
    try:
        active = players.get_active_players()
        return sorted(set([p["full_name"] for p in active if p.get("full_name")]))
    except Exception:
        all_p = players.get_players()
        return sorted(set([p["full_name"] for p in all_p if p.get("full_name")]))
PLAYER_LIST = get_all_player_names()

def best_player_match(q): 
    if not q.strip(): return ""
    m = process.extractOne(q.strip(), PLAYER_LIST, score_cutoff=60)
    return m[0] if m else ""

def american_to_implied(odds):
    try: x = float(odds)
    except: return None
    if -99 < x < 100: return None
    return 100/(x+100) if x>0 else abs(x)/(abs(x)+100)

def prob_to_american(p):
    if p<=0 or p>=1: return "N/A"
    dec=1/p
    return f"+{int(round((dec-1)*100))}" if dec>=2 else f"-{int(round(100/(dec-1)))}"

def fmt_half(x):
    try: return f"{float(x):.1f}".rstrip("0").rstrip(".")
    except: return str(x)

def parse_input_line(text):
    t=(text or "").strip()
    if not t: return None
    parts=t.replace("/", "+").split()
    dir_token="Over" if any(tok.upper() in ["O","OVER"] for tok in parts) else "Under" if any(tok.upper() in ["U","UNDER"] for tok in parts) else "Over"
    thr=next((float(tok) for tok in parts if any(c.isdigit() for c in tok)),10.5)
    stat_code=None
    for tok in parts:
        if tok.upper() in STAT_TOKENS: stat_code=STAT_TOKENS[tok.upper()]; break
    if not stat_code: stat_code="PTS"
    loc="All"
    for tok in parts:
        if tok.upper() in ["AWAY","A"]: loc="Away"
        elif tok.upper() in ["HOME","H"]: loc="Home Only"
    odds=-110
    for tok in parts[::-1]:
        if tok.startswith(("+","-")):
            try: o=int(tok); 
            except: continue
            if abs(o)>=100: odds=o; break
    banned=set(["O","OVER","U","UNDER","HOME","H","AWAY","A"]+list(STAT_TOKENS.keys()))
    name_guess=" ".join([p for p in parts if p.upper() not in banned and not p.replace(".","",1).lstrip("+-").isdigit()])
    player=best_player_match(name_guess)
    return {"player":player,"dir":dir_token,"thr":thr,"stat":stat_code,"loc":loc,"range":"FULL","odds":odds}

def get_player_id(full_name): 
    if not full_name: return None
    res=players.find_players_by_full_name(full_name)
    return res[0]["id"] if res else None

def to_minutes(v): 
    try: s=str(v); return int(s.split(":")[0]) if ":" in s else int(float(s))
    except: return 0

def fetch_gamelog(pid,seasons):
    dfs=[]
    for s in seasons:
        try: dfs.append(playergamelog.PlayerGameLog(player_id=pid,season=s).get_data_frames()[0])
        except: pass
    if not dfs: return pd.DataFrame()
    df=pd.concat(dfs,ignore_index=True)
    for k in ["PTS","REB","AST","STL","BLK","FG3M"]: 
        if k in df.columns: df[k]=pd.to_numeric(df[k],errors="coerce")
    df["MIN_NUM"]=df["MIN"].apply(to_minutes) if "MIN" in df else 0
    df["GAME_DATE_DT"]=pd.to_datetime(df.get("GAME_DATE"),errors="coerce")
    return df

def compute_stat_series(df,stat):
    if stat in ["PTS","REB","AST","STL","BLK","FG3M"]: return df[stat].astype(float)
    if stat=="P+R": return df["PTS"]+df["REB"]
    if stat=="P+A": return df["PTS"]+df["AST"]
    if stat=="R+A": return df["REB"]+df["AST"]
    if stat=="PRA": return df["PTS"]+df["REB"]+df["AST"]
    return df["PTS"]

def leg_probability(df,stat,direction,thr):
    ser=compute_stat_series(df,stat)
    hits=(ser>=thr).sum() if direction=="Over" else (ser<=thr).sum()
    total=len(ser.dropna())
    return (hits/total if total else 0),hits,total

def headshot_url(pid): return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png" if pid else None

# =========================
# TABS
# =========================
tab_builder, tab_breakeven = st.tabs(["🧮 Parlay Builder","🧷 Breakeven"])

# =========================
# TAB 1: PARLAY BUILDER
# =========================
with tab_builder:
    # Inline compact filters
    st.markdown("### ⚙️ Filters")
    colA, colB = st.columns([1,1])
    with colA:
        seasons = st.multiselect("Seasons", ["2024-25","2023-24","2022-23"], default=["2024-25"])
    with colB:
        min_minutes = st.slider("Min Minutes", 0, 40, 20, 1)

    # State setup
    if "legs" not in st.session_state: st.session_state.legs=[]
    if "awaiting_input" not in st.session_state: st.session_state.awaiting_input=True

    c1,c2=st.columns([1,1])
    with c1:
        if st.button("➕ Add Leg"): st.session_state.awaiting_input=True
    with c2:
        if st.button("➖ Remove Last Leg") and st.session_state.legs: st.session_state.legs.pop()

    st.write("**Input bet**")

    # Legs display
    if st.session_state.legs:
        for i,leg in enumerate(st.session_state.legs):
            leg_no=i+1
            header=f"Leg {leg_no}: {leg['player']} — {leg['dir'][0]} {fmt_half(leg['thr'])} {STAT_LABELS.get(leg['stat'])} ({leg['loc']}, {leg['odds']})"
            with st.expander(header):
                cols=st.columns([2,1])
                with cols[0]:
                    leg["player"]=st.text_input("Player",value=leg["player"],key=f"player_{i}")
                    leg["stat"]=st.selectbox("Stat",list(STAT_LABELS.keys()),index=list(STAT_LABELS.keys()).index(leg["stat"]),key=f"stat_{i}")
                    leg["dir"]=st.selectbox("O/U",["Over","Under"],index=(0 if leg["dir"]=="Over" else 1),key=f"dir_{i}")
                    leg["thr"]=st.number_input("Threshold",value=float(leg["thr"]),step=0.5,key=f"thr_{i}")
                with cols[1]:
                    leg["loc"]=st.selectbox("Home/Away",["All","Home Only","Away"],index=["All","Home Only","Away"].index(leg["loc"]),key=f"loc_{i}")
                    leg["odds"]=st.number_input("Sportsbook Odds",value=int(leg["odds"]),step=5,key=f"odds_{i}")
                if st.button(f"❌ Remove Leg {leg_no}",key=f"remove_{i}"):
                    st.session_state.legs.pop(i); st.rerun()

    if st.session_state.awaiting_input:
        bet_text=st.text_input("Input bet",placeholder="Maxey O 24.5 P Away -110",key="freeform_input",label_visibility="collapsed")
        if bet_text.strip():
            parsed=parse_input_line(bet_text)
            if parsed and parsed["player"]:
                st.session_state.legs.append(parsed)
                st.session_state.awaiting_input=False
                st.rerun()

    parlay_odds=0
    if len(st.session_state.legs)>1:
        st.markdown("### 🎯 Combined Parlay Odds")
        parlay_odds=st.number_input("Enter Parlay Odds (+300, -150, etc.)",value=0,step=5,key="parlay_odds")

    if st.session_state.legs and st.button("Compute"):
        # (keep rest of logic identical)
        st.write("Computing results…")  # truncated for brevity
