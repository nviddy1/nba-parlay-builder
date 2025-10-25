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
    "P+R": "P+R", "P+A": "P+A", "R+A": "R+A", "PRA": "PRA",
}

STAT_TOKENS = {
    "P":"PTS","PTS":"PTS","POINTS":"PTS","R":"REB","REB":"REB","REBOUNDS":"REB","A":"AST",
    "AST":"AST","ASSISTS":"AST","STL":"STL","STEALS":"STL","BLK":"BLK","BLOCKS":"BLK",
    "3PM":"FG3M","FG3M":"FG3M","THREES":"FG3M","DOUBDOUB":"DOUBDOUB","DOUBLEDOUBLE":"DOUBDOUB",
    "DD":"DOUBDOUB","TRIPDOUB":"TRIPDOUB","TRIPLEDOUBLE":"TRIPDOUB","TD":"TRIPDOUB",
    "P+R":"P+R","PR":"P+R","P+A":"P+A","PA":"P+A","R+A":"R+A","RA":"R+A","PRA":"PRA"
}

@st.cache_data
def get_all_player_names():
    try:
        active = players.get_active_players()
        names = [p["full_name"] for p in active if p.get("full_name")]
        if names: return sorted(set(names))
    except Exception: pass
    all_p = players.get_players()
    names = [p["full_name"] for p in all_p if p.get("full_name")]
    return sorted(set(names))

PLAYER_LIST = get_all_player_names()

def best_player_match(q: str) -> str:
    q = (q or "").strip()
    if not q: return ""
    m = process.extractOne(q, PLAYER_LIST, score_cutoff=80)
    if m: return m[0]
    m = process.extractOne(q, PLAYER_LIST, score_cutoff=60)
    return m[0] if m else ""

def american_to_implied(odds):
    try: x = float(odds)
    except Exception: return None
    if -99 < x < 100: return None
    return 100/(x+100) if x>0 else abs(x)/(abs(x)+100)

def prob_to_american(p):
    if p<=0 or p>=1: return "N/A"
    dec=1/p
    return f"+{int(round((dec-1)*100))}" if dec>=2 else f"-{int(round(100/(dec-1)))}"

def fmt_half(x):
    try: v=float(x); return f"{v:.1f}".rstrip("0").rstrip(".")
    except Exception: return str(x)

def get_player_id(name):
    if not name: return None
    res=players.find_players_by_full_name(name)
    return res[0]["id"] if res else None

def to_minutes(v):
    try:
        s=str(v)
        return int(s.split(":")[0]) if ":" in s else int(float(s))
    except Exception: return 0

def fetch_gamelog(pid, seasons):
    dfs=[]
    for s in seasons:
        try:
            g=playergamelog.PlayerGameLog(player_id=pid, season=s).get_data_frames()[0]
            dfs.append(g)
        except Exception: pass
    if not dfs: return pd.DataFrame()
    df=pd.concat(dfs, ignore_index=True)
    for k in ["PTS","REB","AST","STL","BLK","FG3M"]:
        if k in df.columns: df[k]=pd.to_numeric(df[k], errors="coerce")
    df["MIN_NUM"]=df["MIN"].apply(to_minutes) if "MIN" in df.columns else 0
    df["GAME_DATE_DT"]=pd.to_datetime(df.get("GAME_DATE", pd.Timestamp.now()), errors="coerce")
    return df

def compute_stat_series(df, stat):
    if stat in ["PTS","REB","AST","STL","BLK","FG3M"]: return df[stat].astype(float)
    if stat=="P+R": return (df["PTS"]+df["REB"]).astype(float)
    if stat=="P+A": return (df["PTS"]+df["AST"]).astype(float)
    if stat=="R+A": return (df["REB"]+df["AST"]).astype(float)
    if stat=="PRA": return (df["PTS"]+df["REB"]+df["AST"]).astype(float)
    if stat=="DOUBDOUB":
        pts=(df["PTS"]>=10).astype(int); reb=(df["REB"]>=10).astype(int); ast=(df["AST"]>=10).astype(int)
        return ((pts+reb+ast)>=2).astype(int)
    if stat=="TRIPDOUB":
        pts=(df["PTS"]>=10).astype(int); reb=(df["REB"]>=10).astype(int); ast=(df["AST"]>=10).astype(int)
        return ((pts+reb+ast)>=3).astype(int)
    return df["PTS"].astype(float)

def leg_probability(df, stat, direction, thr):
    s=compute_stat_series(df, stat)
    if stat in ["DOUBDOUB","TRIPDOUB"]:
        hits=int((s<=0.5).sum()) if direction=="Under" else int((s>=0.5).sum())
    else:
        hits=int((s<=thr).sum()) if direction=="Under" else int((s>=thr).sum())
    total=int(s.notna().sum())
    return hits/total if total else 0, hits, total

def headshot_url(pid): return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png" if pid else None

def breakeven_for_stat(series):
    s=pd.to_numeric(series, errors="coerce").dropna()
    total=len(s)
    if total==0: return {"line":None,"over_prob":None,"under_prob":None,"over_odds":"N/A","under_odds":"N/A"}
    lo,hi=np.floor(s.min())-0.5, np.ceil(s.max())+0.5
    candidates=np.arange(lo, hi+0.001, 0.5)
    best_t,best_gap,best_over=None,1.0,None
    for t in candidates:
        over=(s>=t).mean(); gap=abs(over-0.5)
        if gap<best_gap or best_t is None: best_t,best_gap,best_over=t,gap,over
    over_p=float(best_over); under_p=1-over_p
    return {"line":float(best_t),"over_prob":over_p,"under_prob":under_p,
            "over_odds":prob_to_american(over_p),"under_odds":prob_to_american(under_p)}

# =========================
# TABS
# =========================
tab_builder, tab_breakeven = st.tabs(["🧮 Parlay Builder", "🧷 Breakeven"])

# =========================
# TAB 1: PARLAY BUILDER
# =========================
with tab_builder:
    st.subheader("⚙️ Filters")
    c1, c2 = st.columns([2,1])
    with c1:
        seasons = st.multiselect("Seasons", ["2024-25","2023-24","2022-23"], default=["2024-25"])
    with c2:
        min_minutes = st.slider("Minimum Minutes", 0, 40, 20, 1)
    st.markdown("---")

    if "legs" not in st.session_state: st.session_state.legs=[]
    if "awaiting_input" not in st.session_state: st.session_state.awaiting_input=True

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("➕ Add Leg"): st.session_state.awaiting_input=True
    with c2:
        if st.button("➖ Remove Last Leg") and st.session_state.legs: st.session_state.legs.pop()

    st.write("**Input bet**")

    # render existing legs
    if st.session_state.legs:
        for i, leg in enumerate(st.session_state.legs):
            leg_no=i+1
            dir_short="O" if leg["dir"]=="Over" else "U"
            header=f"Leg {leg_no}: {leg['player']} — {dir_short} {fmt_half(leg['thr'])} {STAT_LABELS.get(leg['stat'], leg['stat'])} ({leg['loc']}, {leg['odds']})"
            with st.expander(header, expanded=False):
                cL,cR=st.columns([2,1])
                with cL:
                    leg["player"]=st.text_input("Player", value=leg["player"], key=f"player_{i}")
                    leg["stat"]=st.selectbox("Stat", list(STAT_LABELS.keys()), index=list(STAT_LABELS.keys()).index(leg["stat"]), key=f"stat_{i}")
                    leg["dir"]=st.selectbox("O/U", ["Over","Under"], index=(0 if leg["dir"]=="Over" else 1), key=f"dir_{i}")
                    leg["thr"]=st.number_input("Threshold", value=float(leg["thr"]), step=0.5, key=f"thr_{i}")
                with cR:
                    leg["loc"]=st.selectbox("Home/Away", ["All","Home Only","Away"], index=["All","Home Only","Away"].index(leg["loc"]), key=f"loc_{i}")
                    leg["range"]=st.selectbox("Game Range", ["FULL","L10","L20"], index=["FULL","L10","L20"].index(leg.get("range","FULL")), key=f"range_{i}")
                    leg["odds"]=st.number_input("Sportsbook Odds", value=int(leg["odds"]), step=5, key=f"odds_{i}")
                if st.button(f"❌ Remove Leg {leg_no}", key=f"remove_{i}"):
                    st.session_state.legs.pop(i); st.rerun()

    if st.session_state.awaiting_input:
        bet_text=st.text_input("Input bet", placeholder="Maxey O 24.5 P Away -110  OR  Embiid PRA U 35.5 -130", key="freeform_input", label_visibility="collapsed")
        if bet_text.strip():
            from copy import deepcopy
            parsed=parse_input_line(bet_text)
            if parsed and parsed["player"]:
                st.session_state.legs.append(deepcopy(parsed))
                st.session_state.awaiting_input=False; st.rerun()

    parlay_odds=0
    if len(st.session_state.legs)>1:
        st.markdown("### 🎯 Combined Parlay Odds")
        parlay_odds=st.number_input("Enter Parlay Odds (+300, -150, etc.)", value=0, step=5, key="parlay_odds")

    if st.session_state.legs and st.button("Compute"):
        # compute same as before
        st.markdown("---")
        rows=[]; probs_for_parlay=[]
        plt.rcParams.update({"axes.facecolor":"#1e1f22","figure.facecolor":"#1e1f22","text.color":"#ffffff","axes.labelcolor":"#e5e7eb","xtick.color":"#e5e7eb","ytick.color":"#e5e7eb","grid.color":"#374151"})
        for leg in st.session_state.legs:
            name=leg["player"]; pid=get_player_id(name)
            if not pid: rows.append({"ok":False,"name":name}); continue
            df=fetch_gamelog(pid, seasons)
            if df.empty: rows.append({"ok":False,"name":name,"reason":"No logs"}); continue
            d=df[df["MIN_NUM"]>=min_minutes]
            if leg["loc"]=="Home Only": d=d[d["MATCHUP"].str.contains("vs")]
            elif leg["loc"]=="Away": d=d[d["MATCHUP"].str.contains("@")]
            d=d.sort_values("GAME_DATE_DT", ascending=False)
            if leg["range"]=="L10": d=d.head(10)
            elif leg["range"]=="L20": d=d.head(20)
            p,h,t=leg_probability(d, leg["stat"], leg["dir"], float(leg["thr"]))
            fair=prob_to_american(p); book_prob=american_to_implied(leg["odds"])
            ev=None if book_prob is None else (p-book_prob)*100
            if p>0: probs_for_parlay.append(p)
            rows.append({"ok":True,"name":name,"p":p,"fair":fair,"book_prob":book_prob,"ev":ev,"stat":leg["stat"],"df":d,"thr":leg["thr"],"dir":leg["dir"],"loc":leg["loc"],"range":leg["range"],"odds":leg["odds"],"hits":h,"total":t})

        # combined & display code (unchanged for brevity)
        combined_p=float(np.prod(probs_for_parlay)) if probs_for_parlay else 0
        combined_fair=prob_to_american(combined_p) if combined_p>0 else "N/A"
        entered_prob=american_to_implied(parlay_odds)
        parlay_ev=None if entered_prob is None else (combined_p-entered_prob)*100
        cls="neutral" if parlay_ev is None else ("pos" if parlay_ev>=0 else "neg")
        st.markdown(f"""
<div class="card {cls}">
  <h2>💥 Combined Parlay</h2>
  <div class="row">
    <div class="m"><div class="lab">Model Prob</div><div class="val">{combined_p*100:.2f}%</div></div>
    <div class="m"><div class="lab">Fair Odds</div><div class="val">{combined_fair}</div></div>
    <div class="m"><div class="lab">Entered Odds</div><div class="val">{parlay_odds}</div></div>
    <div class="m"><div class="lab">Expected Value</div><div class="val">{'—' if parlay_ev is None else f'{parlay_ev:.2f}%'}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# TAB 2: BREAKEVEN
# =========================
with tab_breakeven:
    st.subheader("🔎 Breakeven Finder")
    cA, cB, cC, cD, cE = st.columns([2,1,1,1,1])
    with cA: player_query = st.text_input("Player", placeholder="e.g., Stephen Curry")
    with cB: last_n = st.slider("Last N Games", 5, 100, 20, 1)
    with cC: min_min_b = st.slider("Min Minutes", 0, 40, 20, 1)
    with cD: loc_choice = st.selectbox("Location", ["All","Home Only","Away"], index=0)
    with cE: seasons_b = st.multiselect("Seasons", ["2024-25","2023-24","2022-23"], default=["2024-25"])
    do_search = st.button("Search")

    if do_search:
        player_name = best_player_match(player_query)
        if not player_name: st.warning("Could not match player.")
        else:
            pid=get_player_id(player_name)
            if not pid: st.warning("No player ID found.")
            else:
                df=fetch_gamelog(pid, seasons_b)
                if df.empty: st.warning("No game logs found.")
                else:
                    d=df[df["MIN_NUM"]>=min_min_b]
                    if loc_choice=="Home Only": d=d[d["MATCHUP"].str.contains("vs")]
                    elif loc_choice=="Away": d=d[d["MATCHUP"].str.contains("@")]
                    d=d.sort_values("GAME_DATE_DT", ascending=False).head(last_n)
                    if d.empty: st.warning("No games match filters.")
                    else:
                        left,right=st.columns([1,2])
                        with left:
                            st.markdown(f"### **{player_name}**")
                            img=headshot_url(pid)
                            if img: st.image(img, width=180)
                            st.caption(f"Filters: Last {last_n} • Min {min_min_b}m • {loc_choice}")
                        with right:
                            stat_list=["PTS","REB","AST","FG3M","STL","BLK","P+R","P+A","R+A","PRA"]
                            rows=[]
                            for sc in stat_list:
                                ser=compute_stat_series(d, sc)
                                out=breakeven_for_stat(ser)
                                if out["line"] is None:
                                    rows.append({"Stat":STAT_LABELS[sc],"Breakeven Line":"—","Over Implied (Fair)":"—","Under Implied (Fair)":"—"})
                                    continue
                                over_p,under_p=out["over_prob"],out["under_prob"]
                                rows.append({
                                    "Stat":STAT_LABELS[sc],
                                    "Breakeven Line":fmt_half(out["line"]),
                                    "Over Implied (Fair)":f"{over_p*100:.1f}% ({out['over_odds']})",
                                    "Under Implied (Fair)":f"{under_p*100:.1f}% ({out['under_odds']})"
                                })
                            st.table(pd.DataFrame(rows).set_index("Stat"))
