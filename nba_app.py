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

input, select, textarea {
  background: #202225 !important;
  color: var(--text) !important;
  border: 1px solid #3b3f45 !important;
  border-radius: 8px !important;
  font-size: 0.9rem !important;
  padding: 6px !important;
}

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

table { border-collapse: collapse; }
thead th { border-bottom: 1px solid #374151 !important; }
tbody td, thead th { padding: 8px 10px !important; }

.filters-row { 
  display: flex; 
  gap: 10px; 
  align-items: flex-end; 
  max-width: 33%; 
  min-width: 300px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
STAT_LABELS = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists", "STL": "Steals", "BLK": "Blocks", "FG3M": "3PM",
    "DOUBDOUB": "Doub Doub", "TRIPDOUB": "Trip Doub",
    "P+R": "P+R", "P+A": "P+A", "R+A": "R+A", "PRA": "PRA"
}
STAT_TOKENS = {
    "P":"PTS","PTS":"PTS","R":"REB","REB":"REB","A":"AST","AST":"AST","STL":"STL","BLK":"BLK",
    "3PM":"FG3M","FG3M":"FG3M","THREES":"FG3M","DOUBDOUB":"DOUBDOUB","TRIPDOUB":"TRIPDOUB",
    "P+R":"P+R","P+A":"P+A","R+A":"R+A","PRA":"PRA","PR":"P+R","PA":"P+A","RA":"R+A",
    "DD":"DOUBDOUB","TD":"TRIPDOUB"
}

@st.cache_data
def get_all_player_names():
    try:
        active = players.get_active_players()
        names = [p["full_name"] for p in active if p.get("full_name")]
        if names: return sorted(set(names))
    except Exception: pass
    all_p = players.get_players()
    return sorted(set([p["full_name"] for p in all_p if p.get("full_name")]))

PLAYER_LIST = get_all_player_names()

def best_player_match(query):
    if not query: return ""
    m = process.extractOne(query, PLAYER_LIST, score_cutoff=70)
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
    try: v=float(x); return f"{v:.1f}".rstrip("0").rstrip(".")
    except: return str(x)

def parse_input_line(text):
    t=(text or "").strip()
    if not t: return None
    parts=t.replace("/", "+").split()
    dir_token="Over" if any(p.upper() in ["O","OVER"] for p in parts) else ("Under" if any(p.upper() in ["U","UNDER"] for p in parts) else "Over")
    thr=next((float(p) for p in parts if any(c.isdigit() for c in p) and ("." in p or p.isdigit())),10.5)
    stat_code=next((STAT_TOKENS.get(p.upper(),p.upper()) for p in parts if p.upper() in STAT_TOKENS), "PTS")
    loc="Away" if any(p.upper() in ["AWAY","A"] for p in parts) else ("Home Only" if any(p.upper() in ["HOME","H"] for p in parts) else "All")
    odds=next((int(p) for p in parts[::-1] if p.startswith(("+","-")) and (abs(int(p))>=100)), -110)
    banned=set(["O","OVER","U","UNDER","HOME","H","AWAY","A"]+list(STAT_TOKENS.keys()))
    name_tokens=[p for p in parts if p.upper() not in banned and not p.replace(".","",1).lstrip("+-").isdigit()]
    player=best_player_match(" ".join(name_tokens))
    return {"player":player,"dir":dir_token,"thr":float(thr),"stat":stat_code,"loc":loc,"range":"FULL","odds":int(odds)}

def get_player_id(name):
    if not name: return None
    res=players.find_players_by_full_name(name)
    return res[0]["id"] if res else None

def to_minutes(v):
    try:
        s=str(v)
        return int(s.split(":")[0]) if ":" in s else int(float(s))
    except: return 0

def fetch_gamelog(pid, seasons):
    dfs=[]
    for s in seasons:
        try: dfs.append(playergamelog.PlayerGameLog(player_id=pid, season=s).get_data_frames()[0])
        except: pass
    if not dfs: return pd.DataFrame()
    df=pd.concat(dfs, ignore_index=True)
    for k in ["PTS","REB","AST","STL","BLK","FG3M"]:
        if k in df.columns: df[k]=pd.to_numeric(df[k], errors="coerce")
    df["MIN_NUM"]=df["MIN"].apply(to_minutes) if "MIN" in df.columns else 0
    df["GAME_DATE_DT"]=pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df

def compute_stat_series(df, sc):
    if sc in ["PTS","REB","AST","STL","BLK","FG3M"]: return df[sc].astype(float)
    if sc=="P+R": return (df["PTS"]+df["REB"]).astype(float)
    if sc=="P+A": return (df["PTS"]+df["AST"]).astype(float)
    if sc=="R+A": return (df["REB"]+df["AST"]).astype(float)
    if sc=="PRA": return (df["PTS"]+df["REB"]+df["AST"]).astype(float)
    if sc=="DOUBDOUB": return ((df["PTS"]>=10)+(df["REB"]>=10)+(df["AST"]>=10)>=2).astype(int)
    if sc=="TRIPDOUB": return ((df["PTS"]>=10)+(df["REB"]>=10)+(df["AST"]>=10)>=3).astype(int)
    return df["PTS"].astype(float)

def leg_probability(df, sc, direction, thr):
    ser=compute_stat_series(df, sc)
    if sc in ["DOUBDOUB","TRIPDOUB"]:
        hits=(ser>=1).sum() if direction=="Over" else (ser<=0).sum()
    else:
        hits=(ser>=thr).sum() if direction=="Over" else (ser<=thr).sum()
    total=ser.notna().sum()
    return (hits/total if total else 0.0, hits, total)

def headshot_url(pid): 
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png" if pid else None

def breakeven_for_stat(series):
    s=pd.to_numeric(series, errors="coerce").dropna()
    if len(s)==0: return {"line":None}
    lo, hi=np.floor(s.min())-0.5, np.ceil(s.max())+0.5
    best_t, best_gap, best_over=None, 1.0, None
    for t in np.arange(lo,hi+0.001,0.5):
        over=(s>=t).mean()
        gap=abs(over-0.5)
        if gap<best_gap: best_t, best_gap, best_over=t,gap,over
    over_prob=float(best_over); under_prob=1.0-over_prob
    return {"line":float(best_t),"over_prob":over_prob,"under_prob":under_prob,
            "over_odds":prob_to_american(over_prob),"under_odds":prob_to_american(under_prob)}

# =========================
# TABS
# =========================
tab_builder, tab_breakeven = st.tabs(["🧮 Parlay Builder", "🧷 Breakeven"])

# ---------- PARLAY BUILDER ----------
with tab_builder:
    st.markdown('<div class="filters-row">', unsafe_allow_html=True)
    seasons = st.multiselect("Season(s)", ["2024-25","2023-24","2022-23"], default=["2024-25"], label_visibility="collapsed")
    min_minutes = st.slider("Min Minutes", 0, 40, 20, 1, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if "legs" not in st.session_state: st.session_state.legs=[]
    if "awaiting_input" not in st.session_state: st.session_state.awaiting_input=True

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("➕ Add Leg"): st.session_state.awaiting_input=True
    with c2:
        if st.button("➖ Remove Last Leg") and st.session_state.legs: st.session_state.legs.pop()

    st.write("**Input bet**")

    if st.session_state.legs:
        for i, leg in enumerate(st.session_state.legs):
            leg_no=i+1
            header=f"Leg {leg_no}: {leg['player']} — {'O' if leg['dir']=='Over' else 'U'} {fmt_half(leg['thr'])} {STAT_LABELS.get(leg['stat'],leg['stat'])}"
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
                    st.session_state.legs.pop(i)
                    st.rerun()

    if st.session_state.awaiting_input:
        bet_text=st.text_input("Input bet", placeholder="Maxey O 24.5 P Away -110  OR  Embiid PRA U 35.5 -130", key="freeform_input", label_visibility="collapsed")
        if bet_text.strip():
            parsed=parse_input_line(bet_text)
            if parsed and parsed["player"]:
                st.session_state.legs.append(parsed)
                st.session_state.awaiting_input=False
                st.rerun()

    parlay_odds=0
    if len(st.session_state.legs)>1:
        st.markdown("### 🎯 Combined Parlay Odds")
        parlay_odds=st.number_input("Enter Parlay Odds (+300, -150, etc.)", value=0, step=5, key="parlay_odds")

    if st.session_state.legs and st.button("Compute"):
        st.markdown("---")
        rows=[]; probs_for_parlay=[]
        plt.rcParams.update({
            "axes.facecolor":"#1e1f22","figure.facecolor":"#1e1f22","text.color":"#ffffff",
            "axes.labelcolor":"#e5e7eb","xtick.color":"#e5e7eb","ytick.color":"#e5e7eb","grid.color":"#374151",
        })

        for leg in st.session_state.legs:
            name=leg["player"]; pid=get_player_id(name)
            if not pid: rows.append({"ok":False,"name":name}); continue
            df=fetch_gamelog(pid, seasons)
            if df.empty: rows.append({"ok":False,"name":name,"reason":"No logs"}); continue

            d=df[df["MIN_NUM"]>=min_minutes].copy()
            if leg["loc"]=="Home Only": d=d[d["MATCHUP"].str.contains("vs", regex=False)]
            elif leg["loc"]=="Away": d=d[d["MATCHUP"].str.contains("@", regex=False)]
            d=d.sort_values("GAME_DATE_DT", ascending=False)
            if leg["range"]=="L10": d=d.head(10)
            elif leg["range"]=="L20": d=d.head(20)

            p,hits,total=leg_probability(d, leg["stat"], leg["dir"], leg["thr"]) if not d.empty else (0,0,0)
            fair=prob_to_american(p)
            book_prob=american_to_implied(leg["odds"])
            ev=None if book_prob is None else (p-book_prob)*100
            if p>0: probs_for_parlay.append(p)

            rows.append({"ok":True,"name":name,"stat":leg["stat"],"thr":leg["thr"],"dir":leg["dir"],"loc":leg["loc"],
                         "range":leg["range"],"odds":leg["odds"],"p":p,"hits":hits,"total":total,"fair":fair,
                         "book_prob":book_prob,"ev":ev,"df":d})

        combined_p=float(np.prod(probs_for_parlay)) if probs_for_parlay else 0.0
        combined_fair=prob_to_american(combined_p) if combined_p>0 else "N/A"
        entered_prob=american_to_implied(parlay_odds)
        parlay_ev=None if entered_prob is None else (combined_p-entered_prob)*100
        cls="neutral" if parlay_ev is None else ("pos" if parlay_ev>=0 else "neg")

        st.markdown(f"""
<div class="card {cls}">
  <h2>💥 Combined Parlay</h2>
  <div class="cond">Includes all legs with your filters</div>
  <div class="row">
    <div class="m"><div class="lab">Model Parlay Probability</div><div class="val">{combined_p*100:.2f}%</div></div>
    <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{combined_fair}</div></div>
    <div class="m"><div class="lab">Entered Odds</div><div class="val">{parlay_odds if parlay_odds else '—'}</div></div>
    <div class="m"><div class="lab">Book Implied</div><div class="val">{'—' if entered_prob is None else f'{entered_prob*100:.2f}%'}</div></div>
    <div class="m"><div class="lab">Expected Value</div><div class="val">{'—' if parlay_ev is None else f'{parlay_ev:.2f}%'}</div></div>
  </div>
  <div class="chip">{'🔥 +EV Parlay Detected' if (parlay_ev is not None and parlay_ev>=0) else ('⚠️ Negative EV Parlay' if parlay_ev is not None else 'ℹ️ Enter parlay odds')}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        for r in rows:
            if not r.get("ok"):
                st.warning(f"Could not compute for **{r.get('name','Unknown')}**")
                continue

            cls="neutral" if r["ev"] is None else ("pos" if r["ev"]>=0 else "neg")
            stat_label=STAT_LABELS.get(r["stat"],r["stat"])
            book_implied="—" if r["book_prob"] is None else f"{r['book_prob']*100:.1f}%"
            ev_disp="—" if r["ev"] is None else f"{r['ev']:.2f}%"
            dir_word="O" if r["dir"]=="Over" else "U"
            cond_text=f"{dir_word} {fmt_half(r['thr'])} {stat_label} — {r['range']} — {r['loc'].replace(' Only','')}"

            st.markdown(f"""
<div class="card {cls}">
  <h2>{r['name']}</h2>
  <div class="cond">Condition: {cond_text}</div>
  <div class="row">
    <div class="m"><div class="lab">Model Hit Rate</div><div class="val">{r['p']*100:.1f}% ({r['hits']}/{r['total']})</div></div>
    <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{r['fair']}</div></div>
    <div class="m"><div class="lab">Sportsbook Odds</div><div class="val">{r['odds']}</div></div>
    <div class="m"><div class="lab">Book Implied</div><div class="val">{book_implied}</div></div>
    <div class="m"><div class="lab">Expected Value</div><div class="val">{ev_disp}</div></div>
  </div>
  <div class="chip">{'🔥 +EV Play Detected' if (r['ev'] is not None and r['ev']>=0) else ('⚠️ Negative EV Play' if r['ev'] is not None else 'ℹ️ Add odds to compute EV')}</div>
</div>
""", unsafe_allow_html=True)

            if not r["df"].empty and r["stat"] not in ["DOUBDOUB","TRIPDOUB"]:
                ser=compute_stat_series(r["df"], r["stat"])
                fig, ax = plt.subplots()
                ax.hist(ser, bins=20, edgecolor="white",
                        color=("#00c896" if (r["ev"] is not None and r["ev"]>=0) else "#e05a5a"))
                ax.axvline(r["thr"], color="w", linestyle="--", label=f"Threshold {fmt_half(r['thr'])}")
                ax.set_title(f"{r['name']} — {stat_label}")
                ax.set_xlabel(stat_label); ax.set_ylabel("Games"); ax.legend()
                st.pyplot(fig)

# ---------- BREAKEVEN ----------
with tab_breakeven:
    st.markdown('<div class="filters-row">', unsafe_allow_html=True)
    player_query = st.text_input("Player", placeholder="e.g., Stephen Curry", label_visibility="collapsed")
    last_n = st.slider("Last N Games", 5, 100, 20, 1, label_visibility="collapsed")
    min_min_b = st.slider("Min Minutes", 0, 40, 20, 1, label_visibility="collapsed")
    loc_choice = st.selectbox("Location", ["All","Home Only","Away"], index=0, label_visibility="collapsed")
    seasons_b = st.multiselect("Seasons", ["2024-25","2023-24","2022-23"], default=["2024-25"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Search"):
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
                    d=df[df["MIN_NUM"]>=min_min_b].copy()
                    if loc_choice=="Home Only": d=d[d["MATCHUP"].str.contains("vs", regex=False)]
                    elif loc_choice=="Away": d=d[d["MATCHUP"].str.contains("@", regex=False)]
                    d=d.sort_values("GAME_DATE_DT", ascending=False).head(last_n)
                    if d.empty:
                        st.warning("No games match your filters.")
                    else:
                        left,right = st.columns([1,2], vertical_alignment="top")
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
                                    rows.append({"Stat":STAT_LABELS.get(sc,sc),"Breakeven Line":"—",
                                                 "Over Implied (Fair)":"—","Under Implied (Fair)":"—"})
                                    continue
                                over_disp=f"{out['over_prob']*100:.1f}% ({out['over_odds']})"
                                under_disp=f"{out['under_prob']*100:.1f}% ({out['under_odds']})"
                                rows.append({"Stat":STAT_LABELS.get(sc,sc),"Breakeven Line":fmt_half(out["line"]),
                                             "Over Implied (Fair)":over_disp,"Under Implied (Fair)":under_disp})
                            breakeven_df=pd.DataFrame(rows, columns=["Stat","Breakeven Line","Over Implied (Fair)","Under Implied (Fair)"])
                            st.table(breakeven_df.set_index("Stat"))
