import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from difflib import get_close_matches
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NBA Player Prop Parlay Builder", page_icon="🏀", layout="wide")
st.title("🏀 NBA Player Prop Parlay Builder")

# =========================
# STYLING — DARK MODE
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
  box-shadow: 0 6px 18px rgba(0,200,150,0.25) !important;
}

/* Expanders */
.stExpander {
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  border-radius: 12px !important;
}
.streamlit-expanderHeader { font-weight: 800 !important; color: var(--text) !important; }

/* Cards */
.card {
  padding: 20px; border-radius: 14px;
  margin: 10px 0 20px 0;
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  box-shadow: 0 0 14px rgba(0,0,0,0.25);
  width: 100%;
}
.neutral { --card-bg:#222; --card-border:#777; }
.pos { --card-bg:#0b3d23; --card-border:#00FF99; }
.neg { --card-bg:#3d0b0b; --card-border:#FF5555; }

.card h2 { color:#fff; font-weight:800; margin-bottom:4px; }
.cond { color:#a9b1bb; font-size:0.9rem; margin-bottom:10px; }
.row { display:flex; flex-wrap:wrap; gap:12px; justify-content:space-between; }
.m { min-width:140px; flex:1; }
.lab { color:#cbd5e1; font-size:0.8rem; margin-bottom:4px; }
.val { color:#fff; font-size:1.2rem; font-weight:800; }
.chip {
  display:inline-block; margin-top:10px;
  padding:6px 12px; border-radius:999px;
  font-size:0.8rem; color:#a7f3d0; border:1px solid #16a34a33;
}

/* Make select dropdown text visible everywhere */
[data-baseweb="select"] span {
  color: #fff !important;
  font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
@st.cache_data
def get_all_player_names():
    try:
        all_players = players.get_active_players()
    except Exception:
        all_players = players.get_players()
    return sorted({p.get("full_name") for p in all_players if p.get("full_name")})

PLAYER_LIST = [""] + get_all_player_names()

STAT_LABELS = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists", "STL": "Steals", "BLK": "Blocks", "FG3M": "3PM",
    "Doub Doub": "Double Double", "Trip Doub": "Triple Double",
    "P+R": "Points + Rebounds", "P+A": "Points + Assists", "R+A": "Rebounds + Assists", "PRA": "Points + Rebounds + Assists"
}

def get_player_id(name: str):
    if not name: return None
    res = players.find_players_by_full_name(name)
    return res[0]["id"] if res else None

def to_minutes(val):
    try:
        s = str(val)
        if ":" in s: return int(s.split(":")[0])
        return int(float(s))
    except: return 0

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
        if k in df.columns: df[k] = pd.to_numeric(df[k], errors="coerce")
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
    if -100 < x < 100: return None
    return 100/(x+100) if x>0 else abs(x)/(abs(x)+100)

def prob_to_american(p: float):
    if p<=0 or p>=1: return "N/A"
    return f"{int(round((-100*p/(1-p)) if p>0.5 else (100*(1-p)/p))):+}"

def normalize_american_odds(x: int) -> int:
    try:
        x = int(x)
    except:
        return 0
    if x == 0:
        return 0
    if -100 < x < 100:
        return -100 if x < 0 else 100
    return x

def calc_prob(df, stat, thr, min_minutes, loc_filter, range_key, direction):
    if df.empty: return 0.0,0,0,df
    d = df.copy()
    d = d[d["MIN_NUM"] >= min_minutes]
    if loc_filter == "Home Only": d = d[d["MATCHUP"].astype(str).str.contains("vs", regex=False)]
    elif loc_filter == "Away Only": d = d[d["MATCHUP"].astype(str).str.contains("@", regex=False)]

    d = d.sort_values("GAME_DATE_DT", ascending=False)
    if range_key == "L10": d = d.head(10)
    elif range_key == "L20": d = d.head(20)

    # Derived stats
    if "PTS" in d.columns and "REB" in d.columns and "AST" in d.columns:
        d["P+R"] = d["PTS"] + d["REB"]
        d["P+A"] = d["PTS"] + d["AST"]
        d["R+A"] = d["REB"] + d["AST"]
        d["PRA"] = d["PTS"] + d["REB"] + d["AST"]
        d["Doub Doub"] = (
            ((d["PTS"] >= 10) & (d["REB"] >= 10)) |
            ((d["PTS"] >= 10) & (d["AST"] >= 10)) |
            ((d["REB"] >= 10) & (d["AST"] >= 10))
        ).astype(int)
        d["Trip Doub"] = ((d["PTS"] >= 10) & (d["REB"] >= 10) & (d["AST"] >= 10)).astype(int)

    total = len(d)
    if total==0 or stat not in d.columns: return 0.0,0,total,d
    hits = (d[stat] <= thr).sum() if str(direction).startswith("U") else (d[stat] >= thr).sum()
    return hits/total, int(hits), int(total), d

# ---- Natural language parser for a single-line leg input
def parse_bet_query(text, player_list):
    text = text.strip()
    if not text: return {}
    # Odds
    odds_match = re.search(r'([+-]\d{2,4})', text)
    odds = int(odds_match.group(1)) if odds_match else -110
    odds = normalize_american_odds(odds)
    # Direction
    direction = "O" if re.search(r'\b(o|over)\b', text, re.I) else ("U" if re.search(r'\b(u|under)\b', text, re.I) else "O")
    # Threshold
    threshold_match = re.search(r'(\d+(?:\.\d)?)', text)
    threshold = float(threshold_match.group(1)) if threshold_match else 10.5
    # Location
    loc = "Away Only" if re.search(r'\b(away|road|@)\b', text, re.I) else ("Home Only" if re.search(r'\b(home|vs)\b', text, re.I) else "All")
    # Stat
    stat_aliases = {"P":"PTS","R":"REB","A":"AST","S":"STL","B":"BLK","3PM":"FG3M",
                    "PR":"P+R","PA":"P+A","RA":"R+A","PRA":"PRA","DD":"Doub Doub","TD":"Trip Doub"}
    stat_code = "PTS"
    for alias, code in stat_aliases.items():
        if re.search(rf'\b{alias}\b', text, re.I): stat_code = code; break
    # Player (fuzzy)
    tokens = re.sub(r'[^a-zA-Z\s]', '', text)
    matches = get_close_matches(tokens.title(), player_list, n=1, cutoff=0.4)
    player = matches[0] if matches else ""
    return {"player":player,"stat":stat_code,"dir":direction,"thr":threshold,"odds":odds,"loc":loc,"range":"FULL"}

# =========================
# SIDEBAR FILTERS
# =========================
with st.sidebar:
    st.subheader("⚙️ Filters")
    seasons = st.multiselect("Seasons", ["2024-25","2023-24","2022-23"], default=["2024-25"])
    min_minutes = st.slider("Minimum Minutes", 0, 40, 20, 1)

# =========================
# STATE
# =========================
if "legs" not in st.session_state:
    st.session_state.legs = [{"player":"","stat":"PTS","dir":"O","thr":10.5,"odds":-110,"loc":"All","range":"FULL"}]

# =========================
# ADD / REMOVE
# =========================
c1, c2 = st.columns(2)
with c1:
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({"player":"","stat":"PTS","dir":"O","thr":10.5,"odds":-110,"loc":"All","range":"FULL"})
with c2:
    if st.button("➖ Remove Leg") and len(st.session_state.legs)>1:
        st.session_state.legs.pop()

# =========================
# MANUAL INPUTS (shown only when user checks "Edit manually")
def render_manual_inputs(leg, i):
    left, right = st.columns(2)
    with left:
        leg["player"] = st.selectbox("Player", PLAYER_LIST, index=PLAYER_LIST.index(leg["player"]) if leg["player"] in PLAYER_LIST else 0, key=f"p{i}")
        leg["loc"] = st.selectbox("Home/Away", ["All","Home Only","Away Only"], index=["All","Home Only","Away Only"].index(leg["loc"]), key=f"l{i}")
        leg["range"] = st.selectbox("Game Range", ["FULL","L10","L20"], index=["FULL","L10","L20"].index(leg["range"]), key=f"r{i}")
    with right:
        leg["stat"] = st.selectbox("Stat", list(STAT_LABELS.keys()), format_func=lambda k: STAT_LABELS[k], index=list(STAT_LABELS.keys()).index(leg["stat"]), key=f"s{i}")
        leg["dir"] = st.selectbox("O/U", ["O","U"], index=["O","U"].index(leg["dir"]), key=f"d{i}")
        leg["thr"] = st.number_input("Threshold", min_value=0.0, max_value=100.0, value=float(leg["thr"]), step=0.5, format="%.1f", key=f"t{i}")
        entered_odds = st.number_input("Sportsbook Odds", min_value=-10000, max_value=10000, value=int(leg["odds"]), step=5, key=f"o{i}")
        leg["odds"] = normalize_american_odds(entered_odds)

# =========================
# LEGS INPUT (one-line parser + optional manual edit)
# =========================
for i, leg in enumerate(st.session_state.legs):
    query = st.text_input("Input bet", placeholder="e.g. Horford O 10.5 R Away -110", key=f"q{i}")
    if query:
        leg.update(parse_bet_query(query, PLAYER_LIST))
    # Summary header
    thr_disp = str(leg['thr']).rstrip('0').rstrip('.')
    stat_disp = STAT_LABELS.get(leg['stat'], leg['stat'])
    loc_disp = leg['loc'].replace(' Only','')
    header = f"{leg['player']} — {leg['dir']} {thr_disp} {stat_disp} ({loc_disp}, {leg['odds']})" if leg['player'] else f"Leg {i+1}"
    with st.expander(header, expanded=True):
        edit = st.checkbox("Edit manually", key=f"edit{i}")
        if edit:
            render_manual_inputs(leg, i)

# =========================
# PARLAY ODDS INPUT
# =========================
parlay_odds = st.number_input("Combined Parlay Odds (+300, -150, etc.)", value=0, step=5, key="parlay_odds")

# =========================
# COMPUTE SECTION
# =========================
if st.button("Compute"):
    st.markdown("---")
    rows, probs_for_parlay = [], []
    plt.rcParams.update({
        "axes.facecolor": "#1e1f22","figure.facecolor": "#1e1f22","text.color": "#ffffff",
        "axes.labelcolor": "#e5e7eb","xtick.color": "#e5e7eb","ytick.color": "#e5e7eb","grid.color": "#374151",
    })

    for leg in st.session_state.legs:
        name = (leg["player"] or "").strip()
        stat, thr, book = leg["stat"], float(leg["thr"]), int(leg["odds"])
        loc_key, range_key, direction = leg["loc"], leg["range"], leg["dir"]
        pid = get_player_id(name)
        if not pid:
            rows.append({"ok":False,"name":name or "Unknown","stat":stat,"thr":thr,"book":book})
            continue

        df = fetch_gamelog(pid, seasons)

        # Build derived columns needed for combo/milestone stats
        if not df.empty and {"PTS","REB","AST"}.issubset(df.columns):
            df["P+R"] = df["PTS"] + df["REB"]
            df["P+A"] = df["PTS"] + df["AST"]
            df["R+A"] = df["REB"] + df["AST"]
            df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
            df["Doub Doub"] = (
                ((df["PTS"] >= 10) & (df["REB"] >= 10)) |
                ((df["PTS"] >= 10) & (df["AST"] >= 10)) |
                ((df["REB"] >= 10) & (df["AST"] >= 10))
            ).astype(int)
            df["Trip Doub"] = ((df["PTS"] >= 10) & (df["REB"] >= 10) & (df["AST"] >= 10)).astype(int)

        p, hits, total, df_filt = calc_prob(df, stat, thr, min_minutes, loc_key, range_key, direction)
        fair = prob_to_american(p)
        book_prob = american_to_implied(book)
        ev = None if book_prob is None else (p - book_prob) * 100.0
        if p>0: probs_for_parlay.append(p)

        rows.append({
            "ok":True,"name":name,"stat":stat,"thr":thr,"book":book,
            "p":p,"hits":hits,"total":total,"fair":fair,"book_prob":book_prob,
            "ev":ev,"df":df_filt,"loc":loc_key,"range":range_key,"dir":direction
        })

    combined_p = float(np.prod(probs_for_parlay)) if probs_for_parlay else 0.0
    combined_fair = prob_to_american(combined_p) if combined_p>0 else "N/A"
    entered_prob = american_to_implied(parlay_odds)
    parlay_ev = None if entered_prob is None else (combined_p - entered_prob)*100.0
    cls = "neutral" if parlay_ev is None else ("pos" if parlay_ev>=0 else "neg")

    st.markdown(f"""
<div class="card {cls}">
  <h2>💥 Combined Parlay</h2>
  <div class="cond">Includes all selected legs</div>
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
            st.warning(f"Could not find player: **{r['name']}**")
            continue

        cls = "neutral" if r["ev"] is None else ("pos" if r["ev"]>=0 else "neg")
        stat_label = STAT_LABELS.get(r["stat"], r["stat"])
        book_implied = "—" if r["book_prob"] is None else f"{r['book_prob']*100:.1f}%"
        ev_disp = "—" if r["ev"] is None else f"{r['ev']:.2f}%"
        cond_text = f"{r['dir']} {str(r['thr']).rstrip('0').rstrip('.')} {stat_label.lower()} — {r['range']} — {r['loc'].replace(' Only','')}"

        st.markdown(f"""
<div class="card {cls}">
  <h2>{r['name']}</h2>
  <div class="cond">Condition: {cond_text}</div>
  <div class="row">
    <div class="m"><div class="lab">Model Hit Rate</div><div class="val">{r['p']*100:.1f}% ({r['hits']}/{r['total']})</div></div>
    <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{r['fair']}</div></div>
    <div class="m"><div class="lab">Sportsbook Odds</div><div class="val">{r['book']}</div></div>
    <div class="m"><div class="lab">Book Implied</div><div class="val">{book_implied}</div></div>
    <div class="m"><div class="lab">Expected Value</div><div class="val">{ev_disp}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Histogram
        df_f = r["df"]
        if not df_f.empty and r["stat"] in df_f.columns:
            fig, ax = plt.subplots()
            ax.hist(df_f[r["stat"]], bins=20, edgecolor="white",
                    color=("#00c896" if (r["ev"] is not None and r["ev"]>=0) else "#e05a5a"))
            ax.axvline(r["thr"], color="w", linestyle="--", label=f"Threshold {r['thr']}")
            ax.set_title(f"{r['name']} — {stat_label}")
            ax.set_xlabel(stat_label); ax.set_ylabel("Games"); ax.legend()
            st.pyplot(fig)
