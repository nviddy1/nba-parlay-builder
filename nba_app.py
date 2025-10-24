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
# STYLE — DARK & RESPONSIVE
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

/* Expander styling */
.stExpander {
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  border-radius: 12px !important;
}
.streamlit-expanderHeader { font-weight: 800 !important; color: var(--text) !important; font-size: 0.9rem !important; }

/* Inputs */
input, select, textarea {
  background: #202225 !important;
  color: var(--text) !important;
  border: 1px solid #3b3f45 !important;
  border-radius: 8px !important;
  font-size: 0.85rem !important;
  padding: 6px !important;
}

/* Compact spacing on mobile */
.block-container { padding-top: 1rem !important; }
@media (max-width: 768px) {
  .stExpander { font-size: 0.85rem !important; }
  .stButton > button { font-size: 0.8rem !important; padding: 6px 10px !important; }
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

.card h2 { color:#fff; margin:0 0 6px 0; font-weight:800; font-size:1.1rem; }
.cond { color:#a9b1bb; font-size:0.85rem; margin: 2px 0 10px 0; }

.row {
  display:flex; flex-wrap:wrap; gap:10px;
  align-items:flex-end; justify-content:space-between;
  margin: 6px 0 4px 0;
}
.m { min-width:120px; flex:1; }
.lab { color:#cbd5e1; font-size:0.75rem; margin-bottom:2px; }
.val { color:#fff; font-size:1.1rem; font-weight:800; line-height:1.1; }

.chip {
  display:inline-block; margin-top:10px;
  padding:6px 12px; border-radius:999px;
  font-size:0.75rem; color:#a7f3d0; border:1px solid #16a34a33;
  background: transparent;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
@st.cache_data
def get_all_player_names():
    """Get all active NBA player names for autocomplete."""
    try:
        all_players = players.get_active_players()
    except Exception:
        all_players = players.get_players()  # fallback, includes retired
    return sorted({p.get("full_name") for p in all_players if p.get("full_name")})

PLAYER_LIST = [""] + get_all_player_names()  # leading blank for unselected

STAT_LABELS = {"PTS":"Points","REB":"Rebounds","AST":"Assists","STL":"Steals","BLK":"Blocks","FG3M":"3PM"}

def get_player_id(name: str):
    if not name:
        return None
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
    for k in STAT_LABELS.keys():
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
    return 100/(x+100) if x>0 else abs(x)/(abs(x)+100)

def prob_to_american(p: float):
    if p<=0 or p>=1: return "N/A"
    return f"{int(round((-100*p/(1-p)) if p>0.5 else (100*(1-p)/p))):+}"

def calc_prob(df, stat, thr, min_minutes, loc_filter, range_key, direction):
    if df.empty:
        return 0.0, 0, 0, df
    d = df.copy()
    d = d[d["MIN_NUM"] >= min_minutes]
    if loc_filter == "Home Only":
        d = d[d["MATCHUP"].astype(str).str.contains("vs", regex=False)]
    elif loc_filter == "Away Only":
        d = d[d["MATCHUP"].astype(str).str.contains("@", regex=False)]
    # ✅ FIX: correct sort call
    d = d.sort_values("GAME_DATE_DT", ascending=False)
    if range_key == "L10":
        d = d.head(10)
    elif range_key == "L20":
        d = d.head(20)
    total = len(d)
    if total == 0 or stat not in d.columns:
        return 0.0, 0, total, d
    hits = (d[stat] <= thr).sum() if direction.startswith("Under") else (d[stat] >= thr).sum()
    return hits / total, int(hits), int(total), d


# =========================
# SIDEBAR FILTERS
# =========================
with st.sidebar:
    st.subheader("⚙️ Filters")
    seasons = st.multiselect("Seasons", ["2024-25","2023-24","2022-23"], default=["2024-25"])
    min_minutes = st.slider("Minimum Minutes", 0, 40, 20, 1)

# =========================
# LEGS
# =========================
if "legs" not in st.session_state:
    st.session_state.legs = [{
        "player":"", "stat":"PTS", "dir":"Over (≥)", "thr":10,
        "odds":-110, "loc":"All", "range":"FULL"
    }]

# Add/Remove controls
c_add, c_remove = st.columns(2)
with c_add:
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({
            "player":"", "stat":"PTS", "dir":"Over (≥)", "thr":10,
            "odds":-110, "loc":"All", "range":"FULL"
        })
with c_remove:
    if st.button("➖ Remove Leg") and len(st.session_state.legs)>1:
        st.session_state.legs.pop()

# =========================
# RENDER LEGS
# =========================
def render_leg(i, leg, col):
    with col.expander(f"Leg {i+1}", expanded=True):
        left, right = st.columns(2)
        with left:
            # Dynamic player search (autocomplete)
            current_idx = PLAYER_LIST.index(leg["player"]) if leg.get("player") in PLAYER_LIST else 0
            leg["player"] = st.selectbox("Player", PLAYER_LIST, index=current_idx, key=f"p{i}")
            leg["loc"] = st.selectbox("Home/Away", ["All","Home Only","Away Only"],
                                      index=["All","Home Only","Away Only"].index(leg.get("loc","All")), key=f"l{i}")
            leg["range"] = st.selectbox("Game Range", ["FULL","L10","L20"],
                                        index=["FULL","L10","L20"].index(leg.get("range","FULL")), key=f"r{i}")
        with right:
            leg["stat"] = st.selectbox("Stat", list(STAT_LABELS.keys()),
                                       index=list(STAT_LABELS.keys()).index(leg.get("stat","PTS")),
                                       format_func=lambda k: STAT_LABELS[k], key=f"s{i}")
            c_ou, c_thr = st.columns([1,2])
            with c_ou:
                leg["dir"] = st.selectbox("O/U", ["Over (≥)", "Under (≤)"],
                                          index=["Over (≥)","Under (≤)"].index(leg.get("dir","Over (≥)")), key=f"d{i}")
            with c_thr:
                leg["thr"] = st.number_input("Threshold", min_value=0, max_value=100,
                                             value=int(leg.get("thr",10)), key=f"t{i}")
            leg["odds"] = st.number_input("FanDuel Odds", min_value=-10000, max_value=10000,
                                          value=int(leg.get("odds",-110)), step=5, key=f"o{i}")

for i in range(0, len(st.session_state.legs), 3):
    cols = st.columns(3)
    for j in range(3):
        k = i + j
        if k < len(st.session_state.legs):
            render_leg(k, st.session_state.legs[k], cols[j])

# =========================
# PARLAY ODDS (Centered under legs, above Compute)
# =========================
if len(st.session_state.legs) > 1:
    st.markdown("---")
    st.markdown("### 🎯 Combined Parlay Odds")
    parlay_odds = st.number_input("Enter Parlay Odds (+300, -150, etc.)", value=0, step=5, key="parlay_odds")
else:
    parlay_odds = 0

# =========================
# COMPUTE
# =========================
if st.button("Compute"):
    st.markdown("---")

    rows = []
    probs_for_parlay = []

    # Dark plot theme
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
        name = (leg["player"] or "").strip()
        stat = leg["stat"]; thr = int(leg["thr"]); book = int(leg["odds"])
        loc_key = leg.get("loc","All"); range_key = leg.get("range","FULL")
        direction = leg.get("dir","Over (≥)")

        pid = get_player_id(name) if name else None
        if not pid:
            rows.append({"ok":False,"name":name or "Unknown","stat":stat,"thr":thr,"book":book})
            continue

        df = fetch_gamelog(pid, seasons)
        p, hits, total, df_filt = calc_prob(df, stat, thr, min_minutes, loc_key, range_key, direction)
        fair = prob_to_american(p)
        book_prob = american_to_implied(book)
        ev = None if book_prob is None else (p - book_prob) * 100.0
        if p>0: probs_for_parlay.append(p)

        rows.append({
            "ok":True, "name":name, "stat":stat, "thr":thr, "book":book,
            "p":p, "hits":hits, "total":total, "fair":fair, "book_prob":book_prob,
            "ev":ev, "df":df_filt, "loc":loc_key, "range":range_key, "dir":direction
        })

    # ---------- Combined Parlay ----------
    combined_p = float(np.prod(probs_for_parlay)) if probs_for_parlay else 0.0
    combined_fair = prob_to_american(combined_p) if combined_p>0 else "N/A"
    entered_prob = american_to_implied(parlay_odds)
    parlay_ev = None if entered_prob is None else (combined_p - entered_prob) * 100.0
    cls = "neutral"
    if parlay_ev is not None: cls = "pos" if parlay_ev >= 0 else "neg"

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

    # ---------- Individual Legs ----------
    for r in rows:
        if not r.get("ok"):
            st.warning(f"Could not find player: **{r['name']}**")
            continue

        cls = "neutral"
        if r["ev"] is not None:
            cls = "pos" if r["ev"] >= 0 else "neg"

        stat_label = STAT_LABELS.get(r["stat"], r["stat"])
        book_implied = "—" if r["book_prob"] is None else f"{r['book_prob']*100:.1f}%"
        ev_disp = "—" if r["ev"] is None else f"{r['ev']:.2f}%"
        dir_word = "O" if r["dir"].startswith("Over") else "U"
        cond_text = f"{dir_word} {r['thr']} {stat_label.lower()} — {r['range']} — {r['loc'].replace(' Only','')}"

        st.markdown(f"""
<div class="card {cls}">
  <h2>{r['name']}</h2>
  <div class="cond">Condition: {cond_text}</div>
  <div class="row">
    <div class="m"><div class="lab">Model Hit Rate</div><div class="val">{r['p']*100:.1f}% ({r['hits']}/{r['total']})</div></div>
    <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{r['fair']}</div></div>
    <div class="m"><div class="lab">FanDuel Odds</div><div class="val">{r['book']}</div></div>
    <div class="m"><div class="lab">Book Implied</div><div class="val">{book_implied}</div></div>
    <div class="m"><div class="lab">Expected Value</div><div class="val">{ev_disp}</div></div>
  </div>
  <div class="chip">{'🔥 +EV Play Detected' if (r['ev'] is not None and r['ev']>=0) else ('⚠️ Negative EV Play' if r['ev'] is not None else 'ℹ️ Add odds to compute EV')}</div>
</div>
""", unsafe_allow_html=True)

        df_f = r["df"]
        if not df_f.empty and r["stat"] in df_f.columns:
            fig, ax = plt.subplots()
            ax.hist(df_f[r["stat"]], bins=20, edgecolor="white",
                    color=("#00c896" if (r["ev"] is not None and r["ev"]>=0) else "#e05a5a"))
            ax.axvline(r["thr"], color="w", linestyle="--", label=f"Threshold {r['thr']}")
            ax.set_title(f"{r['name']} — {stat_label}")
            ax.set_xlabel(stat_label); ax.set_ylabel("Games"); ax.legend()
            st.pyplot(fig)
