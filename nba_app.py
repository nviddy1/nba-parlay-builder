import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="NBA Player Prop Parlay Builder", page_icon="🏀", layout="wide")
st.title("🏀 NBA Player Prop Parlay Builder")

# ----------------------------
# STYLES (Dark Theme Only)
# ----------------------------
st.markdown("""
<style>
:root {
  --bg: #0e0f11;
  --text: #f9fafb;
  --muted: #9ca3af;
  --card: #1a1b1e;
  --border: #32353b;
  --accent: #00ffaa;
}
body, .block-container {
  background-color: var(--bg);
  color: var(--text);
}
input, select, textarea {
  background-color: #202225 !important;
  border: 1px solid #3b3f45 !important;
  color: #f9fafb !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}
.stExpander {
  background-color: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  margin-bottom: 10px;
}
.stButton > button {
  background-color: #0d9488 !important;
  color: white !important;
  border-radius: 8px !important;
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HELPERS
# ----------------------------
def get_player_id(name: str):
    res = players.find_players_by_full_name(name)
    return res[0]["id"] if res else None

def fetch_gamelog(player_id: int, season: str) -> pd.DataFrame:
    try:
        g = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        g["MIN_NUM"] = g["MIN"].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) and ":" in x else int(float(x)))
        return g
    except Exception:
        return pd.DataFrame()

def calc_prob(df, stat, thr, min_minutes, loc_filter, range_key):
    if df.empty: return 0.0, 0, 0
    d = df[df["MIN_NUM"] >= min_minutes]
    if loc_filter == "Home Only":
        d = d[d["MATCHUP"].astype(str).str.contains("vs")]
    elif loc_filter == "Away Only":
        d = d[d["MATCHUP"].astype(str).str.contains("@")]
    if range_key == "L10":
        d = d.head(10)
    elif range_key == "L20":
        d = d.head(20)
    total = len(d)
    if total == 0: return 0.0, 0, 0
    hits = (d[stat] >= thr).sum()
    return hits / total, hits, total

def american_to_implied(odds):
    if odds > 0: return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def prob_to_american(prob):
    if prob <= 0 or prob >= 1: return "N/A"
    dec = 1 / prob
    return int((dec - 1) * 100) if dec >= 2 else int(-100 / (dec - 1))

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
with st.sidebar:
    st.subheader("⚙️ Filters")
    season = st.selectbox("Season", ["2024-25", "2023-24"], index=0)
    min_minutes = st.slider("Minimum Minutes", 0, 40, 20, 1)

# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "legs" not in st.session_state:
    st.session_state.legs = [{"player": "", "stat": "PTS", "thr": 10, "odds": -110, "loc": "All", "range": "FULL"}]

# ----------------------------
# ADD/REMOVE LEG BUTTONS
# ----------------------------
col_add, col_remove = st.columns([1, 1])
with col_add:
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({"player": "", "stat": "PTS", "thr": 10, "odds": -110, "loc": "All", "range": "FULL"})
with col_remove:
    if st.button("➖ Remove Leg") and len(st.session_state.legs) > 1:
        st.session_state.legs.pop()

# ----------------------------
# LEG EXPANDERS
# ----------------------------
stat_opts = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST", "Steals": "STL", "Blocks": "BLK", "3PM": "FG3M"}
range_opts = ["FULL", "L10", "L20"]
loc_opts = ["All", "Home Only", "Away Only"]

cols_per_row = 2
rows = [st.session_state.legs[i:i+cols_per_row] for i in range(0, len(st.session_state.legs), cols_per_row)]

for row_legs in rows:
    cols = st.columns(cols_per_row)
    for i, (leg, col) in enumerate(zip(row_legs, cols)):
        with col:
            with st.expander(f"Leg {st.session_state.legs.index(leg) + 1}", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    leg["player"] = st.text_input(f"Player {i}", value=leg["player"], key=f"player_{i}")
                with c2:
                    leg["stat"] = st.selectbox(f"Stat {i}", list(stat_opts.keys()), index=0, key=f"stat_{i}")
                with c3:
                    leg["thr"] = st.number_input(f"Threshold (≥) {i}", min_value=0, max_value=100, value=leg["thr"], key=f"thr_{i}")

                c4, c5, c6 = st.columns(3)
                with c4:
                    leg["loc"] = st.selectbox(f"Home/Away {i}", loc_opts, index=0, key=f"loc_{i}")
                with c5:
                    leg["range"] = st.selectbox(f"Game Range {i}", range_opts, index=0, key=f"range_{i}")
                with c6:
                    leg["odds"] = st.number_input(f"FanDuel Odds {i}", value=leg["odds"], step=5, key=f"odds_{i}")

# ----------------------------
# COMPUTE
# ----------------------------
if st.button("Compute"):
    probs = []
    st.markdown("### 📊 Results")
    for leg in st.session_state.legs:
        pid = get_player_id(leg["player"])
        if not pid:
            st.warning(f"Player not found: {leg['player']}")
            continue

        df = fetch_gamelog(pid, season)
        p, hits, total = calc_prob(df, stat_opts[leg["stat"]], leg["thr"], min_minutes, leg["loc"], leg["range"])
        implied = american_to_implied(leg["odds"])
        fair_odds = prob_to_american(p)
        ev = (p * (100/abs(leg["odds"])) - (1-p)) if leg["odds"] < 0 else (p * (leg["odds"]/100) - (1-p))

        color = "🟢" if ev > 0 else "🔴"
        st.markdown(f"**{color} {leg['player']} {leg['thr']}+ {leg['stat']}**")
        st.write(f"- Model Hit Rate: `{p*100:.1f}%` ({hits}/{total})")
        st.write(f"- Model Fair Odds: `{fair_odds}`")
        st.write(f"- Book Implied: `{implied*100:.1f}%`")
        st.write(f"- Expected Value: `{ev*100:.2f}%`")
        probs.append(p)

    if len(probs) > 1:
        parlay_prob = np.prod(probs)
        st.success(f"🔥 Combined Parlay Probability: **{parlay_prob*100:.2f}%**")

