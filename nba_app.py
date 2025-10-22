import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

st.set_page_config(page_title="NBA Parlay Builder", layout="wide")
st.title("🏀 NBA Parlay Builder (add as many legs as you like)")

# -----------------------------
# Helpers
# -----------------------------
def get_player_id(name):
    result = players.find_players_by_full_name(name)
    return result[0]["id"] if result else None

def get_player_gamelog(player_id, seasons):
    dfs = []
    for s in seasons:
        try:
            df = playergamelog.PlayerGameLog(player_id=player_id, season=s).get_data_frames()[0]
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs)
    df = df.astype({"PTS": float, "REB": float, "AST": float, "STL": float, "BLK": float, "FG3M": float, "MIN": float}, errors="ignore")
    return df

def calculate_probability(df, stat, threshold, home_only=None, min_minutes=20):
    if df.empty:
        return 0, 0, 0
    df = df[df["MIN"] >= min_minutes]
    if home_only is True:
        df = df[df["MATCHUP"].str.contains("vs.")]
    elif home_only is False:
        df = df[df["MATCHUP"].str.contains("@")]
    total = len(df)
    if total == 0:
        return 0, 0, 0
    hits = (df[stat] >= threshold).sum()
    return hits / total, hits, total

def prob_to_american(prob):
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob > 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int((1 - prob) / prob * 100)

def american_to_implied(odds):
    if odds == 0 or odds is None:
        return None
    try:
        odds = float(odds)
    except:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
season_options = ["2024-25", "2023-24", "2022-23"]
selected_seasons = st.sidebar.multiselect("Seasons", season_options, default=["2024-25"])
min_minutes = st.sidebar.slider("Min Minutes", 0, 40, 20)
home_filter = st.sidebar.selectbox("Game Location", ["All", "Home", "Away"])

# -----------------------------
# Manage Legs
# -----------------------------
if "legs" not in st.session_state:
    st.session_state.legs = [{"player": "", "stat": "PTS", "threshold": 10, "odds": -110}]

col1, col2 = st.columns(2)
with col1:
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({"player": "", "stat": "PTS", "threshold": 10, "odds": -110})
with col2:
    if st.button("➖ Remove Leg") and len(st.session_state.legs) > 1:
        st.session_state.legs.pop()

stat_options = {"PTS": "Points", "REB": "Rebounds", "AST": "Assists", "STL": "Steals", "BLK": "Blocks", "FG3M": "3PM"}

for i, leg in enumerate(st.session_state.legs):
    with st.expander(f"Leg {i+1}", expanded=True):
        leg["player"] = st.text_input(f"Player {i+1}", leg["player"], key=f"player_{i}")
        leg["stat"] = st.selectbox(f"Stat {i+1}", list(stat_options.keys()), format_func=lambda x: stat_options[x], key=f"stat_{i}")
        leg["threshold"] = st.number_input(f"Threshold (≥)", 0, 100, leg["threshold"], key=f"thr_{i}")
        leg["odds"] = st.number_input(f"FanDuel Odds", -1000, 1000, leg["odds"], key=f"odds_{i}")

# -----------------------------
# Compute Results
# -----------------------------
if st.button("Compute"):
    st.markdown("---")

    legs = st.session_state.legs
    all_probs = []
    results = []

    home_only = None
    if home_filter == "Home":
        home_only = True
    elif home_filter == "Away":
        home_only = False

    for leg in legs:
        pid = get_player_id(leg["player"])
        if not pid:
            continue
        df = get_player_gamelog(pid, selected_seasons)
        prob, hits, total = calculate_probability(df, leg["stat"], leg["threshold"], home_only, min_minutes)
        fair_odds = prob_to_american(prob)
        implied = american_to_implied(leg["odds"])
        ev = None if implied is None else (prob - implied) * 100
        results.append({
            "player": leg["player"],
            "stat": leg["stat"],
            "threshold": leg["threshold"],
            "prob": prob,
            "hits": hits,
            "total": total,
            "fair_odds": fair_odds,
            "odds": leg["odds"],
            "implied": implied,
            "ev": ev
        })
        if prob > 0:
            all_probs.append(prob)

    # -----------------------------
    # Combined Parlay Summary (top)
    # -----------------------------
    st.subheader("💥 Combined Parlay Summary")

    combined_prob = np.prod(all_probs) if all_probs else 0
    fair_odds = prob_to_american(combined_prob)
    parlay_odds = st.number_input("Enter Combined Parlay Odds (e.g. +300, -150)", value=0, step=5, key="parlay_odds")

    implied = american_to_implied(parlay_odds)
    parlay_ev = (combined_prob - implied) * 100 if implied else None

    # Style colors
    color = "#0b3d23" if (parlay_ev and parlay_ev >= 0) else "#3d0b0b" if parlay_ev else "#222"
    border = "#00FF99" if (parlay_ev and parlay_ev >= 0) else "#FF5555" if parlay_ev else "#888"
    emoji = "🔥" if (parlay_ev and parlay_ev >= 0) else "⚠️" if parlay_ev else "ℹ️"

    implied_str = f"{implied*100:.2f}%" if implied else "—"
    ev_str = f"{parlay_ev:.2f}%" if parlay_ev else "—"

    st.markdown(f"""
    <div style='background-color:{color};padding:25px;border-radius:15px;border:1px solid {border};margin-bottom:25px;'>
        <h2 style='color:white;'>Combined Parlay</h2>
        <p><b>Model Probability:</b> {combined_prob*100:.2f}%</p>
        <p><b>Model Fair Odds:</b> {fair_odds}</p>
        <p><b>Entered Odds:</b> {parlay_odds}</p>
        <p><b>Book Implied:</b> {implied_str}</p>
        <p><b>Expected Value:</b> {ev_str}</p>
        <p style='font-size:17px;color:white;'>{emoji} <b>{'+EV Parlay' if parlay_ev and parlay_ev >= 0 else ('Negative EV Parlay' if parlay_ev else 'Enter parlay odds to calculate')}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Individual Leg Results
    # -----------------------------
    for r in results:
        color = "#0b3d23" if (r["ev"] and r["ev"] >= 0) else "#3d0b0b"
        border = "#00FF99" if (r["ev"] and r["ev"] >= 0) else "#FF5555"
        emoji = "🔥" if (r["ev"] and r["ev"] >= 0) else "⚠️"

        st.markdown(f"""
        <div style='background-color:{color};padding:20px;border-radius:15px;border:1px solid {border};margin-bottom:15px;'>
            <h3 style='color:white;'>{r["player"]} — {r["threshold"]}+ {stat_options[r["stat"]]}</h3>
            <p><b>Model Hit Rate:</b> {r["prob"]*100:.1f}% ({r["hits"]}/{r["total"]})</p>
            <p><b>Model Fair Odds:</b> {r["fair_odds"]}</p>
            <p><b>FanDuel Odds:</b> {r["odds"]}</p>
            <p><b>Book Implied:</b> {r["implied"]*100:.2f}%</p>
            <p><b>Expected Value:</b> {r["ev"]:.2f}%</p>
            <p>{emoji} <b>{'+EV Play' if r["ev"] and r["ev"] >= 0 else 'Negative EV Play'}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Histogram
        df = get_player_gamelog(get_player_id(r["player"]), selected_seasons)
        if not df.empty and r["stat"] in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df[r["stat"]], bins=20, color="#00c896" if (r["ev"] and r["ev"] >= 0) else "#e05a5a")
            ax.axvline(r["threshold"], color="red", linestyle="--", label=f"Threshold {r['threshold']}")
            ax.set_title(f"{r['player']} — {stat_options[r['stat']]}")
            ax.legend()
            st.pyplot(fig)
