import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

st.set_page_config(page_title="NBA Parlay Builder", layout="wide")

st.title("🏀 NBA Parlay Builder (add as many legs as you like)")

# -----------------------------
# Helper functions
# -----------------------------
def get_player_id(name):
    result = players.find_players_by_full_name(name)
    return result[0]['id'] if result else None

def get_player_gamelog(player_id, seasons):
    dfs = []
    for season in seasons:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        dfs.append(gamelog)
    df = pd.concat(dfs, ignore_index=True)
    df = df.astype({
        "PTS": float,
        "REB": float,
        "AST": float,
        "STL": float,
        "BLK": float,
        "FG3M": float,
        "MIN": float
    }, errors="ignore")
    return df

def calculate_probability(df, stat, threshold, home_only=None, min_minutes=20):
    df = df[df["MIN"] >= min_minutes]
    if home_only is not None:
        if home_only:
            df = df[df["MATCHUP"].str.contains("vs.")]
        else:
            df = df[df["MATCHUP"].str.contains("@")]
    if len(df) == 0:
        return 0.0, 0, 0
    hits = (df[stat] >= threshold).sum()
    prob = hits / len(df)
    return prob, hits, len(df)

def prob_to_american(prob):
    if prob == 0:
        return "N/A"
    if prob > 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int((1 - prob) / prob * 100)

def american_to_implied(prob_odds):
    if prob_odds > 0:
        return 100 / (prob_odds + 100)
    else:
        return abs(prob_odds) / (abs(prob_odds) + 100)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

season_options = ["2024-25", "2023-24", "2022-23"]
selected_seasons = st.sidebar.multiselect("Seasons to include", season_options, default=["2024-25"])
min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 40, 20, 1)
home_filter = st.sidebar.selectbox("Game Location", ["All", "Home Only", "Away Only"])

# -----------------------------
# Parlay Legs
# -----------------------------
legs = st.session_state.get("legs", [{"player": "", "stat": "PTS", "threshold": 10, "fanduel_odds": -110}])

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("➕ Add Leg"):
        legs.append({"player": "", "stat": "PTS", "threshold": 10, "fanduel_odds": -110})
with col2:
    if st.button("➖ Remove Leg") and len(legs) > 1:
        legs.pop()

st.session_state["legs"] = legs

# -----------------------------
# Leg Inputs
# -----------------------------
stat_options = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "FG3M": "3PM"
}

for i, leg in enumerate(legs):
    with st.expander(f"Leg {i+1}", expanded=True):
        leg["player"] = st.text_input(f"Player {i+1}", leg["player"], key=f"player_{i}")
        leg["stat"] = st.selectbox(f"Stat {i+1}", list(stat_options.keys()), format_func=lambda x: stat_options[x], key=f"stat_{i}")
        leg["threshold"] = st.number_input(f"Threshold {i+1} (≥)", min_value=0, max_value=100, value=leg["threshold"], step=1, key=f"thresh_{i}")
        leg["fanduel_odds"] = st.number_input(f"FanDuel Odds {i+1}", value=leg["fanduel_odds"], step=5, key=f"fdodds_{i}")

# -----------------------------
# Compute
# -----------------------------
if st.button("Compute"):
    st.markdown("---")

    rows = []
    all_probs = []
    all_implieds = []
    evs = []

    for leg in legs:
        player_name = leg["player"]
        stat = leg["stat"]
        threshold = leg["threshold"]
        fanduel_odds = leg["fanduel_odds"]

        player_id = get_player_id(player_name)
        if not player_id:
            st.warning(f"Player '{player_name}' not found.")
            continue

        df = get_player_gamelog(player_id, selected_seasons)
        home_only = None
        if home_filter == "Home Only":
            home_only = True
        elif home_filter == "Away Only":
            home_only = False

        prob, hits, total = calculate_probability(df, stat, threshold, home_only, min_minutes)
        fair_odds = prob_to_american(prob)
        model_implied = prob
        book_implied = american_to_implied(fanduel_odds)
        ev = (model_implied - book_implied) * 100
        evs.append(ev)
        all_probs.append(model_implied)
        all_implieds.append(book_implied)

        color = "#0b3d23" if ev >= 0 else "#3d0b0b"
        border_color = "#00FF99" if ev >= 0 else "#FF5555"
        emoji = "🔥" if ev >= 0 else "⚠️"

        st.markdown(f"""
        <div style='background-color:{color};padding:20px;border-radius:15px;margin-bottom:20px;border:1px solid {border_color};'>
            <h3 style='color:white;'>{player_name} — <span style='color:#9AE6B4;'>{selected_seasons[0]}</span></h3>
            <p><b>Condition:</b> {threshold}+ {stat_options[stat].lower()}</p>
            <p><b>Model Hit Rate:</b> {model_implied*100:.1f}% ({hits}/{total})</p>
            <p><b>Model Fair Odds:</b> {fair_odds}</p>
            <p><b>FanDuel Odds:</b> {fanduel_odds}</p>
            <p><b>Book Implied:</b> {book_implied*100:.1f}%</p>
            <p><b>Expected Value:</b> <span style='font-size:22px;color:#00FF99;'>{ev:.2f}%</span></p>
            <div style='height:10px;background-color:#333;border-radius:10px;'>
                <div style='height:10px;width:{min(abs(ev),100)}%;background-color:{border_color};border-radius:10px;'></div>
            </div>
            <p style='margin-top:10px;font-size:17px;color:white;'>{emoji} <b>{'+EV Play Detected' if ev >= 0 else 'Negative EV Play'}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Histogram
        fig, ax = plt.subplots()
        ax.hist(df[stat], bins=20, edgecolor="black", color="#00c896" if ev >= 0 else "#e05a5a")
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold {threshold}")
        ax.set_title(f"{player_name} — {stat_options[stat]}")
        ax.set_xlabel(stat_options[stat])
        ax.set_ylabel("Games")
        ax.legend()
        st.pyplot(fig)

    # -----------------------------
    # Parlay Summary (NEW SECTION)
    # -----------------------------
    if all_probs:
        combined_prob = np.prod(all_probs)
        combined_odds = prob_to_american(combined_prob)
        combined_book_prob = np.prod(all_implieds)
        parlay_ev = (combined_prob - combined_book_prob) * 100

        color = "#0b3d23" if parlay_ev >= 0 else "#3d0b0b"
        border_color = "#00FF99" if parlay_ev >= 0 else "#FF5555"
        emoji = "🔥" if parlay_ev >= 0 else "⚠️"

        st.markdown("---")
        st.subheader("💥 Combined Parlay Summary")

        st.markdown(f"""
        <div style='background-color:{color};padding:25px;border-radius:15px;border:1px solid {border_color};margin-bottom:25px;'>
            <h2 style='color:white;'>Combined Parlay — <span style='color:#9AE6B4;'>{selected_seasons[0]}</span></h2>
            <p><b>Parlay Probability:</b> {combined_prob*100:.2f}%</p>
            <p><b>Model Fair Odds:</b> {combined_odds}</p>
            <p><b>Expected Value:</b> <span style='font-size:22px;color:#00FF99;'>{parlay_ev:.2f}%</span></p>
            <div style='height:10px;background-color:#333;border-radius:10px;'>
                <div style='height:10px;width:{min(abs(parlay_ev),100)}%;background-color:{border_color};border-radius:10px;'></div>
            </div>
            <p style='margin-top:10px;font-size:17px;color:white;'>{emoji} <b>{'+EV Parlay Detected' if parlay_ev >= 0 else 'Negative EV Parlay'}</b></p>
        </div>
        """, unsafe_allow_html=True)
