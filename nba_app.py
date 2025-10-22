# Clean & stable version — no pyarrow, no table, no combined summary
# Run:
#   pip install streamlit pandas numpy matplotlib nba_api
#   streamlit run nba_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# =========================
# Page setup
# =========================
st.set_page_config(page_title="NBA Parlay Builder", layout="wide")
st.title("🏀 NBA Parlay Builder (clean + stable)")

# =========================
# Helper functions
# =========================
@st.cache_data(show_spinner=False)
def get_player_id(name):
    result = players.find_players_by_full_name(name)
    return result[0]['id'] if result else None

@st.cache_data(show_spinner=True, ttl=3600)
def get_player_gamelog(player_id, seasons):
    dfs = []
    for season in seasons:
        try:
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
            dfs.append(gamelog)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.astype({
        "PTS": float, "REB": float, "AST": float,
        "STL": float, "BLK": float, "FG3M": float, "MIN": float
    }, errors="ignore")
    return df

def calculate_probability(df, stat, threshold, home_only=None, min_minutes=20):
    if df.empty:
        return 0.0, 0, 0
    df = df[df["MIN"] >= min_minutes]
    if home_only is not None:
        if home_only:
            df = df[df["MATCHUP"].str.contains("vs.")]
        else:
            df = df[df["MATCHUP"].str.contains("@")]
    total = len(df)
    if total == 0:
        return 0.0, 0, 0
    hits = (df[stat] >= threshold).sum()
    return hits / total, hits, total

def prob_to_american(prob):
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob > 0.5:
        odds = -100 * prob / (1 - prob)
    else:
        odds = 100 * (1 - prob) / prob
    return f"{int(round(odds)):+}"

def american_to_prob(odds):
    if odds == 0:
        return 0
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Filters")

season_options = ["2024-25", "2023-24", "2022-23"]
selected_seasons = st.sidebar.multiselect("Seasons to include", season_options, default=["2024-25"])
min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 40, 20, 1)
home_filter = st.sidebar.selectbox("Game Location", ["All", "Home Only", "Away Only"])
show_positive_only = st.sidebar.checkbox("Show only +EV plays", value=False)

# Manage legs in session state
if "legs" not in st.session_state:
    st.session_state.legs = [{"player": "", "stat": "PTS", "threshold": 10, "book_odds": -110}]

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({"player": "", "stat": "PTS", "threshold": 10, "book_odds": -110})
with col2:
    if st.button("➖ Remove Leg") and len(st.session_state.legs) > 1:
        st.session_state.legs.pop()

# Stat options
stat_options = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "FG3M": "3PM"
}

# Leg inputs
for i, leg in enumerate(st.session_state.legs):
    with st.expander(f"Leg {i+1}", expanded=True):
        leg["player"] = st.text_input(f"Player {i+1}", leg["player"], key=f"player_{i}")
        leg["stat"] = st.selectbox(f"Stat {i+1}", list(stat_options.keys()),
                                   format_func=lambda x: stat_options[x], key=f"stat_{i}")
        leg["threshold"] = st.number_input(f"Threshold {i+1} (≥)", min_value=0, max_value=100,
                                           value=leg["threshold"], step=1, key=f"thresh_{i}")
        leg["book_odds"] = st.number_input(f"FanDuel Odds {i+1} (e.g. -110, +120)",
                                           value=leg["book_odds"], step=5, key=f"odds_{i}")

# =========================
# Compute
# =========================
if st.button("Compute"):
    st.subheader("📊 Parlay Results")

    for leg in st.session_state.legs:
        player_name = leg["player"]
        stat = leg["stat"]
        threshold = leg["threshold"]
        book_odds = leg["book_odds"]

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
        book_prob = american_to_prob(book_odds)
        ev = (prob * (1 - book_prob)) - ((1 - prob) * book_prob)
        ev_pct = ev * 100

        if show_positive_only and ev_pct <= 0:
            continue

        # =========================
        # Styled Card
        # =========================
        bar_color = "#00E676" if ev_pct > 0 else "#E53935"
        bg_color = "#173B1A" if ev_pct > 0 else "#2B2B2B"
        border_color = "#00C853" if ev_pct > 0 else "#444444"
        emoji = "🔥" if ev_pct > 0 else "❄️"

        card_html = f"""
        <div style="background:{bg_color}; border:2px solid {border_color};
                    border-radius:12px; padding:20px; margin-bottom:25px;">
            <h3 style="margin:0 0 10px 0; color:white;">{player_name} — <span style="opacity:0.6;">{', '.join(selected_seasons)}</span></h3>
            <p style="color:#DDD; font-size:16px; margin:0 0 10px 0;">
                <strong>Condition:</strong> {threshold}+ {stat_options[stat].lower()}
            </p>
            <div style="display:flex; flex-wrap:wrap; justify-content:space-between;">
                <div style="flex:1 1 45%; padding:5px;">
                    <span style="color:#bbb;">Model Hit Rate</span><br>
                    <strong style="color:white;">{prob*100:.1f}%</strong>
                    <span style="color:#aaa;">({hits}/{total})</span>
                </div>
                <div style="flex:1 1 45%; padding:5px;">
                    <span style="color:#bbb;">Model Fair Odds</span><br>
                    <strong style="color:white;">{fair_odds}</strong>
                </div>
                <div style="flex:1 1 45%; padding:5px;">
                    <span style="color:#bbb;">FanDuel Odds</span><br>
                    <strong style="color:white;">{book_odds}</strong>
                </div>
                <div style="flex:1 1 45%; padding:5px;">
                    <span style="color:#bbb;">Book Implied</span><br>
                    <strong style="color:white;">{book_prob*100:.1f}%</strong>
                </div>
                <div style="flex:1 1 100%; padding:5px; margin-top:6px;">
                    <span style="color:#bbb;">Expected Value</span><br>
                    <strong style="font-size:22px; color:{bar_color};">{ev_pct:.2f}%</strong>
                    <div style="width:100%; background:#444; height:10px; border-radius:6px; margin-top:6px;">
                        <div style="width:{min(abs(ev_pct), 100)}%; background:{bar_color};
                                    height:100%; border-radius:6px;"></div>
                    </div>
                </div>
            </div>
            <p style="margin-top:10px; font-size:17px; color:white;">
                {emoji} {"<strong>+EV Play Detected</strong>" if ev_pct > 0 else "<strong>No +EV Edge</strong>"}
            </p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # =========================
        # Histogram
        # =========================
        if stat in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df[stat], bins=20, edgecolor="black")
            ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold {threshold}")
            ax.set_title(f"{player_name} — {stat_options[stat]}")
            ax.set_xlabel(stat_options[stat])
            ax.set_ylabel("Games")
            ax.legend()
            st.pyplot(fig)
