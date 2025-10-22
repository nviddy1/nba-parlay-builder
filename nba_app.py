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
def get_player_id(name: str):
    result = players.find_players_by_full_name(name)
    return result[0]["id"] if result else None

def get_player_gamelog(player_id: int, seasons: list[str]) -> pd.DataFrame:
    dfs = []
    for season in seasons:
        try:
            g = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
            dfs.append(g)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.astype(
        {
            "PTS": float,
            "REB": float,
            "AST": float,
            "STL": float,
            "BLK": float,
            "FG3M": float,
            "MIN": float,
        },
        errors="ignore",
    )
    return df

def calculate_probability(df: pd.DataFrame, stat: str, threshold: int, home_only=None, min_minutes=20):
    if df.empty:
        return 0.0, 0, 0, df
    df = df[df["MIN"] >= min_minutes]
    if home_only is not None:
        if home_only:
            df = df[df["MATCHUP"].str.contains("vs.")]
        else:
            df = df[df["MATCHUP"].str.contains("@")]
    total = len(df)
    if total == 0:
        return 0.0, 0, 0, df
    hits = (df[stat] >= threshold).sum()
    prob = hits / total
    return prob, hits, total, df

def prob_to_american(prob: float):
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob > 0.5:
        odds = -100 * prob / (1 - prob)
    else:
        odds = 100 * (1 - prob) / prob
    return f"{int(round(odds)):+}"

def american_to_implied(odds: int | float) -> float:
    """Return implied probability from American odds."""
    if odds == 0:
        return None  # caller must handle None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Filters")
season_options = ["2024-25", "2023-24", "2022-23"]
selected_seasons = st.sidebar.multiselect("Seasons to include", season_options, default=["2024-25"])
min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 40, 20, 1)
home_filter = st.sidebar.selectbox("Game Location", ["All", "Home Only", "Away Only"])

# -----------------------------
# Leg state + controls
# -----------------------------
if "legs" not in st.session_state:
    st.session_state.legs = [{"player": "", "stat": "PTS", "threshold": 10, "fanduel_odds": -110}]

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({"player": "", "stat": "PTS", "threshold": 10, "fanduel_odds": -110})
with col2:
    if st.button("➖ Remove Leg") and len(st.session_state.legs) > 1:
        st.session_state.legs.pop()

stat_options = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "FG3M": "3PM",
}

for i, leg in enumerate(st.session_state.legs):
    with st.expander(f"Leg {i+1}", expanded=True):
        leg["player"] = st.text_input(f"Player {i+1}", leg["player"], key=f"player_{i}")
        leg["stat"] = st.selectbox(
            f"Stat {i+1}", list(stat_options.keys()), format_func=lambda x: stat_options[x], key=f"stat_{i}"
        )
        leg["threshold"] = st.number_input(
            f"Threshold {i+1} (≥)", min_value=0, max_value=100, value=leg["threshold"], step=1, key=f"thresh_{i}"
        )
        leg["fanduel_odds"] = st.number_input(
            f"FanDuel Odds {i+1}", value=leg["fanduel_odds"], step=5, key=f"odds_{i}"
        )

# -----------------------------
# Compute
# -----------------------------
if st.button("Compute"):
    # First compute all legs silently so we can show the parlay summary FIRST
    rows = []
    all_model_probs = []
    all_book_probs = []

    # Decide location filter
    home_only = None
    if home_filter == "Home Only":
        home_only = True
    elif home_filter == "Away Only":
        home_only = False

    for leg in st.session_state.legs:
        player_name = leg["player"].strip()
        stat = leg["stat"]
        threshold = int(leg["threshold"])
        book_odds = int(leg["fanduel_odds"])

        player_id = get_player_id(player_name)
        if not player_id:
            # Store a stub row so the UI keeps order; skip probability
            rows.append(
                {
                    "player": player_name,
                    "stat": stat,
                    "threshold": threshold,
                    "hits": 0,
                    "total": 0,
                    "prob": 0.0,
                    "fair_odds": "N/A",
                    "book_odds": book_odds,
                    "book_implied": american_to_implied(book_odds),
                    "ev": -999,  # force red card
                    "df": pd.DataFrame(),
                }
            )
            continue

        df = get_player_gamelog(player_id, selected_seasons)
        prob, hits, total, filtered_df = calculate_probability(df, stat, threshold, home_only, min_minutes)
        fair_odds = prob_to_american(prob)
        book_implied = american_to_implied(book_odds)

        # EV in percentage points
        ev = None if book_implied is None else (prob - book_implied) * 100

        if prob > 0:
            all_model_probs.append(prob)
        if book_implied is not None:
            all_book_probs.append(book_implied)

        rows.append(
            {
                "player": player_name,
                "stat": stat,
                "threshold": threshold,
                "hits": hits,
                "total": total,
                "prob": prob,
                "fair_odds": fair_odds,
                "book_odds": book_odds,
                "book_implied": book_implied,
                "ev": ev if ev is not None else -999,
                "df": filtered_df,
            }
        )

    # ===== Combined Parlay Summary FIRST =====
    st.subheader("💥 Combined Parlay Summary")

    # Model parlay prob = product of leg probs (only those > 0)
    combined_prob = np.prod(all_model_probs) if all_model_probs else 0.0
    combined_fair_odds = prob_to_american(combined_prob) if combined_prob > 0 else "N/A"

    # Input: user-entered combined parlay odds
    parlay_odds = st.number_input("Enter Combined Parlay Odds (e.g., +300, -150)", value=0, step=5)
    parlay_book_prob = american_to_implied(parlay_odds)  # None if 0
    parlay_ev = None if parlay_book_prob is None else (combined_prob - parlay_book_prob) * 100

    # Colors (neutral if EV not computable yet)
    if parlay_ev is None:
        color = "#1f2937"        # neutral
        border_color = "#64748b"
        emoji = "ℹ️"
    else:
        color = "#0b3d23" if parlay_ev >= 0 else "#3d0b0b"
        border_color = "#00FF99" if parlay_ev >= 0 else "#FF5555"
        emoji = "🔥" if parlay_ev >= 0 else "⚠️"

    book_implied_str = f"{parlay_book_prob*100:.2f}%" if parlay_book_prob is not None else "—"
    ev_str = f"{parlay_ev:.2f}%" if parlay_ev is not None else "—"
    ev_bar = min(abs(parlay_ev), 100) if parlay_ev is not None else 0

    st.markdown(
        f"""
        <div style='background-color:{color};padding:25px;border-radius:15px;border:1px solid {border_color};margin-bottom:25px;'>
            <h2 style='color:white;'>Combined Parlay — <span style='color:#9AE6B4;'>{", ".join(selected_seasons) if selected_seasons else "—"}</span></h2>
            <p><b>Model Parlay Probability:</b> {combined_prob*100:.2f}%</p>
            <p><b>Model Fair Odds:</b> {combined_fair_odds}</p>
            <p><b>Entered Parlay Odds:</b> {parlay_odds if parlay_odds != 0 else "—"}</p>
            <p><b>Book Implied:</b> {book_implied_str}</p>
            <p><b>Expected Value:</b> <span style='font-size:22px;color:#00FF99;'>{ev_str}</span></p>
            <div style='height:10px;background-color:#333;border-radius:10px;'>
                <div style='height:10px;width:{ev_bar}%;background-color:{border_color};border-radius:10px;'></div>
            </div>
            <p style='margin-top:10px;font-size:17px;color:white;'>{emoji} <b>{'+EV Parlay Detected' if (parlay_ev is not None and parlay_ev >= 0) else ('Negative EV Parlay' if parlay_ev is not None else 'Enter parlay odds above')}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ===== Individual Legs (cards + histograms) =====
    for leg in rows:
        ev = leg["ev"]
        is_pos = ev is not None and ev >= 0
        color = "#0b3d23" if is_pos else "#3d0b0b"
        border_color = "#00FF99" if is_pos else "#FF5555"
        emoji = "🔥" if is_pos else "⚠️"
        ev_str_leg = "—" if ev == -999 or ev is None else f"{ev:.2f}%"
        ev_bar_leg = 0 if ev == -999 or ev is None else min(abs(ev), 100)

        st.markdown(
            f"""
            <div style='background-color:{color};padding:20px;border-radius:15px;margin-bottom:20px;border:1px solid {border_color};'>
                <h3 style='color:white;'>{leg["player"] or "Unknown Player"} — <span style='color:#9AE6B4;'>{", ".join(selected_seasons) if selected_seasons else "—"}</span></h3>
                <p><b>Condition:</b> {leg["threshold"]}+ {stat_options[leg["stat"]].lower()}</p>
                <p><b>Model Hit Rate:</b> {(leg["prob"]*100):.1f}% ({leg["hits"]}/{leg["total"]})</p>
                <p><b>Model Fair Odds:</b> {leg["fair_odds"]}</p>
                <p><b>FanDuel Odds:</b> {leg["book_odds"]}</p>
                <p><b>Book Implied:</b> {f'{leg["book_implied"]*100:.1f}%' if leg["book_implied"] is not None else "—"}</p>
                <p><b>Expected Value:</b> <span style='font-size:22px;color:#00FF99;'>{ev_str_leg}</span></p>
                <div style='height:10px;background-color:#333;border-radius:10px;'>
                    <div style='height:10px;width:{ev_bar_leg}%;background-color:{border_color};border-radius:10px;'></div>
                </div>
                <p style='margin-top:10px;font-size:17px;color:white;'>{emoji} <b>{'+EV Play Detected' if is_pos else 'Negative EV Play'}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Histogram (only if df has the stat)
        df_hist = leg["df"]
        if not df_hist.empty and leg["stat"] in df_hist.columns:
            fig, ax = plt.subplots()
            ax.hist(
                df_hist[leg["stat"]],
                bins=20,
                edgecolor="black",
                color="#00c896" if is_pos else "#e05a5a",
            )
            ax.axvline(leg["threshold"], color="red", linestyle="--", label=f"Threshold {leg['threshold']}")
            ax.set_title(f"{leg['player'] or 'Unknown'} — {stat_options[leg['stat']]}")
            ax.set_xlabel(stat_options[leg["stat"]])
            ax.set_ylabel("Games")
            ax.legend()
            st.pyplot(fig)
