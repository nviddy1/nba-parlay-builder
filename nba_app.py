import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# Use a non-GUI backend for reliability
matplotlib.use("Agg")

st.set_page_config(page_title="NBA Stat Tracker", page_icon="🏀", layout="wide")
st.title("🏀 NBA Player Stat Explorer — Dual Stat Edition")

# Inputs
player_name = st.text_input("Enter player name (e.g., Chet Holmgren):")
season = "2024-25"

c1, c2, c3 = st.columns(3)
with c1:
    home_only = st.checkbox("Home only")
with c2:
    away_only = st.checkbox("Away only")
with c3:
    min20 = st.checkbox("Played ≥ 20 min only")

vs_team = st.text_input("Filter by opponent (e.g., LAL, DEN, BOS). Leave blank for all:")

st.markdown("### Stat Filters")
s1, s2 = st.columns(2)
with s1:
    stat1 = st.selectbox("Primary Stat", ["Points", "Rebounds", "Assists", "Blocks", "Steals"])
    thresh1 = st.number_input("Threshold for primary stat", min_value=0, max_value=100, value=10)
with s2:
    enable_second = st.checkbox("Add a second stat filter (AND condition)")
    if enable_second:
        stat2 = st.selectbox("Secondary Stat", ["Points", "Rebounds", "Assists", "Blocks", "Steals"], index=1)
        thresh2 = st.number_input("Threshold for secondary stat", min_value=0, max_value=100, value=5)

if st.button("Analyze"):
    try:
        # --- Player lookup ---
        hit = players.find_players_by_full_name(player_name)
        if not hit:
            st.error("Player not found. Check spelling.")
            st.stop()
        player_id = hit[0]["id"]
        full_name = hit[0]["full_name"]

        # --- Fetch game log for the season ---
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gl.get_data_frames()[0]
        if df.empty:
            st.warning(f"No game data for {full_name} in {season}.")
            st.stop()

        # --- Tidy columns ---
        df = df.rename(columns={
            "PTS": "points", "REB": "rebounds", "AST": "assists",
            "BLK": "blocks", "STL": "steals", "MIN": "minutes"
        })
        # MIN can be "28:34" or numeric; convert to integer minutes
        df["minutes"] = (
            df["minutes"].astype(str)
            .apply(lambda x: int(x.split(":")[0]) if ":" in x else pd.to_numeric(x, errors="coerce"))
        )

        # --- Filters ---
        if home_only and not away_only:
            df = df[df["MATCHUP"].str.contains("vs")]
        elif away_only and not home_only:
            df = df[df["MATCHUP"].str.contains("@")]

        if vs_team:
            df = df[df["MATCHUP"].str.contains(vs_team.upper())]

        if min20:
            df = df[df["minutes"] >= 20]

        # --- Derived metrics ---
        stat_cols = ["points", "rebounds", "assists", "blocks", "steals"]
        df["double_double"] = df[stat_cols].apply(lambda r: (r >= 10).sum() >= 2, axis=1)
        df["triple_double"] = df[stat_cols].apply(lambda r: (r >= 10).sum() >= 3, axis=1)

        # --- Apply stat conditions ---
        stat1_col = stat1.lower()
        cond = df[stat1_col] >= thresh1
        if enable_second:
            stat2_col = stat2.lower()
            cond &= df[stat2_col] >= thresh2

        filtered_df = df[cond]
        total_games = len(df)
        hits = len(filtered_df)

        # --- Summary ---
        if enable_second:
            st.subheader(
                f"{full_name} had {thresh1}+ {stat1.lower()} AND {thresh2}+ {stat2.lower()} "
                f"in {hits}/{total_games} games ({season})."
            )
        else:
            st.subheader(
                f"{full_name} had {thresh1}+ {stat1.lower()} in {hits}/{total_games} games ({season})."
            )

        st.markdown(
            f"🏀 **{int(df['double_double'].sum())} Double-Doubles** & "
            f"**{int(df['triple_double'].sum())} Triple-Doubles** this season (filtered view)."
        )

        # --- Visualization ---
        if enable_second:
            # Clustered per-game bar chart (Matplotlib – avoids Altair/TypedDict bug)
            y1 = df[stat1_col].astype(float).values
            y2 = df[stat2_col].astype(float).values
            n = len(df)
            x = range(n)
            width = 0.45

            fig, ax = plt.subplots(figsize=(min(12, max(6, n * 0.22)), 5))
            ax.bar([i - width/2 for i in x], y1, width, label=stat1)
            ax.bar([i + width/2 for i in x], y2, width, label=stat2)

            # Make x labels readable (use GAME_DATE to help)
            dates = pd.to_datetime(df["GAME_DATE"])
            labels = dates.dt.strftime("%m/%d")
            step = max(1, n // 12)
            ax.set_xticks(list(x)[::step])
            ax.set_xticklabels(labels[::step], rotation=0)
            ax.set_title(f"{full_name} — {stat1} vs {stat2} by Game ({season})")
            ax.set_xlabel("Game date")
            ax.set_ylabel("Stat value")
            ax.legend()
            st.pyplot(fig)
        else:
            # Single-stat histogram
            fig, ax = plt.subplots()
            ax.hist(df[stat1_col], bins=15, edgecolor="black")
            ax.axvline(thresh1, color="red", linestyle="--", label=f"{thresh1} {stat1}")
            ax.set_title(f"{full_name} — {stat1} Distribution ({season})")
            ax.set_xlabel(stat1)
            ax.set_ylabel("Games")
            ax.legend()
            st.pyplot(fig)

        # --- Opponent averages (only show when not already filtering by a team) ---
        if not vs_team:
            opp = df["MATCHUP"].str.extract(r"([A-Z]{3})$")[0]
            opp_avg = df.groupby(opp)[stat1_col].mean().sort_values(ascending=False)
            st.subheader("Average by Opponent")
            st.bar_chart(opp_avg)

    except Exception as e:
        st.error(f"Error: {e}")
