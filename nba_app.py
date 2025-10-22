import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# -----------------------------
# PAGE CONFIG + STYLES
# -----------------------------
st.set_page_config(page_title="NBA Parlay Builder", layout="wide")

st.markdown("""
<style>
  body { background-color: #111; color: white; }

  .card {
    padding: 26px 32px;
    border-radius: 18px;
    margin-bottom: 24px;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    box-shadow: 0 0 15px rgba(0,0,0,0.25);
    width: 100%;
  }
  .neutral { --card-bg: #222; --card-border: #888; }
  .pos     { --card-bg: #0b3d23; --card-border: #00FF99; }
  .neg     { --card-bg: #3d0b0b; --card-border: #FF5555; }

  .card h2 {
    color: #fff;
    margin-top: 0;
    margin-bottom: 4px;
    font-weight: 700;
  }

  .condition-line {
    color: #a3a3a3;
    font-size: 15px;
    margin-bottom: 10px;
  }

  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 10px;
    margin-bottom: 12px;
  }

  .metric-box {
    flex: 1;
    min-width: 130px;
  }

  .metric-label {
    color: #cbd5e1;
    font-size: 15px;
    margin-bottom: 4px;
  }

  .metric-value {
    color: #fff;
    font-size: 28px;
    font-weight: 700;
    line-height: 1.1;
  }

  .small-chip {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    border: 1px solid #16a34a33;
    color: #a7f3d0;
    font-size: 14px;
    margin-top: 14px;
  }
</style>
""", unsafe_allow_html=True)

st.title("🏀 NBA Parlay Builder (Add as many legs as you like)")

# -----------------------------
# HELPERS
# -----------------------------
def get_player_id(name: str):
    res = players.find_players_by_full_name(name)
    return res[0]["id"] if res else None

def get_player_gamelog(player_id: int, seasons: list[str]) -> pd.DataFrame:
    dfs = []
    for s in seasons:
        try:
            g = playergamelog.PlayerGameLog(player_id=player_id, season=s).get_data_frames()[0]
            dfs.append(g)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.astype(
        {"PTS": float, "REB": float, "AST": float, "STL": float, "BLK": float, "FG3M": float, "MIN": float},
        errors="ignore",
    )
    return df

def calculate_probability(df: pd.DataFrame, stat: str, threshold: int, home_only=None, min_minutes=20):
    if df.empty:
        return 0.0, 0, 0, df
    df = df[df["MIN"] >= min_minutes]
    if home_only is True:
        df = df[df["MATCHUP"].str.contains("vs.")]
    elif home_only is False:
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

def american_to_implied(odds: float | int):
    if odds in (None, 0, "0", ""):
        return None
    try:
        x = float(odds)
    except Exception:
        return None
    if x > 0:
        return 100 / (x + 100)
    else:
        return abs(x) / (abs(x) + 100)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

season_options = ["2024-25", "2023-24", "2022-23"]
selected_seasons = st.sidebar.multiselect("Seasons to include", season_options, default=["2024-25"])
min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 40, 20, 1)
home_filter = st.sidebar.selectbox("Game Location", ["All", "Home Only", "Away Only"])

if "legs" not in st.session_state:
    st.session_state.legs = [{"player": "", "stat": "PTS", "threshold": 10, "odds": -110}]

lc1, lc2 = st.columns(2)
with lc1:
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({"player": "", "stat": "PTS", "threshold": 10, "odds": -110})
with lc2:
    if st.button("➖ Remove Leg") and len(st.session_state.legs) > 1:
        st.session_state.legs.pop()

stat_options = {"PTS": "Points", "REB": "Rebounds", "AST": "Assists", "STL": "Steals", "BLK": "Blocks", "FG3M": "3PM"}

for i, leg in enumerate(st.session_state.legs):
    with st.expander(f"Leg {i+1}", expanded=True):
        leg["player"] = st.text_input(f"Player {i+1}", leg["player"], key=f"p_{i}")
        leg["stat"] = st.selectbox(f"Stat {i+1}", list(stat_options.keys()), format_func=lambda x: stat_options[x], key=f"s_{i}")
        leg["threshold"] = st.number_input(f"Threshold (≥) {i+1}", 0, 100, leg["threshold"], key=f"t_{i}")
        leg["odds"] = st.number_input(f"FanDuel Odds {i+1}", -10000, 10000, leg["odds"], step=5, key=f"o_{i}")

# Show parlay odds input if more than one leg
if len(st.session_state.legs) > 1:
    st.sidebar.markdown("---")
    parlay_odds = st.sidebar.number_input("Combined Parlay Odds (e.g., +300, -150)", value=0, step=5)
else:
    parlay_odds = 0

# -----------------------------
# COMPUTE
# -----------------------------
if st.button("Compute"):
    st.markdown("---")
    home_only = True if home_filter == "Home Only" else False if home_filter == "Away Only" else None

    rows, model_probs = [], []
    for leg in st.session_state.legs:
        name, stat, thr, book_odds = leg["player"].strip(), leg["stat"], int(leg["threshold"]), int(leg["odds"])
        pid = get_player_id(name)
        if not pid:
            rows.append(dict(name=name or "Unknown", label=f"{thr}+ {stat_options[stat]}", prob=0.0, hits=0, total=0,
                             fair="N/A", book_odds=book_odds, book_prob=american_to_implied(book_odds), ev=None,
                             df=pd.DataFrame(), stat=stat, thr=thr))
            continue

        df = get_player_gamelog(pid, selected_seasons)
        prob, hits, total, df_filt = calculate_probability(df, stat, thr, home_only, min_minutes)
        fair, book_prob = prob_to_american(prob), american_to_implied(book_odds)
        ev = None if book_prob is None else (prob - book_prob) * 100
        if prob > 0: model_probs.append(prob)
        rows.append(dict(name=name, label=f"{thr}+ {stat_options[stat]}", prob=prob, hits=hits, total=total,
                         fair=fair, book_odds=book_odds, book_prob=book_prob, ev=ev,
                         df=df_filt, stat=stat, thr=thr))

    # COMBINED PARLAY SUMMARY
    combined_prob = float(np.prod(model_probs)) if model_probs else 0.0
    combined_odds = prob_to_american(combined_prob) if combined_prob > 0 else "N/A"
    book_parlay_prob = american_to_implied(parlay_odds)
    parlay_ev = None if book_parlay_prob is None else (combined_prob - book_parlay_prob) * 100
    card_class, emoji = ("pos", "🔥") if parlay_ev and parlay_ev >= 0 else ("neg", "⚠️") if parlay_ev else ("neutral", "ℹ️")

    with st.container():
        st.markdown(f"<div class='card {card_class}'>", unsafe_allow_html=True)
        st.markdown(f"<h2>💥 Combined Parlay — {', '.join(selected_seasons)}</h2>", unsafe_allow_html=True)
        st.markdown("<div class='condition-line'>Includes all selected legs and filters</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Model Parlay Probability</div><div class='metric-value'>{combined_prob*100:.2f}%</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Model Fair Odds</div><div class='metric-value'>{combined_odds}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Entered Parlay Odds</div><div class='metric-value'>{parlay_odds if parlay_odds else '—'}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Book Implied</div><div class='metric-value'>{book_parlay_prob*100:.2f}%</div></div>" if book_parlay_prob else "<div class='metric-box'><div class='metric-label'>Book Implied</div><div class='metric-value'>—</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Expected Value</div><div class='metric-value'>{parlay_ev:.2f}%</div></div>" if parlay_ev is not None else "<div class='metric-box'><div class='metric-label'>Expected Value</div><div class='metric-value'>—</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-chip'>{emoji} {'+EV Parlay Detected' if (parlay_ev and parlay_ev >= 0) else ('Negative EV Parlay' if parlay_ev is not None else 'Enter parlay odds')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # INDIVIDUAL LEGS
    for r in rows:
        card_class, emoji = ("pos", "🔥") if r["ev"] and r["ev"] >= 0 else ("neg", "⚠️") if r["ev"] else ("neutral", "ℹ️")
        with st.container():
            st.markdown(f"<div class='card {card_class}'>", unsafe_allow_html=True)
            st.markdown(f"<h2>{r['name']} — {', '.join(selected_seasons)}</h2>", unsafe_allow_html=True)
            st.markdown(f"<div class='condition-line'>Condition: {r['thr']}+ {stat_options[r['stat']].lower()}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'><div class='metric-label'>Model Hit Rate</div><div class='metric-value'>{r['prob']*100:.1f}% ({r['hits']}/{r['total']})</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'><div class='metric-label'>Model Fair Odds</div><div class='metric-value'>{r['fair']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'><div class='metric-label'>FanDuel Odds</div><div class='metric-value'>{r['book_odds']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'><div class='metric-label'>Book Implied</div><div class='metric-value'>{r['book_prob']*100:.1f}%</div></div>" if r["book_prob"] else "<div class='metric-box'><div class='metric-label'>Book Implied</div><div class='metric-value'>—</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'><div class='metric-label'>Expected Value</div><div class='metric-value'>{r['ev']:.2f}%</div></div>" if r["ev"] is not None else "<div class='metric-box'><div class='metric-label'>Expected Value</div><div class='metric-value'>—</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-chip'>{emoji} {'+EV Play Detected (by your model)' if (r['ev'] and r['ev'] >= 0) else ('Negative EV Play' if r['ev'] else 'Add odds to compute EV')}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Histogram
            if not r["df"].empty and r["stat"] in r["df"].columns:
                fig, ax = plt.subplots()
                ax.hist(r["df"][r["stat"]], bins=20, edgecolor="black",
                        color="#00c896" if (r["ev"] and r["ev"] >= 0) else "#e05a5a")
                ax.axvline(r["thr"], color="red", linestyle="--", label=f"Threshold {r['thr']}")
                ax.set_title(f"{r['name']} — {stat_options[r['stat']]}")
                ax.set_xlabel(stat_options[r["stat"]])
                ax.set_ylabel("Games")
                ax.legend()
                st.pyplot(fig)
