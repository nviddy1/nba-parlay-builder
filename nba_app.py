import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# -----------------------------
# Page setup & minimal CSS
# -----------------------------
st.set_page_config(page_title="NBA Parlay Builder", layout="wide")

st.markdown(
    """
    <style>
      .card {
        padding: 22px;
        border-radius: 18px;
        margin-bottom: 18px;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
      }
      .neutral { --card-bg: #222; --card-border: #888; }
      .pos     { --card-bg: #0b3d23; --card-border: #00FF99; }
      .neg     { --card-bg: #3d0b0b; --card-border: #FF5555; }

      .metric-label {
        color: #cbd5e1;
        font-size: 16px;
        margin-bottom: 6px;
      }
      .metric-value {
        color: #ffffff;
        font-size: 44px;
        font-weight: 800;
        line-height: 1.0;
      }
      .small-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid #16a34a33;
        color: #a7f3d0;
        font-size: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏀 NBA Parlay Builder (add as many legs as you like)")

# -----------------------------
# Helpers
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
    # Accept 0/None/strings, return None for "not usable"
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

def fmt_pct(x, places=1):
    try:
        return f"{x*100:.{places}f}%"
    except Exception:
        return "—"

def metric(col, label, value):
    with col:
        st.markdown(f"<div class='metric-label'>{label}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)

# -----------------------------
# Sidebar Filters (incl. parlay odds)
# -----------------------------
st.sidebar.header("Filters")

season_options = ["2024-25", "2023-24", "2022-23"]
selected_seasons = st.sidebar.multiselect("Seasons to include", season_options, default=["2024-25"])
min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 40, 20, 1)
home_filter = st.sidebar.selectbox("Game Location", ["All", "Home Only", "Away Only"])

st.sidebar.markdown("---")
parlay_odds = st.sidebar.number_input("Combined Parlay Odds (e.g., +300, -150)", value=0, step=5)

# -----------------------------
# Manage Legs
# -----------------------------
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

# -----------------------------
# Compute
# -----------------------------
if st.button("Compute"):
    st.markdown("---")

    home_only = None
    if home_filter == "Home Only":
        home_only = True
    elif home_filter == "Away Only":
        home_only = False

    rows = []
    model_probs = []

    for leg in st.session_state.legs:
        name = leg["player"].strip()
        stat = leg["stat"]
        thr = int(leg["threshold"])
        book_odds = int(leg["odds"])

        pid = get_player_id(name)
        if not pid:
            # still keep a row for UI consistency
            rows.append(
                dict(
                    name=name or "Unknown",
                    label=f"{thr}+ {stat_options[stat]}",
                    prob=0.0, hits=0, total=0,
                    fair="N/A",
                    book_odds=book_odds,
                    book_prob=american_to_implied(book_odds),
                    ev=None,
                    df=pd.DataFrame(),
                    stat=stat, thr=thr,
                )
            )
            continue

        df = get_player_gamelog(pid, selected_seasons)
        prob, hits, total, df_filt = calculate_probability(df, stat, thr, home_only, min_minutes)
        fair = prob_to_american(prob)
        book_prob = american_to_implied(book_odds)
        ev = None if book_prob is None else (prob - book_prob) * 100

        if prob > 0:
            model_probs.append(prob)

        rows.append(
            dict(
                name=name,
                label=f"{thr}+ {stat_options[stat]}",
                prob=prob, hits=hits, total=total,
                fair=fair,
                book_odds=book_odds,
                book_prob=book_prob,
                ev=ev,
                df=df_filt,
                stat=stat, thr=thr,
            )
        )

    # -----------------------------
    # Combined Parlay Summary (FIRST)
    # -----------------------------
    combined_prob = float(np.prod(model_probs)) if model_probs else 0.0
    combined_odds = prob_to_american(combined_prob) if combined_prob > 0 else "N/A"
    book_parlay_prob = american_to_implied(parlay_odds)
    parlay_ev = None if book_parlay_prob is None else (combined_prob - book_parlay_prob) * 100

    # styling
    if parlay_ev is None:
        card_class, emoji = "neutral", "ℹ️"
    else:
        card_class, emoji = ("pos", "🔥") if parlay_ev >= 0 else ("neg", "⚠️")

    st.markdown(f"<div class='card {card_class}'>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:white;margin-top:0;'>💥 Combined Parlay — {', '.join(selected_seasons) if selected_seasons else '—'}</h2>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.2])
    metric(c1, "Model Parlay Probability", f"{combined_prob*100:.2f}%")
    metric(c2, "Model Fair Odds", f"{combined_odds}")
    metric(c3, "Entered Parlay Odds", f"{parlay_odds if parlay_odds else '—'}")
    metric(c4, "Book Implied", f"{book_parlay_prob*100:.2f}%" if book_parlay_prob is not None else "—")
    metric(c5, "Expected Value", f"{parlay_ev:.2f}%" if parlay_ev is not None else "—")

    st.markdown(
        f"<div class='small-chip'>{emoji} {'+EV Parlay' if (parlay_ev is not None and parlay_ev >= 0) else ('Negative EV Parlay' if parlay_ev is not None else 'Enter parlay odds in sidebar')}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------
    # Individual Legs (cards + L→R metrics + histogram)
    # -----------------------------
    for r in rows:
        # card style
        if r["ev"] is None:
            card_class, emoji = "neutral", "ℹ️"
        else:
            card_class, emoji = ("pos", "🔥") if r["ev"] >= 0 else ("neg", "⚠️")

        st.markdown(f"<div class='card {card_class}'>", unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='color:white;margin-top:0;'>{r['name']} — {r['label']}</h2>",
            unsafe_allow_html=True,
        )
        a, b, c, d, e = st.columns([1, 1, 1, 1, 1.2])
        metric(a, "Model Hit Rate", f"{r['prob']*100:.1f}%")
        metric(b, "Model Fair Odds", f"{r['fair']}")
        metric(c, "FanDuel Odds", f"{r['book_odds']}")
        metric(d, "Book Implied", f"{r['book_prob']*100:.1f}%" if r["book_prob"] is not None else "—")
        metric(e, "Expected Value", f"{r['ev']:.2f}%" if r["ev"] is not None else "—")
        st.markdown(
            f"<div class='small-chip'>{emoji} {'+EV Play' if (r['ev'] is not None and r['ev'] >= 0) else ('Negative EV Play' if r['ev'] is not None else 'Add odds to compute EV')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # histogram
        if not r["df"].empty and r["stat"] in r["df"].columns:
            fig, ax = plt.subplots()
            ax.hist(
                r["df"][r["stat"]],
                bins=20,
                edgecolor="black",
                color="#00c896" if (r["ev"] is not None and r["ev"] >= 0) else "#e05a5a",
            )
            ax.axvline(r["thr"], color="red", linestyle="--", label=f"Threshold {r['thr']}")
            ax.set_title(f"{r['name']} — {stat_options[r['stat']]}")
            ax.set_xlabel(stat_options[r["stat"]])
            ax.set_ylabel("Games")
            ax.legend()
            st.pyplot(fig)
