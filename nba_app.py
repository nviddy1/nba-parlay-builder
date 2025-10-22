import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# -----------------------------
# PAGE & THEME
# -----------------------------
st.set_page_config(page_title="NBA Parlay Builder", page_icon="🏀", layout="wide")
st.title("🏀 NBA Parlay Builder")

# -----------------------------
# CSS — full-card background + horizontal metrics
# -----------------------------
st.markdown("""
<style>
  .card {
    --pad-x: 28px; --pad-y: 22px;
    padding: var(--pad-y) var(--pad-x);
    border-radius: 18px;
    margin: 14px 0 26px 0;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    box-shadow: 0 0 14px rgba(0,0,0,0.25);
    width: 100%;
  }
  .neutral { --card-bg:#222; --card-border:#777; }
  .pos     { --card-bg:#0b3d23; --card-border:#00FF99; }
  .neg     { --card-bg:#3d0b0b; --card-border:#FF5555; }

  .card h2 { color:#fff; margin:0 0 6px 0; font-weight:800; }
  .cond { color:#a9b1bb; font-size:15px; margin: 2px 0 12px 0; }

  /* single real row of metrics, wraps if narrow */
  .row {
    display:flex; flex-wrap:wrap; gap:16px;
    align-items:flex-end; justify-content:space-between;
    margin: 8px 0 6px 0;
  }
  .m {
    min-width:140px; flex:1;
  }
  .lab { color:#cbd5e1; font-size:14px; margin-bottom:4px; }
  .val { color:#fff; font-size:28px; font-weight:800; line-height:1.1; }

  .chip {
    display:inline-block; margin-top:12px;
    padding:8px 16px; border-radius:999px;
    font-size:14px; color:#a7f3d0; border:1px solid #16a34a33;
    background: transparent;
  }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HELPERS
# -----------------------------
STAT_LABELS = {"PTS":"Points","REB":"Rebounds","AST":"Assists","STL":"Steals","BLK":"Blocks","FG3M":"3PM"}

def get_player_id(name: str):
    res = players.find_players_by_full_name(name)
    return res[0]["id"] if res else None

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
    # normalize numeric types
    df = df.astype({"PTS":float,"REB":float,"AST":float,"STL":float,"BLK":float,"FG3M":float}, errors="ignore")
    # minutes can be "MM:SS" — keep a numeric helper column
    def to_min(x):
        s = str(x)
        if ":" in s:
            try: return int(s.split(":")[0])
            except: return 0
        try: return int(float(s))
        except: return 0
    df["MIN_NUM"] = df["MIN"].apply(to_min) if "MIN" in df.columns else 0
    return df

def american_to_implied(odds):
    if odds in (None, "", "0", 0): return None
    try: x = float(odds)
    except: return None
    return 100/(x+100) if x>0 else abs(x)/(abs(x)+100)

def prob_to_american(p: float):
    if p<=0 or p>=1: return "N/A"
    return f"{int(round((-100*p/(1-p)) if p>0.5 else (100*(1-p)/p))):+}"

def calc_prob(df: pd.DataFrame, stat: str, thr: int, min_minutes: int, loc_filter: str):
    if df.empty: return 0.0,0,0,df
    d = df.copy()
    d = d[d["MIN_NUM"] >= min_minutes]
    if loc_filter == "Home Only":
        d = d[d["MATCHUP"].str.contains("vs.", regex=False)]
    elif loc_filter == "Away Only":
        d = d[d["MATCHUP"].str.contains("@", regex=False)]
    total = len(d)
    if total==0: return 0.0,0,0,d
    hits = (d[stat] >= thr).sum()
    return hits/total, int(hits), int(total), d

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")
season_opts = ["2024-25","2023-24","2022-23"]
seasons = st.sidebar.multiselect("Seasons", season_opts, default=["2024-25"])
min_minutes = st.sidebar.slider("Minimum Minutes", 0, 40, 20, 1)
location_filter = st.sidebar.selectbox("Game Location", ["All","Home Only","Away Only"])

# -----------------------------
# LEGS
# -----------------------------
if "legs" not in st.session_state:
    st.session_state.legs = [{"player":"","stat":"PTS","thr":10,"odds":-110}]

c1, c2 = st.columns(2)
with c1:
    if st.button("➕ Add Leg"): st.session_state.legs.append({"player":"","stat":"PTS","thr":10,"odds":-110})
with c2:
    if st.button("➖ Remove Leg") and len(st.session_state.legs)>1: st.session_state.legs.pop()

for i, leg in enumerate(st.session_state.legs):
    with st.expander(f"Leg {i+1}", expanded=True):
        leg["player"] = st.text_input(f"Player {i+1}", leg["player"], key=f"p{i}")
        leg["stat"]   = st.selectbox(f"Stat {i+1}", list(STAT_LABELS.keys()),
                                     format_func=lambda k: STAT_LABELS[k], index=list(STAT_LABELS.keys()).index(leg["stat"]),
                                     key=f"s{i}")
        leg["thr"]    = st.number_input(f"Threshold (≥) {i+1}", 0, 100, leg["thr"], key=f"t{i}")
        leg["odds"]   = st.number_input(f"FanDuel Odds {i+1}", -10000, 10000, leg["odds"], step=5, key=f"o{i}")

# Parlay odds only when >1 leg
parlay_odds = 0
if len(st.session_state.legs) > 1:
    st.sidebar.markdown("---")
    parlay_odds = st.sidebar.number_input("Combined Parlay Odds (+300, -150, ...)", value=0, step=5, key="parlay_odds")

# -----------------------------
# COMPUTE
# -----------------------------
if st.button("Compute"):
    st.markdown("---")
    rows = []
    probs_for_parlay = []

    for leg in st.session_state.legs:
        name = leg["player"].strip()
        stat = leg["stat"]; thr = int(leg["thr"]); book = int(leg["odds"])

        pid = get_player_id(name) if name else None
        if not pid:
            rows.append({"ok":False,"name":name or "Unknown","stat":stat,"thr":thr,"book":book})
            continue

        df = fetch_gamelog(pid, seasons)
        p, hits, total, df_filt = calc_prob(df, stat, thr, min_minutes, location_filter)
        fair = prob_to_american(p)
        book_prob = american_to_implied(book)
        ev = None if book_prob is None else (p - book_prob) * 100.0
        if p>0: probs_for_parlay.append(p)

        rows.append({
            "ok":True, "name":name, "stat":stat, "thr":thr, "book":book,
            "p":p, "hits":hits, "total":total, "fair":fair, "book_prob":book_prob,
            "ev":ev, "df":df_filt
        })

    # ---------- Combined Parlay (single HTML block so the background covers everything) ----------
    combined_p = float(np.prod(probs_for_parlay)) if probs_for_parlay else 0.0
    combined_fair = prob_to_american(combined_p) if combined_p>0 else "N/A"
    entered_prob = american_to_implied(parlay_odds)
    parlay_ev = None if entered_prob is None else (combined_p - entered_prob) * 100.0

    cls = "neutral"
    if parlay_ev is not None:
        cls = "pos" if parlay_ev >= 0 else "neg"

    st.markdown(f"""
<div class="card {cls}">
  <h2>💥 Combined Parlay — {', '.join(seasons) if seasons else '—'}</h2>
  <div class="cond">Includes all selected legs and filters</div>
  <div class="row">
    <div class="m"><div class="lab">Model Parlay Probability</div><div class="val">{combined_p*100:.2f}%</div></div>
    <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{combined_fair}</div></div>
    <div class="m"><div class="lab">Entered Parlay Odds</div><div class="val">{parlay_odds if parlay_odds else '—'}</div></div>
    <div class="m"><div class="lab">Book Implied</div><div class="val">{entered_prob*100:.2f}%</div></div>
    <div class="m"><div class="lab">Expected Value</div><div class="val">{'—' if parlay_ev is None else f'{parlay_ev:.2f}%'}</div></div>
  </div>
  <div class="chip">{'🔥 +EV Parlay Detected' if (parlay_ev is not None and parlay_ev>=0) else ('⚠️ Negative EV Parlay' if parlay_ev is not None else 'ℹ️ Enter parlay odds')}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ---------- Individual Legs (single-block HTML + histogram) ----------
    for r in rows:
        if not r.get("ok"):
            st.warning(f"Could not find player: **{r['name']}**")
            continue

        cls = "neutral"
        if r["ev"] is not None:
            cls = "pos" if r["ev"] >= 0 else "neg"

        stat_label = STAT_LABELS[r["stat"]]
        book_implied = "—" if r["book_prob"] is None else f"{r['book_prob']*100:.1f}%"
        ev_disp = "—" if r["ev"] is None else f"{r['ev']:.2f}%"

        # Entire card (title + condition + metrics) in ONE markdown block
        st.markdown(f"""
<div class="card {cls}">
  <h2>{r['name']} — {', '.join(seasons) if seasons else '—'}</h2>
  <div class="cond">Condition: {r['thr']}+ {stat_label.lower()}</div>
  <div class="row">
    <div class="m"><div class="lab">Model Hit Rate</div><div class="val">{r['p']*100:.1f}% ({r['hits']}/{r['total']})</div></div>
    <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{r['fair']}</div></div>
    <div class="m"><div class="lab">FanDuel Odds</div><div class="val">{r['book']}</div></div>
    <div class="m"><div class="lab">Book Implied</div><div class="val">{book_implied}</div></div>
    <div class="m"><div class="lab">Expected Value</div><div class="val">{ev_disp}</div></div>
  </div>
  <div class="chip">{'🔥 +EV Play Detected (by your model)' if (r['ev'] is not None and r['ev']>=0) else ('⚠️ Negative EV Play' if r['ev'] is not None else 'ℹ️ Add odds to compute EV')}</div>
</div>
""", unsafe_allow_html=True)

        # Histogram under the card
        df_f = r["df"]
        if not df_f.empty and r["stat"] in df_f.columns:
            fig, ax = plt.subplots()
            ax.hist(df_f[r["stat"]], bins=20, edgecolor="black",
                    color="#00c896" if (r["ev"] is not None and r["ev"]>=0) else "#e05a5a")
            ax.axvline(r["thr"], color="red", linestyle="--", label=f"Threshold {r['thr']}")
            ax.set_title(f"{r['name']} — {stat_label}")
            ax.set_xlabel(stat_label); ax.set_ylabel("Games"); ax.legend()
            st.pyplot(fig)
