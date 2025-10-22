import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NBA Parlay Builder", page_icon="🏀", layout="wide")
st.title("🏀 NBA Parlay Builder — GOAT++")

# =========================
# THEME TOGGLE (Soft Neutral Light)
# =========================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default

with st.sidebar:
    st.subheader("🎨 Theme")
    dark_mode = st.toggle("Dark mode", value=(st.session_state.theme == "dark"))
    st.session_state.theme = "dark" if dark_mode else "light"

is_dark = st.session_state.theme == "dark"

# CSS variables for themes
if is_dark:
    css_vars = """
    :root {
      --bg: #111214;
      --text: #ffffff;
      --muted: #a9b1bb;
      --card: #222;
      --border: #777;
      --pos-bg: #0b3d23; --pos-border: #00FF99;
      --neg-bg: #3d0b0b; --neg-border: #FF5555;
      --chip-border: #16a34a33; --chip-text: #a7f3d0;
    }
    """
else:
    # Soft neutral light palette (not blinding white)
    css_vars = """
    :root {
      --bg: #f4f6f8;
      --text: #0f172a;
      --muted: #475569;
      --card: #ffffff;
      --border: #d1d5db;
      --pos-bg: #e9f8f0; --pos-border: #22c55e;
      --neg-bg: #fdecec; --neg-border: #ef4444;
      --chip-border: #9ca3af77; --chip-text: #0f172a;
    }
    """

st.markdown(f"""
<style>
  {css_vars}

  body, .block-container {{
    background: var(--bg);
    color: var(--text);
  }}

  .card {{
    --pad-x: 28px; --pad-y: 22px;
    padding: var(--pad-y) var(--pad-x);
    border-radius: 18px;
    margin: 14px 0 26px 0;
    border: 1px solid var(--border);
    background: var(--card);
    box-shadow: 0 0 14px rgba(0,0,0,0.12);
    width: 100%;
  }}
  .neutral {{ background: var(--card); border-color: var(--border); }}
  .pos     {{ background: var(--pos-bg); border-color: var(--pos-border); }}
  .neg     {{ background: var(--neg-bg); border-color: var(--neg-border); }}

  .card h2 {{ color: var(--text); margin:0 0 6px 0; font-weight:800; }}
  .cond {{ color: var(--muted); font-size:15px; margin: 2px 0 12px 0; }}

  /* single real row of metrics, wraps if narrow */
  .row {{
    display:flex; flex-wrap:wrap; gap:16px;
    align-items:flex-end; justify-content:space-between;
    margin: 8px 0 6px 0;
  }}
  .m {{ min-width:140px; flex:1; }}
  .lab {{ color: var(--muted); font-size:14px; margin-bottom:4px; }}
  .val {{ color: var(--text); font-size:28px; font-weight:800; line-height:1.1; }}

  .chip {{
    display:inline-block; margin-top:12px;
    padding:8px 16px; border-radius:999px;
    font-size:14px; color: var(--chip-text); border:1px solid var(--chip-border);
    background: transparent;
  }}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
STAT_LABELS = {"PTS":"Points","REB":"Rebounds","AST":"Assists","STL":"Steals","BLK":"Blocks","FG3M":"3PM"}

def get_player_id(name: str):
    res = players.find_players_by_full_name(name)
    return res[0]["id"] if res else None

def to_minutes(val):
    try:
        s = str(val)
        if ":" in s:
            return int(s.split(":")[0])
        return int(float(s))
    except:
        return 0

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
    # numeric types
    for k in ["PTS","REB","AST","STL","BLK","FG3M"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    # minutes numeric
    if "MIN" in df.columns:
        df["MIN_NUM"] = df["MIN"].apply(to_minutes)
    else:
        df["MIN_NUM"] = 0
    # parse game date
    if "GAME_DATE" in df.columns:
        try:
            df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"])
        except:
            # fallback if e.g. "OCT 21, 2024"
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

def calc_prob(df: pd.DataFrame, stat: str, thr: int, min_minutes: int, loc_filter: str, range_key: str):
    """
    loc_filter: "All", "Home Only", "Away Only"
    range_key: "L10", "L20", "FULL"
    """
    if df.empty: return 0.0,0,0,df
    d = df.copy()
    # minutes
    d = d[d["MIN_NUM"] >= min_minutes]
    # location
    if loc_filter == "Home Only":
        d = d[d["MATCHUP"].astype(str).str.contains("vs", regex=False) | d["MATCHUP"].astype(str).str.contains("vs.", regex=False)]
    elif loc_filter == "Away Only":
        d = d[d["MATCHUP"].astype(str).str.contains("@", regex=False)]
    # sort by date desc & slice range
    d = d.sort_values("GAME_DATE_DT", ascending=False)
    if range_key == "L10":
        d = d.head(10)
    elif range_key == "L20":
        d = d.head(20)
    # probability
    total = len(d)
    if total==0: return 0.0,0,0,d
    if stat not in d.columns:
        return 0.0,0,total,d
    hits = (d[stat] >= thr).sum()
    return hits/total, int(hits), int(total), d

# =========================
# SIDEBAR — GLOBAL FILTERS + QUICK PICKS
# =========================
with st.sidebar:
    st.subheader("⚙️ Filters")
    season_opts = ["2024-25","2023-24","2022-23"]
    seasons = st.multiselect("Seasons", season_opts, default=["2024-25"])
    min_minutes = st.slider("Minimum Minutes", 0, 40, 20, 1)

    st.markdown("---")
    st.subheader("🎯 Quick Picks")
    qp = st.selectbox(
        "Load a star combo (optional)",
        [
            "—",
            "Stephen Curry — 25+ PTS & 4+ 3PM",
            "Nikola Jokic — 10+ REB & 10+ AST",
            "Jayson Tatum — 25+ PTS & 8+ REB",
            "Luka Doncic — 30+ PTS & 8+ AST",
            "Shai Gilgeous-Alexander — 25+ PTS & 5+ AST",
            "Giannis Antetokounmpo — 30+ PTS & 10+ REB",
        ],
        index=0,
    )

# =========================
# LEGS STATE
# =========================
if "legs" not in st.session_state:
    st.session_state.legs = [{"player":"","stat":"PTS","thr":10,"odds":-110,"loc":"All","range":"L20"}]

# Apply Quick Pick (overwrites current legs)
def apply_quick_pick(choice: str):
    mapping = {
        "Stephen Curry — 25+ PTS & 4+ 3PM": [
            {"player":"Stephen Curry","stat":"PTS","thr":25,"odds":-110,"loc":"All","range":"L20"},
            {"player":"Stephen Curry","stat":"FG3M","thr":4,"odds":-110,"loc":"All","range":"L20"},
        ],
        "Nikola Jokic — 10+ REB & 10+ AST": [
            {"player":"Nikola Jokic","stat":"REB","thr":10,"odds":-110,"loc":"All","range":"L20"},
            {"player":"Nikola Jokic","stat":"AST","thr":10,"odds":-110,"loc":"All","range":"L20"},
        ],
        "Jayson Tatum — 25+ PTS & 8+ REB": [
            {"player":"Jayson Tatum","stat":"PTS","thr":25,"odds":-110,"loc":"All","range":"L20"},
            {"player":"Jayson Tatum","stat":"REB","thr":8,"odds":-110,"loc":"All","range":"L20"},
        ],
        "Luka Doncic — 30+ PTS & 8+ AST": [
            {"player":"Luka Doncic","stat":"PTS","thr":30,"odds":-110,"loc":"All","range":"L20"},
            {"player":"Luka Doncic","stat":"AST","thr":8,"odds":-110,"loc":"All","range":"L20"},
        ],
        "Shai Gilgeous-Alexander — 25+ PTS & 5+ AST": [
            {"player":"Shai Gilgeous-Alexander","stat":"PTS","thr":25,"odds":-110,"loc":"All","range":"L20"},
            {"player":"Shai Gilgeous-Alexander","stat":"AST","thr":5,"odds":-110,"loc":"All","range":"L20"},
        ],
        "Giannis Antetokounmpo — 30+ PTS & 10+ REB": [
            {"player":"Giannis Antetokounmpo","stat":"PTS","thr":30,"odds":-110,"loc":"All","range":"L20"},
            {"player":"Giannis Antetokounmpo","stat":"REB","thr":10,"odds":-110,"loc":"All","range":"L20"},
        ],
    }
    if choice in mapping:
        st.session_state.legs = mapping[choice]

if qp != "—":
    apply_quick_pick(qp)

# =========================
# LEG MANAGEMENT
# =========================
c1, c2 = st.columns(2)
with c1:
    if st.button("➕ Add Leg"):
        st.session_state.legs.append({"player":"","stat":"PTS","thr":10,"odds":-110,"loc":"All","range":"L20"})
with c2:
    if st.button("➖ Remove Leg") and len(st.session_state.legs)>1:
        st.session_state.legs.pop()

for i, leg in enumerate(st.session_state.legs):
    with st.expander(f"Leg {i+1}", expanded=True):
        leg["player"] = st.text_input(f"Player {i+1}", leg["player"], key=f"p{i}")
        leg["stat"]   = st.selectbox(f"Stat {i+1}", list(STAT_LABELS.keys()),
                                     format_func=lambda k: STAT_LABELS[k],
                                     index=list(STAT_LABELS.keys()).index(leg["stat"]) if leg["stat"] in STAT_LABELS else 0,
                                     key=f"s{i}")
        leg["thr"]    = st.number_input(f"Threshold (≥) {i+1}", 0, 100, leg["thr"], key=f"t{i}")
        leg["odds"]   = st.number_input(f"FanDuel Odds {i+1}", -10000, 10000, leg["odds"], step=5, key=f"o{i}")
        leg["loc"]    = st.selectbox(f"Home/Away {i+1}", ["All","Home Only","Away Only"],
                                     index=["All","Home Only","Away Only"].index(leg["loc"]) if leg.get("loc") else 0,
                                     key=f"l{i}")
        leg["range"]  = st.selectbox(f"Game Range {i+1}", ["L10","L20","FULL"],
                                     index=["L10","L20","FULL"].index(leg["range"]) if leg.get("range") else 1,
                                     key=f"r{i}")

# Parlay odds only when >1 leg
parlay_odds = 0
if len(st.session_state.legs) > 1:
    st.sidebar.markdown("---")
    parlay_odds = st.sidebar.number_input("Combined Parlay Odds (+300, -150, ...)", value=0, step=5, key="parlay_odds")

# =========================
# COMPUTE
# =========================
if st.button("Compute"):
    st.markdown("---")

    rows = []
    probs_for_parlay = []

    # Matplotlib theming
    plt.rcParams.update({
        "axes.facecolor": ("#1e1f22" if is_dark else "#ffffff"),
        "figure.facecolor": ("#1e1f22" if is_dark else "#ffffff"),
        "text.color": ("#ffffff" if is_dark else "#0f172a"),
        "axes.labelcolor": ("#e5e7eb" if is_dark else "#0f172a"),
        "xtick.color": ("#e5e7eb" if is_dark else "#0f172a"),
        "ytick.color": ("#e5e7eb" if is_dark else "#0f172a"),
        "grid.color": ("#374151" if is_dark else "#e5e7eb"),
    })

    for leg in st.session_state.legs:
        name = leg["player"].strip()
        stat = leg["stat"]; thr = int(leg["thr"]); book = int(leg["odds"])
        loc_key = leg.get("loc","All")
        range_key = leg.get("range","L20")

        pid = get_player_id(name) if name else None
        if not pid:
            rows.append({"ok":False,"name":name or "Unknown","stat":stat,"thr":thr,"book":book})
            continue

        df = fetch_gamelog(pid, st.session_state.get("seasons", []) or seasons)
        p, hits, total, df_filt = calc_prob(df, stat, thr, min_minutes, loc_key, range_key)
        fair = prob_to_american(p)
        book_prob = american_to_implied(book)
        ev = None if book_prob is None else (p - book_prob) * 100.0
        if p>0: probs_for_parlay.append(p)

        rows.append({
            "ok":True, "name":name, "stat":stat, "thr":thr, "book":book,
            "p":p, "hits":hits, "total":total, "fair":fair, "book_prob":book_prob,
            "ev":ev, "df":df_filt, "loc":loc_key, "range":range_key
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
  <h2>💥 Combined Parlay — {', '.join(seasons) if seasons else '—'}</h2>
  <div class="cond">Includes all selected legs with their individual filters</div>
  <div class="row">
    <div class="m"><div class="lab">Model Parlay Probability</div><div class="val">{combined_p*100:.2f}%</div></div>
    <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{combined_fair}</div></div>
    <div class="m"><div class="lab">Entered Parlay Odds</div><div class="val">{parlay_odds if parlay_odds else '—'}</div></div>
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
        cond_bits = [f"{r['thr']}+ {stat_label.lower()}"]
        if r.get("loc") and r["loc"] != "All":
            cond_bits.append(r["loc"].replace(" Only","").lower())
        if r.get("range") == "L10":
            cond_bits.append("last 10")
        elif r.get("range") == "L20":
            cond_bits.append("last 20")
        else:
            cond_bits.append("full season")
        cond_text = " — ".join(cond_bits)

        st.markdown(f"""
<div class="card {cls}">
  <h2>{r['name']} — {', '.join(seasons) if seasons else '—'}</h2>
  <div class="cond">Condition: {cond_text}</div>
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
            # bar color echoes EV signal; background echoes theme
            ax.hist(df_f[r["stat"]], bins=20, edgecolor=("white" if is_dark else "black"),
                    color=("#00c896" if (r["ev"] is not None and r["ev"]>=0) else "#e05a5a"))
            ax.axvline(r["thr"], color=("w" if is_dark else "red"), linestyle="--", label=f"Threshold {r['thr']}")
            ax.set_title(f"{r['name']} — {stat_label}")
            ax.set_xlabel(stat_label); ax.set_ylabel("Games"); ax.legend()
            st.pyplot(fig)
