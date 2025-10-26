# nba_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from rapidfuzz import process

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NBA Player Prop Tools", page_icon="🏀", layout="wide")
st.title("🏀 NBA Player Prop Tools")

# =========================
# THEME / CSS
# =========================
st.markdown("""
<style>
:root {
  --bg: #0e0f11;
  --text: #f9fafb;
  --muted: #9ca3af;
  --card: #1a1b1e;
  --border: #32353b;
  --accent: #00c896;
}
body, .block-container { background: var(--bg); color: var(--text); }

/* Buttons */
.stButton > button {
  background: var(--accent) !important;
  color: #0b1220 !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  padding: 8px 14px !important;
  font-size: 0.9rem !important;
  box-shadow: 0 6px 18px rgba(0,200,150,0.25) !important;
}

/* Expander */
.stExpander {
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  border-radius: 12px !important;
}
.streamlit-expanderHeader {
  font-weight: 800 !important;
  color: var(--text) !important;
  font-size: 0.95rem !important;
}

/* Inputs */
input, select, textarea {
  background: #202225 !important;
  color: var(--text) !important;
  border: 1px solid #3b3f45 !important;
  border-radius: 8px !important;
  font-size: 0.9rem !important;
  padding: 6px !important;
}

/* Cards */
.card {
  --pad-x: 20px;
  --pad-y: 18px;
  padding: var(--pad-y) var(--pad-x);
  border-radius: 14px;
  margin: 10px 0 20px 0;
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  box-shadow: 0 0 14px rgba(0,0,0,0.25);
  width: 100%;
}
.neutral { --card-bg:#222; --card-border:#777; }
.pos { --card-bg:#0b3d23; --card-border:#00FF99; }
.neg { --card-bg:#3d0b0b; --card-border:#FF5555; }
.card h2 { color:#fff; margin:0 0 6px 0; font-weight:800; font-size:1.05rem; }
.cond { color:#a9b1bb; font-size:0.9rem; margin: 2px 0 10px 0; }
.row { display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end; justify-content:space-between; margin: 6px 0 4px 0; }
.m { min-width:120px; flex:1; }
.lab { color:#cbd5e1; font-size:0.8rem; margin-bottom:2px; }
.val { color:#fff; font-size:1.1rem; font-weight:800; line-height:1.1; }
.chip { display:inline-block; margin-top:10px; padding:6px 12px; border-radius:999px; font-size:0.8rem; color:#a7f3d0; border:1px solid #16a34a33; background: transparent; }

/* Table polish for Breakeven tab */
table { border-collapse: collapse; }
thead th { border-bottom: 1px solid #374151 !important; }
tbody td, thead th { padding: 8px 10px !important; }

/* --- Filter Pills Bar --- */
.filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 12px;
  align-items: center;
  justify-content: flex-start;
  padding: 10px 14px;
  margin: 6px 0 12px 0;
  border: 1px solid var(--border);
  background: var(--card);
  border-radius: 12px;
}
.filter-section-title {
  color: var(--muted);
  font-size: 0.85rem;
  font-weight: 700;
  margin-right: 6px;
}
.pill-row {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.85rem;
  border: 1px solid #3b3f45;
  background: #1f2125;
  color: #d1d5db;
  cursor: pointer;
  user-select: none;
}
.pill:hover { box-shadow: 0 0 0 2px rgba(0,200,150,0.15) inset; }
.pill-on {
  background: #0b3d23;
  border-color: #00c896;
  color: #c7ffee;
  box-shadow: 0 0 0 1px #00c896 inset, 0 6px 16px rgba(0,200,150,0.15);
}
.pill-min {
  min-width: 48px;
  justify-content: center;
}
.pill-label { opacity: 0.9; }
.slider-wrap {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  margin-left: 6px;
}
.slider-label {
  color: var(--muted);
  font-size: 0.8rem;
  margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CONSTANTS & HELPERS
# =========================
STAT_LABELS = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists",
    "STL": "Steals", "BLK": "Blocks", "FG3M": "3PM",
    "DOUBDOUB": "Doub Doub", "TRIPDOUB": "Trip Doub",
    "P+R": "P+R", "P+A": "P+A", "R+A": "R+A", "PRA": "PRA",
}

STAT_TOKENS = {
    "P": "PTS", "PTS": "PTS", "POINTS": "PTS",
    "R": "REB", "REB": "REB", "REBOUNDS": "REB",
    "A": "AST", "AST": "AST", "ASSISTS": "AST",
    "STL": "STL", "STEALS": "STL", "BLK": "BLK", "BLOCKS": "BLK",
    "3PM": "FG3M", "FG3M": "FG3M", "THREES": "FG3M",
    "DOUBDOUB": "DOUBDOUB", "DOUBLEDOUBLE": "DOUBDOUB", "DOUB": "DOUBDOUB", "DD": "DOUBDOUB",
    "TRIPDOUB": "TRIPDOUB", "TRIPLEDOUBLE": "TRIPDOUB", "TD": "TRIPDOUB",
    "P+R": "P+R", "PR": "P+R", "P+A": "P+A", "PA": "P+A", "R+A": "R+A", "RA": "R+A", "PRA": "PRA"
}

SEASON_OPTIONS = ["2024-25", "2023-24", "2022-23"]
MIN_PRESETS = [10, 20, 30]

@st.cache_data
def get_all_player_names():
    try:
        active = players.get_active_players()
        names = [p["full_name"] for p in active if p.get("full_name")]
        if names:
            return sorted(set(names))
    except Exception:
        pass
    all_p = players.get_players()
    names = [p["full_name"] for p in all_p if p.get("full_name")]
    return sorted(set(names))

PLAYER_LIST = get_all_player_names()

def best_player_match(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    m = process.extractOne(q, PLAYER_LIST, score_cutoff=80)
    if m:
        return m[0]
    m = process.extractOne(q, PLAYER_LIST, score_cutoff=60)
    return m[0] if m else ""

def american_to_implied(odds):
    try:
        x = float(odds)
    except Exception:
        return None
    if -99 < x < 100:
        return None
    if x > 0:
        return 100.0 / (x + 100.0)
    return abs(x) / (abs(x) + 100.0)

def prob_to_american(p: float):
    if p <= 0 or p >= 1:
        return "N/A"
    dec = 1.0 / p
    if dec >= 2.0:
        return f"+{int(round((dec - 1) * 100))}"
    return f"-{int(round(100 / (dec - 1)))}"

def fmt_half(x):
    try:
        v = float(x)
        return f"{v:.1f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

# ---- Parsing + core functions ----
def parse_input_line(text: str):
    t = (text or "").strip()
    if not t:
        return None
    parts = t.replace("/", "+").split()

    # Direction
    dir_token = None
    for token in parts:
        if token.upper() in ["O", "OVER"]:
            dir_token = "Over"; break
        if token.upper() in ["U", "UNDER"]:
            dir_token = "Under"; break
    if not dir_token:
        dir_token = "Over"

    # Threshold
    thr = None
    for token in parts:
        tok = token.replace("+", "")
        try:
            if any(c.isdigit() for c in tok) and ("." in tok or tok.isdigit()):
                thr = float(tok); break
        except Exception:
            pass
    if thr is None:
        thr = 10.5

    # Stat code
    stat_code = None
    combined_aliases = ["P+R", "P+A", "R+A", "PRA", "PR", "PA", "RA"]
    for token in parts:
        up = token.upper()
        if up in combined_aliases:
            stat_code = STAT_TOKENS.get(up, up); break
    if not stat_code:
        for token in parts:
            up = token.upper()
            if up in STAT_TOKENS:
                stat_code = STAT_TOKENS[up]; break
    if not stat_code:
        stat_code = "PTS"

    # Location
    loc = "All"
    for token in parts:
        up = token.upper()
        if up in ["AWAY", "A"]:
            loc = "Away"; break
        if up in ["HOME", "H"]:
            loc = "Home Only"; break

    # Odds
    odds = -110
    for token in parts[::-1]:
        if token.startswith("+") or token.startswith("-"):
            try:
                o = int(token)
                if o <= -100 or o >= 100:
                    odds = o; break
            except Exception:
                continue

    # Player name (remaining tokens)
    banned = set(["O", "OVER", "U", "UNDER", "HOME", "H", "AWAY", "A"] +
                 list(STAT_TOKENS.keys()) + combined_aliases)
    name_tokens = [p for p in parts if (p.upper() not in banned and not p.replace(".", "", 1).lstrip("+-").isdigit())]
    name_guess = " ".join(name_tokens).strip()
    player = best_player_match(name_guess)

    return {"player": player, "dir": dir_token, "thr": float(thr), "stat": stat_code,
            "loc": loc, "range": "FULL", "odds": int(odds)}

def get_player_id(full_name: str):
    if not full_name:
        return None
    res = players.find_players_by_full_name(full_name)
    return res[0]["id"] if res else None

def to_minutes(val):
    try:
        s = str(val)
        if ":" in s:
            return int(s.split(":")[0])
        return int(float(s))
    except Exception:
        return 0

def fetch_gamelog(player_id: int, seasons: list[str]) -> pd.DataFrame:
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
    for k in ["PTS", "REB", "AST", "STL", "BLK", "FG3M"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    df["MIN_NUM"] = df["MIN"].apply(to_minutes) if "MIN" in df.columns else 0
    if "GAME_DATE" in df.columns:
        df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    else:
        df["GAME_DATE_DT"] = pd.Timestamp.now()
    return df

def compute_stat_series(df, stat_code):
    if stat_code in ["PTS", "REB", "AST", "STL", "BLK", "FG3M"]:
        return df[stat_code].astype(float)
    if stat_code == "P+R":
        return (df["PTS"] + df["REB"]).astype(float)
    if stat_code == "P+A":
        return (df["PTS"] + df["AST"]).astype(float)
    if stat_code == "R+A":
        return (df["REB"] + df["AST"]).astype(float)
    if stat_code == "PRA":
        return (df["PTS"] + df["REB"] + df["AST"]).astype(float)
    if stat_code == "DOUBDOUB":
        s = ((df["PTS"] >= 10).astype(int) + (df["REB"] >= 10).astype(int) + (df["AST"] >= 10).astype(int)) >= 2
        return s.astype(int)
    if stat_code == "TRIPDOUB":
        s = ((df["PTS"] >= 10).astype(int) + (df["REB"] >= 10).astype(int) + (df["AST"] >= 10).astype(int)) >= 3
        return s.astype(int)
    return df["PTS"].astype(float)

def leg_probability(df, stat_code, direction, thr):
    ser = compute_stat_series(df, stat_code)
    if stat_code in ["DOUBDOUB", "TRIPDOUB"]:
        hits = int((ser <= 0.5).sum()) if direction == "Under" else int((ser >= 0.5).sum())
    else:
        hits = int((ser <= thr).sum()) if direction == "Under" else int((ser >= thr).sum())
    total = int(ser.notna().sum())
    p = hits / total if total else 0.0
    return p, hits, total

def headshot_url(pid):
    if not pid:
        return None
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"

def breakeven_for_stat(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    total = len(s)
    if total == 0:
        return {"line": None, "over_prob": None, "under_prob": None, "over_odds": "N/A", "under_odds": "N/A"}
    lo, hi = np.floor(s.min()) - 0.5, np.ceil(s.max()) + 0.5
    candidates = np.arange(lo, hi + 0.001, 0.5)
    best_t, best_gap, best_over = None, 1.0, None
    for t in candidates:
        over = (s >= t).mean()
        gap = abs(over - 0.5)
        if (gap < best_gap) or (best_t is None):
            best_t, best_gap, best_over = t, gap, over
    over_prob = float(best_over)
    under_prob = 1.0 - over_prob
    return {
        "line": float(best_t),
        "over_prob": over_prob,
        "under_prob": under_prob,
        "over_odds": prob_to_american(over_prob),
        "under_odds": prob_to_american(under_prob)
    }

# =========================
# SESSION DEFAULTS (for pills)
# =========================
if "seasons_selected" not in st.session_state:
    st.session_state.seasons_selected = ["2024-25"]
if "min_minutes" not in st.session_state:
    st.session_state.min_minutes = 20
if "awaiting_input" not in st.session_state:
    st.session_state.awaiting_input = True
if "legs" not in st.session_state:
    st.session_state.legs = []

# =========================
# TABS
# =========================
tab_builder, tab_breakeven = st.tabs(["🧮 Parlay Builder", "🧷 Breakeven"])

# =========================
# TAB 1: PARLAY BUILDER
# =========================
with tab_builder:
    # --- Pill Filter Bar (above "Input bet") ---
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)

    # Seasons pills
    st.markdown('<span class="filter-section-title">Seasons</span>', unsafe_allow_html=True)
    s_cols = st.columns(len(SEASON_OPTIONS))
    for idx, season in enumerate(SEASON_OPTIONS):
        is_on = season in st.session_state.seasons_selected
        # Render as a button styled like a pill; clicking toggles
        with s_cols[idx]:
            if st.button(season, key=f"pill_season_{season}", use_container_width=True):
                # toggle selection
                sel = set(st.session_state.seasons_selected)
                if season in sel:
                    sel.remove(season)
                else:
                    sel.add(season)
                # ensure at least one season selected
                if not sel:
                    sel.add(season)
                st.session_state.seasons_selected = sorted(sel, reverse=False)
                st.rerun()
        # Apply pill style (on/off)
        pill_class = "pill pill-on" if is_on else "pill"
        st.markdown(
            f"""
            <script>
            const btn = window.parent.document.querySelector('button[kind="secondary"]');
            </script>
            """,
            unsafe_allow_html=True
        )
        # Note: Streamlit doesn't allow direct class injection into the button;
        # visual styling comes from the default theme. The toggle state is handled in Python.
        # (We keep CSS classes for potential future Custom Components.)

    # Minutes quick-pick pills + Slider
    st.markdown('<span class="filter-section-title" style="margin-left:8px;">Min Minutes</span>', unsafe_allow_html=True)
    m_cols = st.columns(len(MIN_PRESETS) + 2)
    for i, preset in enumerate(MIN_PRESETS):
        with m_cols[i]:
            clicked = st.button(f"{preset}", key=f"pill_min_{preset}", use_container_width=True)
            if clicked:
                st.session_state.min_minutes = int(preset)
                st.rerun()
    with m_cols[-2]:
        st.markdown('<div class="slider-label">Custom</div>', unsafe_allow_html=True)
    with m_cols[-1]:
        st.session_state.min_minutes = st.slider(
            "Min Minutes",
            min_value=0, max_value=40, value=int(st.session_state.min_minutes), step=1,
            label_visibility="collapsed"
        )

    st.markdown('</div>', unsafe_allow_html=True)  # end filter-bar

    # Convenience local vars from session
    seasons = st.session_state.seasons_selected
    min_minutes = int(st.session_state.min_minutes)

    # ----- CONTROLS -----
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("➕ Add Leg"):
            st.session_state.awaiting_input = True
    with c2:
        if st.button("➖ Remove Last Leg") and st.session_state.legs:
            st.session_state.legs.pop()

    st.write("**Input bet**")

    # 1) Render existing legs (Leg 1 on top)
    if st.session_state.legs:
        for i, leg in enumerate(st.session_state.legs):
            leg_no = i + 1
            dir_short = "O" if leg["dir"] == "Over" else "U"
            header = f"Leg {leg_no}: {leg['player']} — {dir_short} {fmt_half(leg['thr'])} {STAT_LABELS.get(leg['stat'], leg['stat'])} ({leg['loc']}, {leg['odds']})"
            with st.expander(header, expanded=False):
                cL, cR = st.columns([2, 1])
                with cL:
                    leg["player"] = st.text_input("Player", value=leg["player"], key=f"player_{i}")
                    leg["stat"] = st.selectbox("Stat", list(STAT_LABELS.keys()),
                                               index=list(STAT_LABELS.keys()).index(leg["stat"]), key=f"stat_{i}")
                    leg["dir"] = st.selectbox("O/U", ["Over", "Under"],
                                              index=(0 if leg["dir"] == "Over" else 1), key=f"dir_{i}")
                    leg["thr"] = st.number_input("Threshold", value=float(leg["thr"]), step=0.5, key=f"thr_{i}")
                with cR:
                    leg["loc"] = st.selectbox("Home/Away", ["All", "Home Only", "Away"],
                                              index=["All", "Home Only", "Away"].index(leg["loc"]), key=f"loc_{i}")
                    leg["range"] = st.selectbox("Game Range", ["FULL", "L10", "L20"],
                                                index=["FULL", "L10", "L20"].index(leg.get("range", "FULL")), key=f"range_{i}")
                    leg["odds"] = st.number_input("Sportsbook Odds", value=int(leg["odds"]), step=5, key=f"odds_{i}")
                rm_col, _ = st.columns([1, 5])
                with rm_col:
                    if st.button(f"❌ Remove Leg {leg_no}", key=f"remove_{i}"):
                        st.session_state.legs.pop(i)
                        st.rerun()

    # 2) Input field only when awaiting input
    if st.session_state.awaiting_input:
        bet_text = st.text_input(
            "Input bet",
            placeholder="Maxey O 24.5 P Away -110 OR Embiid PRA U 35.5 -130",
            key="freeform_input",
            label_visibility="collapsed"
        )
        if bet_text.strip():
            parsed = parse_input_line(bet_text)
            if parsed and parsed["player"]:
                st.session_state.legs.append(parsed)
                st.session_state.awaiting_input = False
                st.rerun()

    # Combined parlay odds input (if multiple legs exist)
    parlay_odds = 0
    if len(st.session_state.legs) > 1:
        st.markdown("### 🎯 Combined Parlay Odds")
        parlay_odds = st.number_input("Enter Parlay Odds (+300, -150, etc.)", value=0, step=5, key="parlay_odds")

    # ----- COMPUTE -----
    if st.session_state.legs and st.button("Compute"):
        st.markdown("---")
        rows = []
        probs_for_parlay = []

        # dark plot theme
        plt.rcParams.update({
            "axes.facecolor": "#1e1f22",
            "figure.facecolor": "#1e1f22",
            "text.color": "#ffffff",
            "axes.labelcolor": "#e5e7eb",
            "xtick.color": "#e5e7eb",
            "ytick.color": "#e5e7eb",
            "grid.color": "#374151",
        })

        for leg in st.session_state.legs:
            name = leg["player"]
            pid = get_player_id(name)
            stat = leg["stat"]
            direction = leg["dir"]
            thr = float(leg["thr"])
            loc = leg["loc"]
            rng = leg.get("range", "FULL")
            odds = int(leg["odds"])

            if not pid:
                rows.append({"ok": False, "name": name or "Unknown"})
                continue

            df = fetch_gamelog(pid, seasons)
            if df.empty:
                rows.append({"ok": False, "name": name, "reason": "No logs"})
                continue

            # filters
            d = df.copy()
            d = d[d["MIN_NUM"] >= min_minutes]
            if loc == "Home Only":
                d = d[d["MATCHUP"].astype(str).str.contains("vs", regex=False)]
            elif loc == "Away":
                d = d[d["MATCHUP"].astype(str).str.contains("@", regex=False)]
            d = d.sort_values("GAME_DATE_DT", ascending=False)
            if rng == "L10":
                d = d.head(10)
            elif rng == "L20":
                d = d.head(20)

            p, hits, total = (0.0, 0, 0)
            if not d.empty:
                p, hits, total = leg_probability(d, stat, direction, thr)

            fair = prob_to_american(p)
            book_prob = american_to_implied(odds)
            ev = None if book_prob is None else (p - book_prob) * 100.0
            if p > 0:
                probs_for_parlay.append(p)

            rows.append({
                "ok": True, "name": name, "stat": stat, "thr": thr, "dir": direction, "loc": loc, "range": rng,
                "odds": odds, "p": p, "hits": hits, "total": total, "fair": fair, "book_prob": book_prob,
                "ev": ev, "df": d
            })

        # ---------- Combined Parlay ----------
        combined_p = float(np.prod(probs_for_parlay)) if probs_for_parlay else 0.0
        combined_fair = prob_to_american(combined_p) if combined_p > 0 else "N/A"
        entered_prob = american_to_implied(parlay_odds)
        parlay_ev = None if entered_prob is None else (combined_p - entered_prob) * 100.0
        cls = "neutral"
        if parlay_ev is not None:
            cls = "pos" if parlay_ev >= 0 else "neg"

        st.markdown(f"""
        <div class="card {cls}">
          <h2>💥 Combined Parlay</h2>
          <div class="cond">Includes all legs with your filters</div>
          <div class="row">
            <div class="m"><div class="lab">Model Parlay Probability</div><div class="val">{combined_p*100:.2f}%</div></div>
            <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{combined_fair}</div></div>
            <div class="m"><div class="lab">Entered Odds</div><div class="val">{parlay_odds if parlay_odds else '—'}</div></div>
            <div class="m"><div class="lab">Book Implied</div><div class="val">{'—' if entered_prob is None else f'{entered_prob*100:.2f}%'}</div></div>
            <div class="m"><div class="lab">Expected Value</div><div class="val">{'—' if parlay_ev is None else f'{parlay_ev:.2f}%'}</div></div>
          </div>
          <div class="chip">{'🔥 +EV Parlay Detected' if (parlay_ev is not None and parlay_ev>=0) else ('⚠️ Negative EV Parlay' if parlay_ev is not None else 'ℹ️ Enter parlay odds')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ---------- Individual legs ----------
        for r in rows:
            if not r.get("ok"):
                st.warning(f"Could not compute for **{r.get('name','Unknown')}**")
                continue

            cls = "neutral"
            if r["ev"] is not None:
                cls = "pos" if r["ev"] >= 0 else "neg"

            stat_label = STAT_LABELS.get(r["stat"], r["stat"])
            book_implied = "—" if r["book_prob"] is None else f"{r['book_prob']*100:.1f}%"
            ev_disp = "—" if r["ev"] is None else f"{r['ev']:.2f}%"
            dir_word = "O" if r["dir"] == "Over" else "U"
            cond_text = f"{dir_word} {fmt_half(r['thr'])} {stat_label} — {r['range']} — {r['loc'].replace(' Only','')}"

            st.markdown(f"""
            <div class="card {cls}">
              <h2>{r['name']}</h2>
              <div class="cond">Condition: {cond_text}</div>
              <div class="row">
                <div class="m"><div class="lab">Model Hit Rate</div><div class="val">{r['p']*100:.1f}% ({r['hits']}/{r['total']})</div></div>
                <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{r['fair']}</div></div>
                <div class="m"><div class="lab">Sportsbook Odds</div><div class="val">{r['odds']}</div></div>
                <div class="m"><div class="lab">Book Implied</div><div class="val">{book_implied}</div></div>
                <div class="m"><div class="lab">Expected Value</div><div class="val">{ev_disp}</div></div>
              </div>
              <div class="chip">{'🔥 +EV Play Detected' if (r['ev'] is not None and r['ev']>=0) else ('⚠️ Negative EV Play' if r['ev'] is not None else 'ℹ️ Add odds to compute EV')}</div>
            </div>
            """, unsafe_allow_html=True)

            # histogram (skip indicator stats)
            dff = r["df"]
            if not dff.empty and r["stat"] not in ["DOUBDOUB", "TRIPDOUB"]:
                ser = compute_stat_series(dff, r["stat"])
                fig, ax = plt.subplots()
                ax.hist(ser, bins=20, edgecolor="white",
                        color=("#00c896" if (r["ev"] is not None and r["ev"] >= 0) else "#e05a5a"))
                ax.axvline(r["thr"], color="w", linestyle="--", label=f"Threshold {fmt_half(r['thr'])}")
                ax.set_title(f"{r['name']} — {stat_label}")
                ax.set_xlabel(stat_label); ax.set_ylabel("Games"); ax.legend()
                st.pyplot(fig)

# =========================
# TAB 2: BREAKEVEN
# =========================
with tab_breakeven:
    st.subheader("🔎 Breakeven Finder")

    # Filters row (Seasons + Last N + Min minutes + Location)
    cA, cB, cC, cD, cE = st.columns([2, 1, 1, 1, 1])
    with cA:
        player_query = st.text_input("Player", placeholder="e.g., Stephen Curry")
    with cB:
        seasons_b = st.multiselect("Seasons", SEASON_OPTIONS, default=["2024-25"])
    with cC:
        last_n = st.slider("Last N Games", min_value=5, max_value=100, value=20, step=1)
    with cD:
        min_min_b = st.slider("Min Minutes", min_value=0, max_value=40, value=20, step=1)
    with cE:
        loc_choice = st.selectbox("Location", ["All", "Home Only", "Away"], index=0)

    do_search = st.button("Search")

    if do_search:
        player_name = best_player_match(player_query)
        if not player_name:
            st.warning("Could not match that player. Try a more specific name.")
        else:
            pid = get_player_id(player_name)
            if not pid:
                st.warning("No player ID found for that name.")
            else:
                df = fetch_gamelog(pid, seasons_b if seasons_b else SEASON_OPTIONS)
                if df.empty:
                    st.warning("No game logs found.")
                else:
                    # Apply filters
                    d = df.copy()
                    d = d[d["MIN_NUM"] >= min_min_b]
                    if loc_choice == "Home Only":
                        d = d[d["MATCHUP"].astype(str).str.contains("vs", regex=False)]
                    elif loc_choice == "Away":
                        d = d[d["MATCHUP"].astype(str).str.contains("@", regex=False)]
                    d = d.sort_values("GAME_DATE_DT", ascending=False).head(last_n)

                    if d.empty:
                        st.warning("No games match your filters.")
                    else:
                        left, right = st.columns([1, 2], vertical_alignment="top")
                        with left:
                            st.markdown(f"### **{player_name}**")
                            img = headshot_url(pid)
                            if img:
                                st.image(img, width=180)
                            st.caption(f"Filters: Seasons={', '.join(seasons_b) if seasons_b else '—'} • Last {last_n} • Min {min_min_b}m • {loc_choice}")
                        with right:
                            stat_list = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "P+R", "P+A", "R+A", "PRA"]
                            rows = []
                            for sc in stat_list:
                                ser = compute_stat_series(d, sc)
                                out = breakeven_for_stat(ser)
                                line = out["line"]
                                if line is None:
                                    rows.append({
                                        "Stat": STAT_LABELS.get(sc, sc),
                                        "Breakeven Line": "—",
                                        "Over Implied (Fair)": "—",
                                        "Under Implied (Fair)": "—",
                                    })
                                    continue
                                over_p = out["over_prob"]
                                under_p = out["under_prob"]
                                over_disp = f"{over_p*100:.1f}% ({out['over_odds']})"
                                under_disp = f"{under_p*100:.1f}% ({out['under_odds']})"
                                rows.append({
                                    "Stat": STAT_LABELS.get(sc, sc),
                                    "Breakeven Line": fmt_half(line),
                                    "Over Implied (Fair)": over_disp,
                                    "Under Implied (Fair)": under_disp,
                                })
                            breakeven_df = pd.DataFrame(
                                rows,
                                columns=["Stat", "Breakeven Line", "Over Implied (Fair)", "Under Implied (Fair)"]
                            )
                            st.table(breakeven_df.set_index("Stat"))
