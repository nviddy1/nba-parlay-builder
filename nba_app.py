# nba_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from rapidfuzz import process
from nba_api.stats.static import teams as teams_static
from nba_api.stats.endpoints import leaguegamelog, commonteamroster


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
body, .block-container {
  background: var(--bg);
  color: var(--text);
}
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
.neutral {
  --card-bg:#222;
  --card-border:#777;
}
.pos {
  --card-bg:#0b3d23;
  --card-border:#00FF99;
}
.neg {
  --card-bg:#3d0b0b;
  --card-border:#FF5555;
}
.card h2 {
  color:#fff;
  margin:0 0 6px 0;
  font-weight:800;
  font-size:1.05rem;
}
.cond {
  color:#a9b1bb;
  font-size:0.9rem;
  margin: 2px 0 10px 0;
}
.row {
  display:flex;
  flex-wrap:wrap;
  gap:10px;
  align-items:flex-end;
  justify-content:space-between;
  margin: 6px 0 4px 0;
}
.m {
  min-width:120px;
  flex:1;
}
.lab {
  color:#cbd5e1;
  font-size:0.8rem;
  margin-bottom:2px;
}
.val {
  color:#fff;
  font-size:1.1rem;
  font-weight:800;
  line-height:1.1;
}
.chip {
  display:inline-block;
  margin-top:10px;
  padding:6px 12px;
  border-radius:999px;
  font-size:0.8rem;
  color:#a7f3d0;
  border:1px solid #16a34a33;
  background: transparent;
}
/* Simple table polish for Breakeven tab */
table {
  border-collapse: collapse;
}
thead th {
  border-bottom: 1px solid #374151 !important;
}
tbody td, thead th {
  padding: 8px 10px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CONSTANTS & HELPERS
# =========================
STAT_LABELS = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "FG3M": "3PM",
    "DOUBDOUB": "Doub Doub",
    "TRIPDOUB": "Trip Doub",
    "P+R": "P+R",
    "P+A": "P+A",
    "R+A": "R+A",
    "PRA": "PRA",
}

STAT_TOKENS = {
    "P": "PTS",
    "PTS": "PTS",
    "POINTS": "PTS",
    "R": "REB",
    "REB": "REB",
    "REBOUNDS": "REB",
    "A": "AST",
    "AST": "AST",
    "ASSISTS": "AST",
    "STL": "STL",
    "STEALS": "STL",
    "BLK": "BLK",
    "BLOCKS": "BLK",
    "3PM": "FG3M",
    "FG3M": "FG3M",
    "THREES": "FG3M",
    "DOUBDOUB": "DOUBDOUB",
    "DOUBLEDOUBLE": "DOUBDOUB",
    "DOUB": "DOUBDOUB",
    "DD": "DOUBDOUB",
    "TRIPDOUB": "TRIPDOUB",
    "TRIPLEDOUBLE": "TRIPDOUB",
    "TD": "TRIPDOUB",
    "P+R": "P+R",
    "PR": "P+R",
    "P+A": "P+A",
    "PA": "P+A",
    "R+A": "R+A",
    "RA": "R+A",
    "PRA": "PRA"
}

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

def american_to_implied(odds: int | float | str):
    try:
        x = float(odds)
    except Exception:
        return None
    if -99 < x < 100:
        return None
    if x > 0:
        return 100.0/(x+100.0)
    return abs(x)/(abs(x)+100.0)

def prob_to_american(p: float):
    if p <= 0 or p >= 1:
        return "N/A"
    dec = 1.0/p
    if dec >= 2.0:
        return f"+{int(round((dec-1)*100))}"
    return f"-{int(round(100/(dec-1)))}"

def fmt_half(x: float | int) -> str:
    try:
        v = float(x)
        return f"{v:.1f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def parse_input_line(text: str):
    t = (text or "").strip()
    if not t:
        return None
    parts = t.replace("/", "+").split()
    dir_token = None
    for token in parts:
        if token.upper() in ["O", "OVER"]:
            dir_token = "Over"; break
        if token.upper() in ["U", "UNDER"]:
            dir_token = "Under"; break
    if not dir_token:
        dir_token = "Over"
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
    stat_code = None
    combined_aliases = ["P+R","P+A","R+A","PRA","PR","PA","RA"]
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
    loc = "All"
    for token in parts:
        up = token.upper()
        if up in ["AWAY","A"]:
            loc = "Away"; break
        if up in ["HOME","H"]:
            loc = "Home Only"; break
    odds = -110
    for token in parts[::-1]:
        if token.startswith("+") or token.startswith("-"):
            try:
                o = int(token)
                if o <= -100 or o >= 100:
                    odds = o
                    break
            except Exception:
                continue
    banned = set(["O","OVER","U","UNDER","HOME","H","AWAY","A"] + list(STAT_TOKENS.keys()) + combined_aliases)
    name_tokens = [p for p in parts if (p.upper() not in banned and not p.replace(".", "", 1).lstrip("+-").isdigit())]
    name_guess = " ".join(name_tokens).strip()
    player = best_player_match(name_guess)
    return {"player": player, "dir": dir_token, "thr": float(thr), "stat": stat_code, "loc": loc, "range": "FULL", "odds": int(odds)}

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

def fetch_gamelog(player_id: int, seasons: list[str], include_playoffs: bool=False, only_playoffs: bool=False) -> pd.DataFrame:
    dfs = []
    for s in seasons:
        # Regular season logs (skip if only playoffs)
        if not only_playoffs:
            try:
                reg = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=s,
                    season_type_all_star="Regular Season"
                ).get_data_frames()[0]
                dfs.append(reg)
            except Exception:
                pass
        # Playoff logs if selected or if only playoffs
        if include_playoffs or only_playoffs:
            try:
                po = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=s,
                    season_type_all_star="Playoffs"
                ).get_data_frames()[0]
                dfs.append(po)
            except Exception:
                pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    for k in ["PTS","REB","AST","STL","BLK","FG3M"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    df["MIN_NUM"] = df["MIN"].apply(to_minutes) if "MIN" in df.columns else 0
    if "GAME_DATE" in df.columns:
        df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    else:
        df["GAME_DATE_DT"] = pd.Timestamp.now()
    return df

def compute_stat_series(df: pd.DataFrame, stat_code: str) -> pd.Series:
    s = pd.Series(dtype=float, index=df.index)
    if stat_code in ["PTS","REB","AST","STL","BLK","FG3M"]:
        s = df[stat_code].astype(float)
    elif stat_code == "P+R":
        s = (df["PTS"] + df["REB"]).astype(float)
    elif stat_code == "P+A":
        s = (df["PTS"] + df["AST"]).astype(float)
    elif stat_code == "R+A":
        s = (df["REB"] + df["AST"]).astype(float)
    elif stat_code == "PRA":
        s = (df["PTS"] + df["REB"] + df["AST"]).astype(float)
    elif stat_code == "DOUBDOUB":
        pts = (df["PTS"] >= 10).astype(int)
        reb = (df["REB"] >= 10).astype(int)
        ast = (df["AST"] >= 10).astype(int)
        s = ((pts + reb + ast) >= 2).astype(int)
    elif stat_code == "TRIPDOUB":
        pts = (df["PTS"] >= 10).astype(int)
        reb = (df["REB"] >= 10).astype(int)
        ast = (df["AST"] >= 10).astype(int)
        s = ((pts + reb + ast) >= 3).astype(int)
    else:
        s = df["PTS"].astype(float)
    return s

def leg_probability(df: pd.DataFrame, stat_code: str, direction: str, thr: float) -> tuple[float,int,int]:
    ser = compute_stat_series(df, stat_code)
    if stat_code in ["DOUBDOUB","TRIPDOUB"]:
        hits = int((ser <= 0.5).sum()) if direction == "Under" else int((ser >= 0.5).sum())
    else:
        hits = int((ser <= thr).sum()) if direction == "Under" else int((ser >= thr).sum())
    total = int(ser.notna().sum())
    p = hits/total if total else 0.0
    return p, hits, total

def headshot_url(pid: int | None) -> str | None:
    if not pid:
        return None
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"

def get_team_logo(player_id: int | None):
    if not player_id:
        return None
    try:
        # NBA CDN format for team logo
        return f"https://cdn.nba.com/logos/nba/{player_id}/global/L/logo.svg"
    except:
        return None

def breakeven_for_stat(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    total = len(s)
    if total == 0:
        return {"line": None, "over_prob": None, "under_prob": None, "over_odds": "N/A", "under_odds": "N/A"}
    lo = np.floor(s.min()) - 0.5
    hi = np.ceil(s.max()) + 0.5
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

def get_team_logo_from_df(df):
    try:
        team_id = df["TEAM_ID"].iloc[0]
        return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
    except:
        return None

def sparkline(values, thr):
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    import base64
    # clean series
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return ""
    # create sparkline
    fig, ax = plt.subplots(figsize=(4,0.35))  # tiny chart
    fig.patch.set_alpha(0.0)
    ax.set_axis_off()
    # line
    ax.plot(vals.index, vals.values, linewidth=1, alpha=0.8)
    # threshold line
    ax.axhline(thr, color="white", linestyle="--", linewidth=1)
    # convert to inline base64
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;height:22px;opacity:0.9;" />'

# -------------------------
# Team constants & colors
# -------------------------
TEAM_COLORS = {
    "ATL": "#E03A3E","BOS": "#007A33","BKN": "#000000","CHA": "#1D1160","CHI": "#CE1141",
    "CLE": "#860038","DAL": "#00538C","DEN": "#0E2240","DET": "#C8102E","GSW": "#1D428A",
    "HOU": "#CE1141","IND": "#002D62","LAC": "#C8102E","LAL": "#552583","MEM": "#5D76A9",
    "MIA": "#98002E","MIL": "#00471B","MIN": "#0C2340","NOP": "#0C2340","NYK": "#F58426",
    "OKC": "#007AC1","ORL": "#0077C0","PHI": "#006BB6","PHX": "#1D1160","POR": "#E03A3E",
    "SAC": "#5A2D81","SAS": "#C4CED4","TOR": "#CE1141","UTA": "#002B5C","WAS": "#002B5C"
}
TEAM_ABBRS = sorted(TEAM_COLORS.keys())


def lighten_hex(color: str, factor: float = 0.35) -> str:
    """Slightly lighten a hex color so the text pops more."""
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# -------------------------
# League logs (player-level)
# -------------------------
@st.cache_data(show_spinner=False)
def get_league_player_logs(season: str) -> pd.DataFrame:
    """All player game logs for a season (used for Hot Matchups & Injury Impact)."""
    df = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",
    ).get_data_frames()[0]

    for col in ["PTS", "REB", "AST", "STL", "BLK", "FG3M"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # opponent team abbrev from MATCHUP (same regex you used before)
    df["OPP"] = (
        df["MATCHUP"]
        .astype(str)
        .str.extract(r"vs\. (\w+)|@ (\w+)", expand=True)
        .bfill(axis=1)
        .iloc[:, 0]
    )
    return df


@st.cache_data(show_spinner=False)
def get_team_defense_table(season: str) -> pd.DataFrame:
    """
    Returns per-game averages allowed by each team:
    PTS, REB, AST, FG3M allowed.
    """
    df = get_league_player_logs(season)
    # Sum all opponent players per game, then average by opponent team
    tmp = (
        df.groupby(["OPP", "GAME_ID"])[["PTS", "REB", "AST", "FG3M"]]
        .sum()
        .reset_index()
    )
    agg = (
        tmp.groupby("OPP")[["PTS", "REB", "AST", "FG3M"]]
        .mean()
        .reset_index()
        .rename(columns={
            "OPP": "Team",
            "PTS": "PTS_allowed",
            "REB": "REB_allowed",
            "AST": "AST_allowed",
            "FG3M": "FG3M_allowed",
        })
    )
    return agg


@st.cache_data(show_spinner=False)
def get_team_roster(season: str, team_abbrev: str) -> pd.DataFrame:
    """Return roster (PLAYER_ID, PLAYER) for a team in a given season."""
    team_meta = [t for t in teams_static.get_teams() if t["abbreviation"] == team_abbrev]
    if not team_meta:
        return pd.DataFrame(columns=["PLAYER_ID", "PLAYER"])
    team_id = team_meta[0]["id"]
    roster_df = commonteamroster.CommonTeamRoster(
        team_id=team_id,
        season=season
    ).get_data_frames()[0]
    return roster_df[["PLAYER_ID", "PLAYER"]]


def style_def_table(df: pd.DataFrame, stat_col: str):
    """Nice colored table for Hot Matchups."""
    def row_style(row):
        team = row["Team"]
        base = TEAM_COLORS.get(team, "#111827")
        bg_team = lighten_hex(base, 0.4)
        bg_val = "#020617"
        return [
            f"background-color:{bg_team}; color:#f9fafb; font-weight:700; text-align:left;",
            f"background-color:{bg_val}; color:#f9fafb; text-align:right;"
        ]

    return (
        df.style.hide(axis="index")
        .format({stat_col: "{:.1f}"})
        .apply(row_style, axis=1)
    )


def monte_carlo_sim(series: pd.Series, n_sims: int = 10000) -> np.ndarray:
    """Bootstrap Monte Carlo from historical stat series."""
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0:
        return np.array([])
    return np.random.choice(vals, size=n_sims, replace=True)

def monte_carlo_predictive(series: pd.Series, n_sims: int = 10000) -> np.ndarray:
    """
    Predictive Monte Carlo using kernel smoothing (SciPy-free).
    Produces a smoother, bell-curve-ish distribution anchored to real stats.
    """
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0:
        return np.array([])

    mean = float(np.mean(vals))
    std = float(np.std(vals))

    # If no variance, just repeat the mean
    if std == 0 or len(vals) == 1:
        return np.full(n_sims, mean)

    # Silverman's rule of thumb for bandwidth
    n = len(vals)
    bandwidth = 1.06 * std * n ** (-1/5)

    base = np.random.choice(vals, size=n_sims, replace=True)
    noise = np.random.normal(0, bandwidth, size=n_sims)

    draws = base + noise
    # stats can't go below zero
    return np.clip(draws, 0, None)

def render_mc_result_card(player, direction, thr, stat_label, loc_text, last_n, p_hit, fair_odds, sb_odds, ev_pct):
    ev_str = "—" if ev_pct is None else f"{ev_pct:.2f}%"
    hit_str = f"{p_hit*100:.1f}%"
    cls = "neutral"

    if ev_pct is not None:
        cls = "pos" if ev_pct >= 0 else "neg"

    return f"""
<style>
.card {{
    background-color: #0f291e;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1e3a2f;
    margin-top: 10px;
}}
.card.pos {{ border-color: #00c896; }}
.card.neg {{ border-color: #ff6b6b; }}
.card h2 {{
    margin: 0 0 8px 0;
    color: #f0fdf4;
    font-size: 1.25rem;
}}
.card .cond {{
    color: #d1fae5;
    font-size: 0.9rem;
    margin-bottom: 12px;
}}
.card .row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}}
.card .m {{
    background-color: #15342a;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #1e4d3b;
}}
.card .lab {{
    font-size: 0.75rem;
    color: #9ca3af;
}}
.card .val {{
    font-size: 1.1rem;
    color: #f0fdf4;
    font-weight: 600;
}}
.chip {{
    display: inline-block;
    background-color: #1e4d3b;
    color: #d1fae5;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
}}
</style>

<div class="card {cls}">
  <h2>🎲 Monte Carlo Result</h2>

  <div class="cond">
    {player} — {direction} {thr} {stat_label} ({loc_text}, last {last_n} games)
  </div>

  <div class="row">
    <div class="m"><div class="lab">Sim Hit %</div><div class="val">{hit_str}</div></div>
    <div class="m"><div class="lab">Fair Odds</div><div class="val">{fair_odds}</div></div>
    <div class="m"><div class="lab">Book Odds</div><div class="val">{sb_odds}</div></div>
    <div class="m"><div class="lab">EV</div><div class="val">{ev_str}</div></div>
  </div>

  <div style="margin-top:10px;">
    <span class="chip">{('🔥 +EV (Monte Carlo)' if (ev_pct is not None and ev_pct >= 0) else '⚠️ Negative EV by simulation')}</span>
  </div>
</div>
"""

def render_mc_distribution_card(mean_val, median_val, stdev, p10, p90, hit_prob):

    html = f"""
<style>
.mc-sum {{
    background-color:#0f291e;
    padding:20px;
    border-radius:12px;
    border:1px solid #1e3a2f;
    margin-top:18px;
}}
.mc-sum-title {{
    font-size:1.15rem;
    font-weight:700;
    color:#d1fae5;
    margin-bottom:12px;
}}
.mc-grid {{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:14px;
}}
.mc-sbox {{
    background-color:#15342a;
    padding:12px 14px;
    border-radius:10px;
    border:1px solid #1e4d3b;
}}
.mc-slab {{
    font-size:0.8rem;
    color:#9ca3af;
    margin-bottom:4px;
}}
.mc-sval {{
    font-size:1.1rem;
    font-weight:600;
    color:#f0fdf4;
}}
</style>

<div class="mc-sum">
    <div class="mc-sum-title">📊 Distribution Summary</div>

    <div class="mc-grid">

        <div class="mc-sbox">
            <div class="mc-slab">Mean</div>
            <div class="mc-sval">{mean_val:.1f}</div>
        </div>

        <div class="mc-sbox">
            <div class="mc-slab">Median</div>
            <div class="mc-sval">{median_val:.1f}</div>
        </div>

        <div class="mc-sbox">
            <div class="mc-slab">Std Dev</div>
            <div class="mc-sval">{stdev:.2f}</div>
        </div>

        <div class="mc-sbox">
            <div class="mc-slab">10th Percentile</div>
            <div class="mc-sval">{p10:.1f}</div>
        </div>

        <div class="mc-sbox">
            <div class="mc-slab">90th Percentile</div>
            <div class="mc-sval">{p90:.1f}</div>
        </div>

        <div class="mc-sbox">
            <div class="mc-slab">Sim Hit %</div>
            <div class="mc-sval">{hit_prob*100:.1f}%</div>
        </div>

    </div>
</div>
"""

    return html




# Define NBA_CUP_DATES (example dates; update as needed for the season)
NBA_CUP_DATES = pd.to_datetime([
    # Add actual NBA In-Season Tournament dates here, e.g.,
    # "2024-11-12", "2024-11-13", etc.
    # For 2025-26 season, placeholder empty for now
])

# =========================
# TABS
# =========================
tab_builder, tab_breakeven, tab_mc, tab_injury, tab_matchups = st.tabs(
    ["🧮 Parlay Builder", "🧷 Breakeven", "🎲 Monte Carlo Sim", "🩹 Injury Impact", "📊 Hot Matchups"]
)

# =========================
# TAB 1: PARLAY BUILDER
# =========================
with tab_builder:
    # Filters Row
    fc1, fc2, fc3 = st.columns([1.2, 1, 1])
    with fc1:
        seasons = st.multiselect(
            "Seasons",
            ["2025-26","2024-25","2023-24","2022-23"],
            default=["2025-26","2024-25"],
            key="seasons_builder"
        )
        include_playoffs = st.checkbox("Include Playoffs", value=False, key="pb_playoffs")
        only_playoffs = st.checkbox("Only Playoffs", value=False, key="pb_only_playoffs")
        nba_cup_only = st.checkbox("NBA Cup Games Only", key="cup_only_builder")
    with fc2:
        min_minutes = st.slider("Min Minutes", 0, 40, 20, 1, key="min_minutes_builder")
    with fc3:
        last_n_games = st.slider("Last N Games", 5, 100, 20, 1, key="parlay_lastn")

    # State
    if "legs" not in st.session_state:
        st.session_state.legs = []
    if "awaiting_input" not in st.session_state:
        st.session_state.awaiting_input = True

    # Buttons row
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("➕ Add Leg"):
            st.session_state.awaiting_input = True
    with c2:
        if st.button("➖ Remove Last Leg") and st.session_state.legs:
            st.session_state.legs.pop()
            st.rerun()

    st.write("**Input bet**")

    # Existing legs edit UI
    if st.session_state.legs:
        for i, leg in enumerate(st.session_state.legs):
            leg_no = i + 1
            dir_short = "O" if leg["dir"] == "Over" else "U"
            header = f"Leg {leg_no}: {leg['player']} — {dir_short} {fmt_half(leg['thr'])} {STAT_LABELS.get(leg['stat'], leg['stat'])} ({leg['loc']}, {leg['odds']})"
            with st.expander(header, expanded=False):
                cL, cR = st.columns([2,1])
                with cL:
                    leg["player"] = st.text_input("Player", value=leg["player"], key=f"player_{i}")
                    leg["stat"] = st.selectbox("Stat", list(STAT_LABELS.keys()), index=list(STAT_LABELS.keys()).index(leg["stat"]), key=f"stat_{i}")
                    leg["dir"] = st.selectbox("O/U", ["Over","Under"], index=(0 if leg["dir"]=="Over" else 1), key=f"dir_{i}")
                    leg["thr"] = st.number_input("Threshold", value=float(leg["thr"]), step=0.5, key=f"thr_{i}")
                with cR:
                    leg["loc"] = st.selectbox("Home/Away", ["All","Home Only","Away"], index=["All","Home Only","Away"].index(leg["loc"]), key=f"loc_{i}")
                    leg["range"] = st.selectbox("Game Range", ["FULL","L10","L20"], index=["FULL","L10","L20"].index(leg.get("range","FULL")), key=f"range_{i}")
                    leg["odds"] = st.number_input("Sportsbook Odds", value=int(leg["odds"]), step=5, key=f"odds_{i}")
                rm_col, _ = st.columns([1,5])
                with rm_col:
                    if st.button(f"❌ Remove Leg {leg_no}", key=f"remove_{i}"):
                        st.session_state.legs.pop(i)
                        st.rerun()

    # Freeform input box
    if st.session_state.awaiting_input:
        bet_text = st.text_input(
            "Input bet",
            placeholder="Maxey O 24.5 P Away -110",
            key="freeform_input",
            label_visibility="collapsed"
        )
        if bet_text.strip():
            parsed = parse_input_line(bet_text)
            if parsed and parsed["player"]:
                st.session_state.legs.append(parsed)
                st.session_state.awaiting_input = False
                st.rerun()

    # Enter parlay odds
    parlay_odds = 0
    if len(st.session_state.legs) > 1:
        st.subheader("🎯 Combined Parlay Odds")
        parlay_odds = st.number_input("Parlay Odds (+300, -150)", value=0, step=5, key="parlay_odds")

    # Compute action
    if st.session_state.legs and st.button("Compute"):
        st.write("---")
        rows = []
        probs_for_parlay = []
        plt.rcParams.update({
            "axes.facecolor": "#1e1f22",
            "figure.facecolor": "#1e1f22",
            "text.color": "#ffffff",
            "axes.labelcolor": "#e5e7eb",
            "xtick.color": "#e5e7eb",
            "ytick.color": "#e5e7eb",
            "grid.color": "#374151",
        })

        def get_team_logo_from_df(df):
            try:
                return f"https://cdn.nba.com/logos/nba/{int(df['TEAM_ID'].iloc[0])}/global/L/logo.svg"
            except:
                return None

        # Loop legs
        for leg in st.session_state.legs:
            name = leg["player"]
            pid = get_player_id(name)
            stat = leg["stat"]
            direction = leg["dir"]
            thr = float(leg["thr"])
            loc = leg["loc"]
            rng = leg.get("range","FULL")
            odds = int(leg["odds"])
            df = pd.DataFrame()
            if pid:
                df = fetch_gamelog(pid, seasons, include_playoffs, only_playoffs)
            if df.empty:
                st.warning(f"Could not compute for **{name}**")
                continue
            d = df.copy()
            d = d[d["MIN_NUM"] >= min_minutes]
            if loc == "Home Only":
                d = d[d["MATCHUP"].astype(str).str.contains("vs")]
            elif loc == "Away":
                d = d[d["MATCHUP"].astype(str).str.contains("@")]
            # NBA Cup filter
            if nba_cup_only:
                d = d[d["GAME_DATE_DT"].isin(NBA_CUP_DATES)]
            d = d.sort_values("GAME_DATE_DT", ascending=False)
            if rng == "L10":
                d = d.head(10)
            elif rng == "L20":
                d = d.head(20)
            else:
                d = d.head(last_n_games)
            p, hits, total = leg_probability(d, stat, direction, thr)
            fair = prob_to_american(p)
            book_prob = american_to_implied(odds)
            ev = None
            if book_prob is None:
                pass
            else:
                ev = (p - book_prob) * 100.0
            if p > 0:
                probs_for_parlay.append(p)
            rows.append({
                "name": name,
                "stat": stat,
                "thr": thr,
                "dir": direction,
                "loc": loc,
                "range": rng,
                "odds": odds,
                "p": p,
                "hits": hits,
                "total": total,
                "fair": fair,
                "book_prob": book_prob,
                "ev": ev,
                "df": d
            })

        # Combined Parlay Stats
        combined_p = float(np.prod(probs_for_parlay)) if probs_for_parlay else 0.0
        combined_fair = prob_to_american(combined_p) if combined_p > 0 else "N/A"
        entered_prob = american_to_implied(parlay_odds)
        book_implied_prob = entered_prob * 100 if entered_prob else None
        parlay_ev = None
        if entered_prob is None:
            pass
        else:
            parlay_ev = (combined_p - entered_prob) * 100.0
        cls = "neutral"
        if parlay_ev is not None:
            cls = "pos" if parlay_ev >= 0 else "neg"
        combined_prob_disp = f"{combined_p*100:.2f}%" if combined_p > 0 else "—"
        parlay_ev_disp = "—" if parlay_ev is None else f"{parlay_ev:.2f}%"
        book_implied_str = "—" if book_implied_prob is None else f"{book_implied_prob:.2f}%"

        # ✅ Combined Parlay Card (top)
        st.markdown(f"""
<div class="card {cls}">
  <h2>💥 Combined Parlay</h2>
  <div style="font-size:0.95rem;color:#9ca3af;margin-bottom:14px;">
    Includes all legs with your filters
  </div>
  <div class="row">
    <div class="m"><div class="lab">Model Parlay Probability</div><div class="val">{combined_prob_disp}</div></div>
    <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{combined_fair}</div></div>
    <div class="m"><div class="lab">Entered Odds</div><div class="val">{parlay_odds}</div></div>
    <div class="m"><div class="lab">Book Implied</div><div class="val">{book_implied_str}</div></div>
    <div class="m"><div class="lab">Expected Value</div><div class="val">{parlay_ev_disp}</div></div>
  </div>
  <div style="margin-top:14px;">
    <span class="chip">
      {('🔥 +EV Parlay Detected' if (parlay_ev is not None and parlay_ev >= 0) else '⚠️ Negative EV Parlay')}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

        # ✅ Player Cards
        for r in rows:
            pid = get_player_id(r["name"])
            head = headshot_url(pid)
            logo = get_team_logo_from_df(r["df"])
            cls = "neutral"
            if r["ev"] is not None:
                cls = "pos" if r["ev"] >= 0 else "neg"
            img_html = f'<img src="{head}" width="72" style="border-radius:12px;">' if head else ""
            logo_html = f'<img src="{logo}" width="38" style="margin-left:10px;">' if logo else ""
            stat_label = STAT_LABELS.get(r["stat"], r["stat"])
            dir_word = "O" if r["dir"] == "Over" else "U"
            cond_text = f"{dir_word} {fmt_half(r['thr'])} {stat_label} — {r['range']} — {r['loc'].replace(' Only','')}"
            book_implied = "—" if r["book_prob"] is None else f"{r['book_prob']*100:.1f}%"
            ev_disp = "—" if r["ev"] is None else f"{r['ev']:.2f}%"
            st.markdown(f"""
<div class="card {cls}">
  <div style="display:flex;align-items:center;gap:12px;">
    {img_html}
    <div style="display:flex;flex-direction:column;">
      <h2 style="margin:0;font-size:1.25rem;">{r['name']}</h2>
      <div style="font-size:0.90rem;color:#9ca3af;margin-top:4px;">{cond_text}</div>
    </div>
    {logo_html}
  </div>
  <div class="row" style="margin-top:18px;">
    <div class="m"><div class="lab">Model Hit Rate</div><div class="val">{r['p']*100:.1f}% ({r['hits']}/{r['total']})</div></div>
    <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{r['fair']}</div></div>
    <div class="m"><div class="lab">Sportsbook Odds</div><div class="val">{r['odds']}</div></div>
    <div class="m"><div class="lab">Book Implied</div><div class="val">{book_implied}</div></div>
    <div class="m"><div class="lab">Expected Value</div><div class="val">{ev_disp}</div></div>
  </div>
  <div style="margin-top:10px;">
    <span class="chip">{('🔥 +EV Play Detected' if (r['ev'] is not None and r['ev'] >= 0) else '⚠️ Negative EV Play')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

            # ✅ Discrete Value Distribution (1 bar per stat result)
            if r["stat"] not in ["DOUBDOUB","TRIPDOUB"]:
                ser = compute_stat_series(r["df"], r["stat"]).dropna().astype(int)
                value_counts = ser.value_counts().sort_index()
                data_min, data_max = int(ser.min()), int(ser.max())
                all_vals = list(range(data_min, data_max + 1))
                counts = [value_counts.get(v, 0) for v in all_vals]
                label_count = max(1, (data_max - data_min) // 5)
                with st.expander("📊 Show Game Distribution", expanded=False):
                    fig, ax = plt.subplots(figsize=(6.5, 2.4))
                    fig.patch.set_facecolor("#1e1f22")
                    ax.set_facecolor("#1e1f22")
                    color = "#00c896" if (r["ev"] is not None and r["ev"] >= 0) else "#e05a5a"
                    ax.bar(all_vals, counts, color=color, alpha=0.75, edgecolor="#d1d5db", linewidth=0.5, width=0.9)
                    ax.axvline(r["thr"], color="white", linestyle="--", linewidth=1.4)
                    ax.set_xticks(all_vals)
                    ax.set_xticklabels(
                        [str(v) if ((v - data_min) % label_count == 0 or v == r["thr"]) else "" for v in all_vals],
                        color="#9ca3af", fontsize=8
                    )
                    ax.tick_params(axis="y", colors="#9ca3af", labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#4b5563")
                    ax.grid(color="#374151", linestyle="--", linewidth=0.5, alpha=0.55)
                    ax.set_axisbelow(True)
                    ax.set_ylabel("")
                    ax.set_xlabel("")
                    st.pyplot(fig, use_container_width=True)

# =========================
# TAB 2: BREAKEVEN
# =========================
with tab_breakeven:
    st.subheader("🔎 Breakeven Finder")

    # Filters
    f1, f2 = st.columns([1.2, 1])
    with f1:
        seasons_b = st.multiselect(
            "Seasons",
            ["2025-26","2024-25","2023-24","2022-23"],
            default=["2025-26","2024-25"],
            key="seasons_breakeven"
        )
        include_playoffs_b = st.checkbox("Include Playoffs", value=False, key="be_playoffs")
        only_playoffs_b = st.checkbox("Only Playoffs", value=False, key="be_only_playoffs")
        nba_cup_only_b = st.checkbox("NBA Cup Games Only", key="cup_only_break")
    cA, cB, cC, cD = st.columns([2,1,1,1])
    with cA:
        player_query = st.text_input("Player", placeholder="e.g., Stephen Curry", key="breakeven_player")
    with cB:
        last_n = st.slider("Last N", 5, 100, 20, 1, key="breakeven_lastn")
    with cC:
        min_min_b = st.slider("Min Min", 0, 40, 20, 1, key="breakeven_minmin")
    with cD:
        loc_choice = st.selectbox("Location", ["All","Home Only","Away"], index=0, key="breakeven_loc")
    do_search = st.button("Search", key="breakeven_search")

    if do_search:
        player_name = best_player_match(player_query)
        if not player_name:
            st.warning("Could not match player.")
        else:
            pid = get_player_id(player_name)
            df = fetch_gamelog(pid, seasons_b, include_playoffs_b, only_playoffs_b)
            d = df.copy()
            d = d[d["MIN_NUM"] >= min_min_b]
            if loc_choice == "Home Only":
                d = d[d["MATCHUP"].astype(str).str.contains("vs", regex=False)]
            elif loc_choice == "Away":
                d = d[d["MATCHUP"].astype(str).str.contains("@", regex=False)]
            if nba_cup_only_b:
                d = d[d["GAME_DATE_DT"].isin(NBA_CUP_DATES)]
            d = d.sort_values("GAME_DATE_DT", ascending=False).head(last_n)
            left, right = st.columns([1,2])

            # ------------------------------
            # LEFT COLUMN: PLAYER CARD
            # ------------------------------
            with left:
                img = headshot_url(pid)
                logo = get_team_logo_from_df(df)
                img_html = f'<img src="{img}" width="120" style="border-radius:10px;">' if img else ""
                logo_html = f'<img src="{logo}" width="55" style="margin-top:8px;">' if logo else ""
                # ✅ Clean inline HTML string (NO triple quotes)
                player_html = (
                    f"<div style='text-align:center; padding:10px;'>"
                    f"{img_html}<br>"
                    f"{logo_html if logo_html else ''}"
                    f"<div style='font-size:1.3rem; font-weight:700; margin-top:6px;'>"
                    f"{player_name}"
                    f"</div></div>"
                )
                st.markdown(player_html, unsafe_allow_html=True)
                st.caption(
                    f"Filters: Seasons {', '.join(seasons_b)} • Last {last_n} • "
                    f"Min {min_min_b}m • {loc_choice}"
                )

            # ------------------------------
            # RIGHT COLUMN: BREAKEVEN TABLE
            # ------------------------------
            with right:
                stat_list = ["PTS","REB","AST","FG3M","STL","BLK","P+R","P+A","R+A","PRA"]
                rows = []
                for sc in stat_list:
                    out = breakeven_for_stat(compute_stat_series(d, sc))
                    line = out["line"]
                    if line is None:
                        rows.append({"Stat": STAT_LABELS[sc], "Breakeven Line": "—", "Over": "—", "Under": "—"})
                        continue
                    rows.append({
                        "Stat": STAT_LABELS[sc],
                        "Breakeven Line": fmt_half(line),
                        "Over": f"{out['over_prob']*100:.1f}% ({out['over_odds']})",
                        "Under": f"{out['under_prob']*100:.1f}% ({out['under_odds']})"
                    })
                st.table(pd.DataFrame(rows).set_index("Stat"))

# =========================
# TAB 3: MONTE CARLO PROP SIMULATOR
# =========================
with tab_mc:
    st.subheader("🎲 Monte Carlo Prop Simulator (Predictive)")

    # ---- Bet Input ----
    mc_text = st.text_input(
        "Prop (e.g., 'Maxey O 29.5 PTS Away -110')",
        placeholder="Enter player + O/U + line + stat + location + odds"
    )

    # ---- Filters ----
    seasons_mc = st.multiselect(
        "Seasons",
        ["2025-26","2024-25","2023-24","2022-23"],
        default=["2025-26","2024-25"]
    )
    last_n_mc = st.slider("Last N Games", 5, 100, 20)
    min_min_mc = st.slider("Min Minutes", 0, 40, 20)
    sims_mc    = st.slider("Number of Simulations", 2000, 50000, 15000, 2000)

    # ==========================================
    # ---------- Predictive Monte Carlo ----------
    # ==========================================
    def mc_predictive(series: pd.Series, n_sims: int = 10000) -> np.ndarray:
        vals = pd.to_numeric(series, errors="coerce").dropna().values
        if len(vals) == 0:
            return np.array([])

        μ = np.mean(vals)
        σ = np.std(vals)

        if σ == 0 or len(vals) == 1:
            return np.full(n_sims, μ)

        # Silverman's bandwidth
        n = len(vals)
        bw = 1.06 * σ * n ** (-1/5)

        base = np.random.choice(vals, size=n_sims, replace=True)
        noise = np.random.normal(0, bw, n_sims)

        draws = base + noise
        return np.clip(draws, 0, None)

    # ==========================================
    # ---------- Run Simulation ----------
    # ==========================================
    if st.button("Run Simulation") and mc_text.strip():

        parsed = parse_input_line(mc_text)
        if not parsed:
            st.warning("Could not parse the input line.")
            st.stop()

        pid = get_player_id(parsed["player"])
        if not pid:
            st.warning("Player not found.")
            st.stop()

        # Fetch logs
        df = fetch_gamelog(pid, seasons_mc, include_playoffs=False, only_playoffs=False)
        d = df.copy()
        d = d[d["MIN_NUM"] >= min_min_mc]

        # Location filter
        if parsed["loc"] == "Home Only":
            d = d[d["MATCHUP"].str.contains("vs")]
        elif parsed["loc"] == "Away":
            d = d[d["MATCHUP"].str.contains("@")]

        d = d.sort_values("GAME_DATE_DT", ascending=False).head(last_n_mc)

        ser = compute_stat_series(d, parsed["stat"]).dropna()
        if ser.empty:
            st.warning("No valid stat history.")
            st.stop()

        # ---------- Monte Carlo ----------
        draws = mc_predictive(ser, sims_mc)

        thr   = parsed["thr"]
        direction = parsed["dir"]

        hit_prob = float((draws <= thr).mean()) if direction == "Under" else float((draws >= thr).mean())
        fair_odds = prob_to_american(hit_prob)
        book_odds = parsed["odds"]
        book_prob = american_to_implied(book_odds)
        ev_pct = None if book_prob is None else (hit_prob - book_prob) * 100

        stat_name = STAT_LABELS.get(parsed["stat"], parsed["stat"])

        # ==========================================
        # ---------- Result Card ----------
        # ==========================================
        st.markdown(
            render_mc_result_card(
                parsed["player"],
                "O" if direction == "Over" else "U",
                thr,
                stat_name,
                parsed["loc"],
                last_n_mc,
                hit_prob,
                fair_odds,
                book_odds,
                ev_pct
            ),
            unsafe_allow_html=True
        )

        # ==========================================
        # ---------- Distribution Summary ----------
        # ==========================================
        mean_val = float(np.mean(draws))
        median_val = float(np.median(draws))
        p10 = float(np.percentile(draws, 10))
        p90 = float(np.percentile(draws, 90))
        stdev = float(np.std(draws))

        st.markdown(
            render_mc_distribution_card(
                mean_val, median_val, stdev, p10, p90, hit_prob
            ),
            unsafe_allow_html=True,
        )

        # ==========================================
        # ---------- Histogram ----------
        # ==========================================
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#1e1f22")
        ax.set_facecolor("#1e1f22")

        ax.hist(
            draws, bins=25,
            color="#00c896", alpha=0.75,
            edgecolor="#d1d5db", linewidth=0.4
        )
        ax.axvline(thr, color="#ff6666", linestyle="--", linewidth=1.8)

        ax.set_xlabel(stat_name, color="#e5e7eb")
        ax.set_ylabel("Frequency", color="#e5e7eb")
        ax.tick_params(colors="#9ca3af")

        for s in ax.spines.values():
            s.set_edgecolor("#4b5563")

        st.pyplot(fig, use_container_width=True)



# =========================
# TAB 4: INJURY IMPACT ANALYZER
# =========================
with tab_injury:
    st.subheader("🩹 Injury Impact Analyzer")

    colL, colR = st.columns([1.2, 2])

    # --------------------------
    # LEFT-SIDE CONTROLS
    # --------------------------
    with colL:
        season_inj = st.selectbox(
            "Season",
            ["2025-26","2024-25","2023-24"],
            index=1,
            key="season_inj"
        )

        team_inj = st.selectbox(
            "Team",
            TEAM_ABBRS,
            index=TEAM_ABBRS.index("PHX") if "PHX" in TEAM_ABBRS else 0
        )

        roster_df = get_team_roster(season_inj, team_inj)

        if roster_df.empty:
            st.warning("Could not load roster for this team/season.")
            injured_name = None
            injured_id = None
        else:
            injured_name = st.selectbox(
                "Injured / Missing Player",
                roster_df["PLAYER"].tolist(),
                key="inj_player"
            )
            injured_id = int(
                roster_df.loc[roster_df["PLAYER"] == injured_name, "PLAYER_ID"].iloc[0]
            ) if injured_name else None

        stat_inj = st.selectbox("Stat", ["PTS","REB","AST","PRA"], index=0, key="stat_inj")

        min_games_without = st.slider(
            "Min games without to include",
            1, 15, 3, 1,
            key="min_g_without"
        )

        run_inj = st.button("Analyze Impact", key="run_inj")

    # --------------------------
    # RIGHT-SIDE OUTPUT
    # --------------------------
    with colR:
        if run_inj:
            if not injured_name or injured_id is None:
                st.warning("Select an injured player first.")
                st.stop()

            logs = get_league_player_logs(season_inj)
            team_logs = logs[logs["TEAM_ABBREVIATION"] == team_inj].copy()

            if team_logs.empty:
                st.warning("No logs for this team/season.")
                st.stop()

            # Build PRA when needed
            if stat_inj == "PRA":
                team_logs["PRA"] = (
                    team_logs["PTS"].fillna(0)
                    + team_logs["REB"].fillna(0)
                    + team_logs["AST"].fillna(0)
                )

            # Games WITH injured player
            inj_logs = team_logs[team_logs["PLAYER_ID"] == injured_id]
            if inj_logs.empty:
                st.warning(f"{injured_name} has no logged games this season.")
                st.stop()

            games_with = set(inj_logs["GAME_ID"].unique())
            all_games = set(team_logs["GAME_ID"].unique())
            games_without = all_games - games_with

            if not games_without:
                st.warning(f"No games where {injured_name} was OUT.")
                st.stop()

            # Teammates WITH
            with_df = team_logs[
                (team_logs["GAME_ID"].isin(games_with)) &
                (team_logs["PLAYER_ID"] != injured_id)
            ].copy()

            # Teammates WITHOUT
            without_df = team_logs[
                (team_logs["GAME_ID"].isin(games_without)) &
                (team_logs["PLAYER_ID"] != injured_id)
            ].copy()

            stat_col = stat_inj

            g_with = with_df.groupby("PLAYER_ID")[stat_col].mean()
            g_without = without_df.groupby("PLAYER_ID")[stat_col].mean()
            n_without = without_df.groupby("PLAYER_ID")["GAME_ID"].nunique()

            # Build rows
            rows = []
            idx = sorted(set(g_with.index) | set(g_without.index))

            for pid in idx:
                w = g_with.get(pid, np.nan)
                wo = g_without.get(pid, np.nan)
                nwo = int(n_without.get(pid, 0))

                if nwo < min_games_without:
                    continue

                delta = wo - w
                name = roster_df.loc[roster_df["PLAYER_ID"] == pid, "PLAYER"]
                name = name.iloc[0] if not name.empty else str(pid)

                rows.append({
                    "Player": name,
                    f"{stat_col} w/ {injured_name}": w,
                    f"{stat_col} w/o {injured_name}": wo,
                    "Games w/o": nwo,
                    "Delta": delta
                })

            if not rows:
                st.warning("No players met the filters.")
                st.stop()

            impact_df = pd.DataFrame(rows)
            impact_df = impact_df.sort_values("Delta", ascending=False).reset_index(drop=True)

            # --------------------------
            # BUILD BEAUTIFUL HTML TABLE (NO INDEX)
            # --------------------------

            # Color delta cells
            def apply_delta_color(val):
                if val > 0:
                    return "color:#7CFCBE; font-weight:700;"
                elif val < 0:
                    return "color:#FF6B6B; font-weight:700;"
                else:
                    return "color:#e5e7eb;"

            html = """
            <style>
                table.custom-table {
                    border-collapse: collapse;
                    width: 100%;
                    border-radius: 10px;
                    overflow: hidden;
                    margin-top: 10px;
                }
                table.custom-table th {
                    background-color: #1f2125;
                    color: #f9fafb;
                    padding: 10px;
                    font-weight: 700;
                    font-size: 0.9rem;
                    border-bottom: 1px solid #333;
                }
                table.custom-table td {
                    background-color: #131417;
                    color: #e5e7eb;
                    padding: 8px 10px;
                    border-bottom: 1px solid #2a2d31;
                    font-size: 0.9rem;
                }
            </style>
            <table class="custom-table">
                <thead>
                    <tr>
            """

            # Build header row
            for col in impact_df.columns:
                html += f"<th>{col}</th>"
            html += "</tr></thead><tbody>"

            # Build rows
            for _, row in impact_df.iterrows():
                html += "<tr>"
                for col in impact_df.columns:
                    val = row[col]
                    if col == "Delta":
                        style = apply_delta_color(val)
                        html += f"<td style='{style}'>{val:+.1f}</td>"
                    elif isinstance(val, float):
                        html += f"<td>{val:.1f}</td>"
                    else:
                        html += f"<td>{val}</td>"
                html += "</tr>"

            html += "</tbody></table>"

            st.caption(
                f"Positive Delta = player gains production when **{injured_name}** is OUT."
            )
            st.markdown(html, unsafe_allow_html=True)


            
# =========================
# TAB 5: HOT MATCHUPS (Team defensive averages)
# =========================
from nba_api.stats.endpoints import leaguegamelog
from datetime import datetime
import matplotlib.colors as mcolors

@st.cache_data(show_spinner=False)
def get_current_season_str():
    now = datetime.now()
    year = now.year if now.month >= 8 else now.year - 1
    return f"{year}-{str(year+1)[-2:]}"

@st.cache_data(show_spinner=True)
def load_team_logs(season: str) -> pd.DataFrame:
    """Fetch team-level game logs (one row per team per game)."""
    df = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
        timeout=60
    ).get_data_frames()[0]

    for k in ["PTS", "REB", "AST", "FG3M"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    df["OPP"] = (
        df["MATCHUP"].astype(str)
        .str.extract(r"vs\. (\w+)|@ (\w+)", expand=True)
        .bfill(axis=1).iloc[:, 0]
    )
    return df

def get_team_color(team_abbr):
    color_map = {
        "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#000000", "CHA": "#1D1160", "CHI": "#CE1141",
        "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240", "DET": "#C8102E", "GSW": "#1D428A",
        "HOU": "#CE1141", "IND": "#002D62", "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9",
        "MIA": "#98002E", "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#0C2340", "NYK": "#006BB6",
        "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#E56020", "POR": "#E03A3E",
        "SAC": "#5A2D81", "SAS": "#C4CED4", "TOR": "#CE1141", "UTA": "#002B5C", "WAS": "#002B5C"
    }
    return color_map.get(team_abbr, "#999999")

def soft_bg(hex_color, opacity=0.15):
    try:
        rgba = mcolors.to_rgba(hex_color, opacity)
        return mcolors.to_hex(rgba)
    except Exception:
        return "#222222"

# integrate this tab with main: 
# tab_builder, tab_breakeven, tab_matchups = st.tabs(["🧮 Parlay Builder", "🧷 Breakeven", "📈 Hot Matchups"])

with tab_matchups:
    st.subheader("📈 Hot Matchups — Team Defensive Averages (Per Game)")
    st.caption("Based on NBA team game logs. Sorted from weakest (top) to strongest (bottom) defense.")

    season = get_current_season_str()
    df = load_team_logs(season)
    if df.empty:
        st.warning("No data yet for this season.")
        st.stop()

    stats = ["PTS", "REB", "AST", "FG3M"]
    cols = st.columns(len(stats))

    for i, stat in enumerate(stats):
        allowed = (
            df.groupby("OPP", as_index=False)[stat]
            .mean()
            .rename(columns={stat: f"{stat}_ALLOWED_PER_GAME"})
        )
        allowed = allowed.sort_values(f"{stat}_ALLOWED_PER_GAME", ascending=False)
        with cols[i]:
            st.markdown(f"### {stat} Allowed / Game")
            for _, row in allowed.iterrows():
                team = row["OPP"]
                val = row[f"{stat}_ALLOWED_PER_GAME"]
                color = get_team_color(team)
                bg = soft_bg(color, 0.18)
                border_color = color + "99"
                st.markdown(
                    f"""
                    <div style='display:flex;align-items:center;justify-content:space-between;
                                margin-bottom:5px;padding:6px 12px;border-radius:8px;
                                background-color:{bg};
                                border:1px solid {border_color};'>
                        <span style='font-weight:700;color:#FFFFFF;font-size:1rem;
                                     text-shadow:0 0 6px {color}99;'>{team}</span>
                        <span style='font-size:0.95rem;color:#FFFFFF;font-weight:600;
                                     text-shadow:0 0 6px {color}66;'>{val:.1f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.divider()
    st.caption(f"Season {season} • Source: NBA Stats API • Regular-season team logs (per-game averages)")


