# nba_app.py
import streamlit as st
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nba_api.stats.static import players, teams as teams_static
from nba_api.stats.endpoints import (
    playergamelog, leaguegamelog, commonteamroster, leaguedashplayerstats,
    teamgamelog, boxscoretraditionalv2, playbyplayv2
)
from rapidfuzz import process
import requests
from datetime import datetime, timedelta, date
import pytz
import base64
from io import BytesIO
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False
@st.cache_data
def get_team_abbrev_to_id():
    return {t['abbreviation']: t['id'] for t in teams_static.get_teams()}

team_abbrev_to_id = get_team_abbrev_to_id()
TEAM_LOGOS = {
    "ATL": "https://a.espncdn.com/i/teamlogos/nba/500/atl.png",
    "BOS": "https://a.espncdn.com/i/teamlogos/nba/500/bos.png",
    "BKN": "https://a.espncdn.com/i/teamlogos/nba/500/bkn.png",
    "CHA": "https://a.espncdn.com/i/teamlogos/nba/500/cha.png",
    "CHI": "https://a.espncdn.com/i/teamlogos/nba/500/chi.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/nba/500/cle.png",
    "DAL": "https://a.espncdn.com/i/teamlogos/nba/500/dal.png",
    "DEN": "https://a.espncdn.com/i/teamlogos/nba/500/den.png",
    "DET": "https://a.espncdn.com/i/teamlogos/nba/500/det.png",
    "GSW": "https://a.espncdn.com/i/teamlogos/nba/500/gs.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/nba/500/hou.png",
    "IND": "https://a.espncdn.com/i/teamlogos/nba/500/ind.png",
    "LAC": "https://a.espncdn.com/i/teamlogos/nba/500/lac.png",
    "LAL": "https://a.espncdn.com/i/teamlogos/nba/500/lal.png",
    "MEM": "https://a.espncdn.com/i/teamlogos/nba/500/mem.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/nba/500/mia.png",
    "MIL": "https://a.espncdn.com/i/teamlogos/nba/500/mil.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/nba/500/min.png",
    "NOP": "https://a.espncdn.com/i/teamlogos/nba/500/no.png",
    "NYK": "https://a.espncdn.com/i/teamlogos/nba/500/ny.png",
    "OKC": "https://a.espncdn.com/i/teamlogos/nba/500/okc.png",
    "ORL": "https://a.espncdn.com/i/teamlogos/nba/500/orl.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/nba/500/phi.png",
    "PHX": "https://a.espncdn.com/i/teamlogos/nba/500/phx.png",
    "POR": "https://a.espncdn.com/i/teamlogos/nba/500/por.png",
    "SAC": "https://a.espncdn.com/i/teamlogos/nba/500/sac.png",
    "SAS": "https://a.espncdn.com/i/teamlogos/nba/500/sa.png",
    "TOR": "https://a.espncdn.com/i/teamlogos/nba/500/tor.png",
    "UTA": "https://a.espncdn.com/i/teamlogos/nba/500/utah.png",
    "WAS": "https://a.espncdn.com/i/teamlogos/nba/500/wsh.png"
}
ABBREV_MAP = {
    "GS": "GSW", "NO": "NOP", "UT": "UTA", "SA": "SAS", "LA": "LAL", "NY": "NYK", "WSH": "WAS",
    "GSW": "GSW", "NOP": "NOP", "UTA": "UTA", "SAS": "SAS", "LAL": "LAL", "NYK": "NYK", "WAS": "WAS"
}
TEAM_COLORS = {
    "ATL": "#E03A3E","BOS": "#007A33","BKN": "#000000","CHA": "#1D1160","CHI": "#CE1141",
    "CLE": "#860038","DAL": "#00538C","DEN": "#0E2240","DET": "#C8102E","GSW": "#1D428A",
    "HOU": "#CE1141","IND": "#002D62","LAC": "#C8102E","LAL": "#552583","MEM": "#5D76A9",
    "MIA": "#98002E","MIL": "#00471B","MIN": "#0C2340","NOP": "#0C2340","NYK": "#F58426",
    "OKC": "#007AC1","ORL": "#0077C0","PHI": "#006BB6","PHX": "#1D1160","POR": "#E03A3E",
    "SAC": "#5A2D81","SAS": "#C4CED4","TOR": "#CE1141","UTA": "#002B5C","WAS": "#002B5C"
}
TEAM_ABBRS = sorted(TEAM_COLORS.keys())
STAT_LABELS = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists", "STL": "Steals", "BLK": "Blocks",
    "FG3M": "3PM", "DOUBDOUB": "Doub Doub", "TRIPDOUB": "Trip Doub", "P+R": "P+R", "P+A": "P+A",
    "R+A": "R+A", "PRA": "PRA"
}
STAT_TOKENS = {
    "P": "PTS", "PTS": "PTS", "POINTS": "PTS", "R": "REB", "REB": "REB", "REBOUNDS": "REB",
    "A": "AST", "AST": "AST", "ASSISTS": "AST", "STL": "STL", "STEALS": "STL", "BLK": "BLK",
    "BLOCKS": "BLK", "3PM": "FG3M", "FG3M": "FG3M", "THREES": "FG3M", "DOUBDOUB": "DOUBDOUB",
    "DOUBLEDOUBLE": "DOUBDOUB", "DOUB": "DOUBDOUB", "DD": "DOUBDOUB", "TRIPDOUB": "TRIPDOUB",
    "TRIPLEDOUBLE": "TRIPDOUB", "TD": "TRIPDOUB", "P+R": "P+R", "PR": "P+R", "P+A": "P+A",
    "PA": "P+A", "R+A": "R+A", "RA": "R+A", "PRA": "PRA"
}
NBA_CUP_DATES = pd.to_datetime([
    "2024-11-12", "2024-11-15", "2024-11-19", "2024-11-22", "2024-11-26", "2024-11-29",
    "2024-12-03", "2024-12-10", "2024-12-11", "2024-12-14", "2024-12-17",
    "2025-10-31", "2025-11-07", "2025-11-14", "2025-11-21", "2025-11-25", "2025-11-26",
    "2025-11-28", "2025-12-09", "2025-12-10", "2025-12-13", "2025-12-16"
])
NORMALIZED_POS = {"PG": "PG", "G": "PG", "SG": "SG", "SF": "SF", "PF": "PF", "F": "PF", "C": "C"}
POSITION_MAP = {}
def get_espn_scoreboard(date):
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}"
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None
def render_espn_banner(scoreboard):
    if not scoreboard or "events" not in scoreboard:
        st.warning("No games found for this date.")
        return
    st.markdown("""
        <style>
        h1, .main-title {font-size: 50px !important; font-weight: 900 !important; margin-bottom: 6px !important;}
        .espn-banner-container {display: flex; overflow-x: auto; white-space: nowrap; padding: 6px 0 !important; gap: 10px !important; border-bottom: 1px solid #333;}
        .espn-game-card {flex: 0 0 auto; display: flex; flex-direction: column; align-items: center; background: #1e1e1e; border-radius: 8px !important; border: 1px solid #333; padding: 8px 12px !important; min-width: 170px !important; max-width: 170px !important; gap: 4px !important;}
        .espn-time {font-size: 11px !important; color: #ccc; text-align: center; margin-bottom: 4px !important; line-height: 1.1;}
        .espn-time .tv {font-size: 9px !important; color: #ff9900; margin-top: 1px;}
        .espn-matchup {display: flex; align-items: center; justify-content: space-between; width: 100%; margin-bottom: 2px !important;}
        .espn-team img {height: 26px !important; width: 26px !important; margin-bottom: 2px !important;}
        .espn-team {flex: 1; display: flex; flex-direction: column; align-items: center; gap: 1px !important;}
        .espn-team-abbr {font-size: 14px !important; font-weight: 700 !important; margin-bottom: 1px !important;}
        .espn-record {font-size: 11px !important; color: #888; line-height: 1.05;}
        .espn-at {font-size: 14px !important; font-weight: 700 !important; margin: 0 6px !important; white-space: nowrap;}
        </style>
        """, unsafe_allow_html=True)
    html = '<div class="espn-banner-container">'
    events = scoreboard["events"]
    for i, game in enumerate(events):
        try:
            comp = game["competitions"][0]
            status = game["status"]["type"]["shortDetail"]
            competitors = comp["competitors"]
            if len(competitors) != 2: raise ValueError("Unexpected number of competitors")
            away_team, home_team = None, None
            for t in competitors:
                ha = t.get("homeAway", None)
                team_dict = t["team"]
                abbr = team_dict.get("abbreviation", "TBD")
                logo = team_dict.get("logo", f"https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{abbr.lower()}.png")
                record = t["records"][0].get("summary", "") if t.get("records") and len(t["records"]) > 0 else ""
                team_info = (abbr, logo, record)
                if ha == "away": away_team = team_info
                elif ha == "home": home_team = team_info
                else: away_team = team_info if away_team is None else (home_team := team_info)
            if away_team is None or home_team is None: raise ValueError("Could not determine home/away")
            away_abbr, away_logo, away_record = away_team
            home_abbr, home_logo, home_record = home_team
            try:
                start_time_str = game["date"].replace("Z", "+00:00")
                start_time = datetime.fromisoformat(start_time_str)
                est = start_time.astimezone(pytz.timezone("US/Eastern"))
                time_str = est.strftime("%-I:%M %p ET")
            except: time_str = status or "TBD"
            if status.lower() in ["final", "in progress", "live"]: time_str = status.upper()
            tv = ""
            if "broadcasts" in comp and len(comp["broadcasts"]) > 0:
                tv_networks = [n for b in comp["broadcasts"] for n in b.get("names", [])]
                if tv_networks: tv = f'<div class="tv">{", ".join(tv_networks[:2])}</div>'
            snippet = textwrap.dedent(f"""
                <div class="espn-game-card">
                    <div class="espn-time">{time_str}{tv}</div>
                    <div class="espn-matchup">
                        <div class="espn-team">
                            <img src="{away_logo}" alt="{away_abbr}" onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{away_abbr.lower()}.png'">
                            <div class="espn-team-abbr">{away_abbr}</div>
                            <div class="espn-record">{away_record}</div>
                        </div>
                        <div class="espn-at">@</div>
                        <div class="espn-team">
                            <img src="{home_logo}" alt="{home_abbr}" onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{home_abbr.lower()}.png'">
                            <div class="espn-team-abbr">{home_abbr}</div>
                            <div class="espn-record">{home_record}</div>
                        </div>
                    </div>
                </div>
            """).strip()
            html += snippet
        except Exception as e:
            st.error(f"Error rendering game {i+1} ({game.get('name', 'Unknown')}): {str(e)}")
            continue
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
def extract_games_from_scoreboard(scoreboard):
    games = []
    if not scoreboard or "events" not in scoreboard: return games
    for ev in scoreboard["events"]:
        try:
            comp = ev["competitions"][0]
            competitors = comp["competitors"]
            if len(competitors) != 2: raise ValueError("Unexpected number of competitors")
            away_abbr, home_abbr = "", ""
            for t in competitors:
                ha = t.get("homeAway", None)
                abbr = t["team"].get("abbreviation", "")
                if ha == "away": away_abbr = abbr
                elif ha == "home": home_abbr = abbr
                else: away_abbr = abbr if away_abbr == "" else (home_abbr := abbr)
            status = ev.get("status", {}).get("type", {}).get("shortDetail", "")
            games.append({"home": home_abbr, "away": away_abbr, "status": status, "event_id": ev["id"]})
        except: continue
    return games
def get_player_headshot(player_id):
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"
st.set_page_config(page_icon="🏀", layout="wide")
today = datetime.now().date()
chosen_date = today.strftime("%Y%m%d")
@st.cache_data(ttl=300)
def fetch_scoreboard_cached(date_str):
    return get_espn_scoreboard(date_str)
scoreboard = fetch_scoreboard_cached(chosen_date)
render_espn_banner(scoreboard)
st.markdown("<hr style='border-color:#333;'>", unsafe_allow_html=True)
st.markdown("""
<style>
:root {--bg: #0e0f11; --text: #f9fafb; --muted: #9ca3af; --card: #1a1b1e; --border: #32353b; --accent: #00c896;}
body, .block-container {background: var(--bg); color: var(--text);}
.stButton > button {background: var(--accent) !important; color: #0b1220 !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; padding: 8px 14px !important; font-size: 0.9rem !important; box-shadow: 0 6px 18px rgba(0,200,150,0.25) !important;}
.stExpander {border: 1px solid var(--border) !important; background: var(--card) !important; border-radius: 12px !important;}
.streamlit-expanderHeader {font-weight: 800 !important; color: var(--text) !important; font-size: 0.95rem !important;}
input, select, textarea {background: #202225 !important; color: var(--text) !important; border: 1px solid #3b3f45 !important; border-radius: 8px !important; font-size: 0.9rem !important; padding: 6px !important;}
.card {--pad-x: 20px; --pad-y: 18px; padding: var(--pad-y) var(--pad-x); border-radius: 14px; margin: 10px 0 20px 0; border: 1px solid var(--card-border); background: var(--card-bg); box-shadow: 0 0 14px rgba(0,0,0,0.25); width: 100%;}
.neutral {--card-bg:#222; --card-border:#777;}
.pos {--card-bg:#0b3d23; --card-border:#00FF99;}
.neg {--card-bg:#3d0b0b; --card-border:#FF5555;}
.card h2 {color:#fff; margin:0 0 6px 0; font-weight:800; font-size:1.05rem;}
.cond {color:#a9b1bb; font-size:0.9rem; margin: 2px 0 10px 0;}
.row {display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end; justify-content:space-between; margin: 6px 0 4px 0;}
.m {min-width:120px; flex:1;}
.lab {color:#cbd5e1; font-size:0.8rem; margin-bottom:2px;}
.val {color:#fff; font-size:1.1rem; font-weight:800; line-height:1.1;}
.chip {display:inline-block; margin-top:10px; padding:6px 12px; border-radius:999px; font-size:0.8rem; color:#a7f3d0; border:1px solid #16a34a33; background: transparent;}
table {border-collapse: collapse;}
thead th {border-bottom: 1px solid #374151 !important;}
tbody td, thead th {padding: 8px 10px !important;}
</style>
""", unsafe_allow_html=True)
@st.cache_data
def get_all_player_names():
    try:
        active = players.get_active_players()
        names = [p["full_name"] for p in active if p.get("full_name")]
        if names: return sorted(set(names))
    except: pass
    all_p = players.get_players()
    return sorted(set(p["full_name"] for p in all_p if p.get("full_name")))
PLAYER_LIST = get_all_player_names()
def best_player_match(query: str) -> str:
    q = (query or "").strip()
    if not q: return ""
    m = process.extractOne(q, PLAYER_LIST, score_cutoff=80)
    if m: return m[0]
    m = process.extractOne(q, PLAYER_LIST, score_cutoff=60)
    return m[0] if m else ""
def american_to_implied(odds: int | float | str):
    try: x = float(odds)
    except: return None
    if -99 < x < 100: return None
    return 100.0/(x+100.0) if x > 0 else abs(x)/(abs(x)+100.0)
def prob_to_american(p: float):
    if p <= 0 or p >= 1: return "N/A"
    dec = 1.0/p
    return f"+{int(round((dec-1)*100))}" if dec >= 2.0 else f"-{int(round(100/(dec-1)))}"
def prob_to_ml(p):
    if p <= 0 or p >= 1:
        return "N/A"
    dec = 1 / p
    if dec >= 2:
        return f"+{int((dec - 1) * 100)}"
    return f"-{int(100 / (dec - 1))}"
def fmt_half(x: float | int) -> str:
    try: return f"{float(x):.1f}".rstrip("0").rstrip(".")
    except: return str(x)
def parse_input_line(text: str):
    t = (text or "").strip()
    if not t: return None
    parts = t.replace("/", "+").split()
    dir_token = "Over"
    for token in parts:
        if token.upper() in ["O", "OVER"]: dir_token = "Over"; break
        if token.upper() in ["U", "UNDER"]: dir_token = "Under"; break
    thr = 10.5
    for token in parts:
        tok = token.replace("+", "")
        try:
            if any(c.isdigit() for c in tok) and ("." in tok or tok.isdigit()): thr = float(tok); break
        except: pass
    stat_code = "PTS"
    combined_aliases = ["P+R","P+A","R+A","PRA","PR","PA","RA"]
    for token in parts:
        up = token.upper()
        if up in combined_aliases: stat_code = STAT_TOKENS.get(up, up); break
    if not stat_code:
        for token in parts:
            up = token.upper()
            if up in STAT_TOKENS: stat_code = STAT_TOKENS[up]; break
    loc = "All"
    for token in parts:
        up = token.upper()
        if up in ["AWAY","A"]: loc = "Away"; break
        if up in ["HOME","H"]: loc = "Home Only"; break
    odds = -110
    for token in parts[::-1]:
        if token.startswith("+") or token.startswith("-"):
            try:
                o = int(token)
                if o <= -100 or o >= 100: odds = o; break
            except: continue
    banned = set(["O","OVER","U","UNDER","HOME","H","AWAY","A"] + list(STAT_TOKENS.keys()) + combined_aliases)
    name_tokens = [p for p in parts if (p.upper() not in banned and not p.replace(".", "", 1).lstrip("+-").isdigit())]
    player = best_player_match(" ".join(name_tokens).strip())
    return {"player": player, "dir": dir_token, "thr": float(thr), "stat": stat_code, "loc": loc, "range": "FULL", "odds": int(odds)}
def get_player_id(full_name: str):
    if not full_name: return None
    res = players.find_players_by_full_name(full_name)
    return res[0]["id"] if res else None
def to_minutes(val):
    try:
        s = str(val)
        if ":" in s: return int(s.split(":")[0])
        return int(float(s))
    except: return 0
@st.cache_data
def fetch_gamelog(player_id: int, seasons: list[str], include_playoffs: bool=False, only_playoffs: bool=False) -> pd.DataFrame:
    dfs = []
    for s in seasons:
        if not only_playoffs:
            try: dfs.append(playergamelog.PlayerGameLog(player_id=player_id, season=s, season_type_all_star="Regular Season").get_data_frames()[0])
            except: pass
        if include_playoffs or only_playoffs:
            try: dfs.append(playergamelog.PlayerGameLog(player_id=player_id, season=s, season_type_all_star="Playoffs").get_data_frames()[0])
            except: pass
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    for k in ["PTS","REB","AST","STL","BLK","FG3M"]:
        if k in df.columns: df[k] = pd.to_numeric(df[k], errors="coerce")
    df["MIN_NUM"] = df["MIN"].apply(to_minutes) if "MIN" in df.columns else 0
    df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"], errors="coerce") if "GAME_DATE" in df.columns else pd.Timestamp.now()
    return df
def compute_stat_series(df: pd.DataFrame, stat_code: str) -> pd.Series:
    s = pd.Series(dtype=float, index=df.index)
    if stat_code in ["PTS","REB","AST","STL","BLK","FG3M"]: s = df[stat_code].astype(float)
    elif stat_code == "P+R": s = (df["PTS"] + df["REB"]).astype(float)
    elif stat_code == "P+A": s = (df["PTS"] + df["AST"]).astype(float)
    elif stat_code == "R+A": s = (df["REB"] + df["AST"]).astype(float)
    elif stat_code == "PRA": s = (df["PTS"] + df["REB"] + df["AST"]).astype(float)
    elif stat_code == "DOUBDOUB":
        pts = (df["PTS"] >= 10).astype(int); reb = (df["REB"] >= 10).astype(int); ast = (df["AST"] >= 10).astype(int)
        s = ((pts + reb + ast) >= 2).astype(int)
    elif stat_code == "TRIPDOUB":
        pts = (df["PTS"] >= 10).astype(int); reb = (df["REB"] >= 10).astype(int); ast = (df["AST"] >= 10).astype(int)
        s = ((pts + reb + ast) >= 3).astype(int)
    else: s = df["PTS"].astype(float)
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
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png" if pid else None
def get_team_logo_from_df(df):
    try: return f"https://cdn.nba.com/logos/nba/{int(df['TEAM_ID'].iloc[0])}/global/L/logo.svg"
    except: return None
def breakeven_for_stat(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    total = len(s)
    if total == 0: return {"line": None, "over_prob": None, "under_prob": None, "over_odds": "N/A", "under_odds": "N/A"}
    lo = np.floor(s.min()) - 0.5; hi = np.ceil(s.max()) + 0.5
    candidates = np.arange(lo, hi + 0.001, 0.5)
    best_t, best_gap, best_over = None, 1.0, None
    for t in candidates:
        over = (s >= t).mean()
        gap = abs(over - 0.5)
        if gap < best_gap or best_t is None: best_t, best_gap, best_over = t, gap, over
    over_prob = float(best_over); under_prob = 1.0 - over_prob
    return {"line": float(best_t), "over_prob": over_prob, "under_prob": under_prob, "over_odds": prob_to_american(over_prob), "under_odds": prob_to_american(under_prob)}
@st.cache_data(show_spinner=False)
def get_current_season_str():
    now = datetime.now()
    year = now.year if now.month >= 8 else now.year - 1
    return f"{year}-{str(year+1)[-2:]}"
@st.cache_data(show_spinner=True)
def load_team_logs(season: str) -> pd.DataFrame:
    df = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season", player_or_team_abbreviation="T", timeout=60).get_data_frames()[0]
    for k in ["PTS", "REB", "AST", "FG3M"]:
        if k in df.columns: df[k] = pd.to_numeric(df[k], errors="coerce")
    df["OPP"] = df["MATCHUP"].astype(str).str.extract(r"vs\. (\w+)|@ (\w+)", expand=True).bfill(axis=1).iloc[:, 0]
    return df
def get_team_color(team_abbr):
    return TEAM_COLORS.get(team_abbr, "#999999")
def soft_bg(hex_color, opacity=0.15):
    try: return mcolors.to_hex(mcolors.to_rgba(hex_color, opacity))
    except: return "#222222"
def get_player_position(pid):
    pos = POSITION_MAP.get(pid)
    return NORMALIZED_POS.get(pos, "SG") if pos else "SG"
def get_positional_defense_data(season):
    logs = get_league_player_logs(season)
    if logs.empty: return pd.DataFrame(columns=["Team", "Pos", "PTS", "REB", "AST", "FG3M"])
    for col in ["PTS", "REB", "AST", "FG3M"]: logs[col] = pd.to_numeric(logs[col], errors="coerce").fillna(0)
    logs["POS"] = logs["PLAYER_ID"].apply(get_player_position)
    if "OPPONENT" not in logs.columns:
        logs["OPPONENT"] = logs["MATCHUP"].str[-3:]
    opp_col = "OPPONENT"
    grouped = logs.groupby([opp_col, "POS"]).agg({"PTS": "mean", "REB": "mean", "AST": "mean", "FG3M": "mean"}).reset_index()
    grouped.rename(columns={opp_col: "Team", "POS": "Pos"}, inplace=True)
    teams = grouped["Team"].unique(); positions = ["PG", "SG", "SF", "PF", "C"]
    full = []
    for team in teams:
        for pos in positions:
            sub = grouped[(grouped["Team"] == team) & (grouped["Pos"] == pos)]
            if len(sub) == 1: full.append(sub.iloc[0])
            else:
                team_avg = grouped[grouped["Team"] == team][["PTS", "REB", "AST", "FG3M"]].mean()
                full.append({"Team": team, "Pos": pos, "PTS": team_avg["PTS"], "REB": team_avg["REB"], "AST": team_avg["AST"], "FG3M": team_avg["FG3M"]})
    df = pd.DataFrame(full); df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df
def build_team_positional_defense(season):
    df = get_positional_defense_data(season); out = {}
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        sub = df[df["Pos"] == pos]
        for stat in ["PTS", "REB", "AST", "PRA", "FG3M"]: sub[f"{stat}_rank"] = sub[stat].rank(ascending=False, method="min")
        for _, row in sub.iterrows():
            team = row["Team"]
            if team not in out: out[team] = {}
            out[team][pos] = {f"{k}_allowed": row[k] if k.endswith("_allowed") else row[k] for k in ["PTS", "REB", "AST", "PRA", "FG3M"]} | {f"{k}_rank": row[f"{k}_rank"] for k in ["PTS", "REB", "AST", "PRA", "FG3M"]}
    return out
@st.cache_data(show_spinner=False)
def get_league_player_logs(season: str) -> pd.DataFrame:
    df = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season", player_or_team_abbreviation="P").get_data_frames()[0]
    for col in ["PTS", "REB", "AST", "STL", "BLK", "FG3M"]: df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else 0
    df["MIN_NUM"] = df["MIN"].apply(to_minutes) if "MIN" in df.columns else 0
    df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"], errors="coerce") if "GAME_DATE" in df.columns else pd.Timestamp.now()
    df["OPP"] = df["MATCHUP"].astype(str).str.extract(r"vs\. (\w+)|@ (\w+)", expand=True).bfill(axis=1).iloc[:, 0]
    return df
def get_team_defense_table(season: str) -> pd.DataFrame:
    df = get_league_player_logs(season)
    tmp = df.groupby(["OPP", "GAME_ID"])[["PTS", "REB", "AST", "FG3M"]].sum().reset_index()
    agg = tmp.groupby("OPP")[["PTS", "REB", "AST", "FG3M"]].mean().reset_index().rename(columns={"OPP": "Team", "PTS": "PTS_allowed", "REB": "REB_allowed", "AST": "AST_allowed", "FG3M": "FG3M_allowed"})
    return agg
@st.cache_data(show_spinner=False)
def get_team_roster(season: str, team_abbrev: str) -> pd.DataFrame:
    team_meta = [t for t in teams_static.get_teams() if t["abbreviation"] == team_abbrev]
    if not team_meta: return pd.DataFrame(columns=["PLAYER_ID", "PLAYER"])
    team_id = team_meta[0]["id"]
    roster_df = commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    return roster_df[["PLAYER_ID", "PLAYER"]]
def style_def_table(df: pd.DataFrame, stat_col: str):
    def row_style(row):
        team = row["Team"]; base = TEAM_COLORS.get(team, "#111827"); bg_team = lighten_hex(base, 0.4); bg_val = "#020617"
        return [f"background-color:{bg_team}; color:#f9fafb; font-weight:700; text-align:left;", f"background-color:{bg_val}; color:#f9fafb; text-align:right;"]
    return df.style.hide(axis="index").format({stat_col: "{:.1f}"}).apply(row_style, axis=1)
def monte_carlo_predictive(series: pd.Series, n_sims: int = 10000) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0: return np.array([])
    mean = float(np.mean(vals)); std = float(np.std(vals))
    if std == 0 or len(vals) == 1: return np.full(n_sims, mean)
    n = len(vals); bandwidth = 1.06 * std * n ** (-1/5)
    base = np.random.choice(vals, size=n_sims, replace=True); noise = np.random.normal(0, bandwidth, size=n_sims)
    draws = base + noise
    return np.clip(draws, 0, None)
def render_mc_result_card(player, direction, thr, stat_label, loc_text, last_n, p_hit, fair_odds, sb_odds, ev_pct):
    ev_str = "—" if ev_pct is None else f"{ev_pct:.2f}%"; hit_str = f"{p_hit*100:.1f}%"
    cls = "neutral" if ev_pct is None else ("pos" if ev_pct >= 0 else "neg")
    return f"""
<div class="card {cls}">
  <h2>🎲 Monte Carlo Result</h2>
  <div class="cond">{player} — {'O' if direction == 'Over' else 'U'} {thr} {stat_label} ({loc_text}, last {last_n} games)</div>
  <div class="row">
    <div class="m"><div class="lab">Sim Hit %</div><div class="val">{hit_str}</div></div>
    <div class="m"><div class="lab">Fair Odds</div><div class="val">{fair_odds}</div></div>
    <div class="m"><div class="lab">Book Odds</div><div class="val">{sb_odds}</div></div>
    <div class="m"><div class="lab">EV</div><div class="val">{ev_str}</div></div>
  </div>
  <div style="margin-top:10px;"><span class="chip">{('🔥 +EV (Monte Carlo)' if (ev_pct is not None and ev_pct >= 0) else '⚠️ Negative EV by simulation')}</span></div>
</div>
"""
def lighten_hex(color: str, factor: float = 0.35) -> str:
    color = color.lstrip("#"); r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    r = int(r + (255 - r) * factor); g = int(g + (255 - g) * factor); b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"
def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20: suffix = "th"
    else: suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
EDGE_THRESHOLDS = {"PTS": {"strong": 4.0, "mild": 2.0}, "REB": {"strong": 2.2, "mild": 1.0}, "AST": {"strong": 2.0, "mild": 1.0}, "PRA": {"strong": 6.0, "mild": 3.0}, "FG3M": {"strong": 1.0, "mild": 0.4}}
STAT_PREFIX_LABEL = {"PTS": "PTS", "REB": "REB", "AST": "AST", "PRA": "PRA", "FG3M": "3PM"}
def player_prefix(name: str, stat: str) -> str:
    tokens = name.split()
    last_name = tokens[-1] if len(tokens) == 1 else (tokens[-2] + " " + tokens[-1] if tokens[-1] in {"Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV"} else tokens[-1])
    label = STAT_PREFIX_LABEL.get(stat, stat)
    return f"{last_name.upper()} {label.upper()}"
def logistic_prob(z: float) -> float:
    p = 0.5 + 0.18 * z
    return float(np.clip(p, 0.05, 0.95))
@st.cache_data(show_spinner=False)
def load_enhanced_team_logs(season: str) -> pd.DataFrame:
    try: df = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season", player_or_team_abbreviation="T", timeout=60).get_data_frames()[0]
    except: return pd.DataFrame()
    if df.empty: return df
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    num_cols = ['FGA', 'FTA', 'OREB', 'TOV', 'PTS']
    for col in num_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) if col in df.columns else 0
    df["OPP"] = df["MATCHUP"].astype(str).str.extract(r"vs\. (\w+)|@ (\w+)", expand=True).bfill(axis=1).iloc[:, 0]
    df['poss'] = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV']
    def get_opp_stats(group):
        if len(group) < 2: group['opp_pts'] = np.nan; group['opp_poss'] = np.nan; return group
        teams = group['TEAM_ABBREVIATION'].unique()
        if len(teams) != 2: group['opp_pts'] = np.nan; group['opp_poss'] = np.nan; return group
        team_a, team_b = teams; pts_a = group[group['TEAM_ABBREVIATION'] == team_a]['PTS'].iloc[0]; poss_a = group[group['TEAM_ABBREVIATION'] == team_a]['poss'].iloc[0]
        pts_b = group[group['TEAM_ABBREVIATION'] == team_b]['PTS'].iloc[0]; poss_b = group[group['TEAM_ABBREVIATION'] == team_b]['poss'].iloc[0]
        group.loc[group['TEAM_ABBREVIATION'] == team_a, 'opp_pts'] = pts_b; group.loc[group['TEAM_ABBREVIATION'] == team_a, 'opp_poss'] = poss_b
        group.loc[group['TEAM_ABBREVIATION'] == team_b, 'opp_pts'] = pts_a; group.loc[group['TEAM_ABBREVIATION'] == team_b, 'opp_poss'] = poss_a
        return group
    df = df.groupby('GAME_ID', group_keys=False).apply(get_opp_stats).reset_index(drop=True)
    df['ortg'] = np.where(df['poss'] > 0, df['PTS'] / df['poss'] * 100, 0)
    df['drtg'] = np.where(df['opp_poss'] > 0, df['opp_pts'] / df['opp_poss'] * 100, 0)
    df['nrtg'] = df['ortg'] - df['drtg']
    return df
def get_rest_days(team: str, logs: pd.DataFrame, game_date: date) -> int:
    team_games = logs[logs['TEAM_ABBREVIATION'] == team].copy()
    if team_games.empty: return 0
    past_games = team_games[team_games['GAME_DATE'] < pd.Timestamp(game_date)]
    if past_games.empty: return 99
    last_game_date = past_games['GAME_DATE'].max().date()
    return (game_date - last_game_date).days
@st.cache_data(show_spinner=False)
def train_margin_model(season: str):
    if not XGB_AVAILABLE: return None
    prev_season = f"{int(season.split('-')[0]) - 1}-{season.split('-')[1]}"
    logs = load_enhanced_team_logs(prev_season)
    if logs.empty: return None
    team_ortg = logs.groupby("TEAM_ABBREVIATION")["ortg"].mean()
    team_drtg = logs.groupby("TEAM_ABBREVIATION")["drtg"].mean()
    team_poss = logs.groupby("TEAM_ABBREVIATION")["poss"].mean()
    games = logs.groupby('GAME_ID')
    features_list, margins = [], []
    hca_val = 2.7
    for gid, group in games:
        if len(group) != 2: continue
        teams = group['TEAM_ABBREVIATION'].unique()
        if len(teams) != 2: continue
        matchup_home = group.iloc[0]['MATCHUP']
        home = group.iloc[0]['TEAM_ABBREVIATION'] if 'vs' in matchup_home else group.iloc[1]['TEAM_ABBREVIATION']
        away = group.iloc[1]['TEAM_ABBREVIATION'] if 'vs' in matchup_home else group.iloc[0]['TEAM_ABBREVIATION']
        home_row = group[group['TEAM_ABBREVIATION'] == home].iloc[0]
        away_row = group[group['TEAM_ABBREVIATION'] == away].iloc[0]
        actual_margin = home_row['PTS'] - away_row['PTS']
        game_date = home_row['GAME_DATE'].date()
        rest_home = get_rest_days(home, logs, game_date); rest_away = get_rest_days(away, logs, game_date); rest_d = rest_home - rest_away
        ortg_home_pre = float(team_ortg.get(home, 110.0)); ortg_away_pre = float(team_ortg.get(away, 110.0))
        drtg_home_pre = float(team_drtg.get(home, 110.0)); drtg_away_pre = float(team_drtg.get(away, 110.0))
        poss_avg = float((team_poss.get(home, 98.0) + team_poss.get(away, 98.0)) / 2)
        ortg_diff = ortg_home_pre - ortg_away_pre; drtg_diff = drtg_home_pre - drtg_away_pre
        features = [ortg_diff, drtg_diff, poss_avg, 0.0, 0.0, hca_val, rest_d]
        features_list.append(features); margins.append(actual_margin)
    if len(margins) < 50: return None
    X = np.array(features_list); y = np.array(margins)
    try:
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        return model
    except ImportError:
        return None
@st.cache_data(show_spinner=False)
def train_total_model(season: str):
    if not XGB_AVAILABLE: return None
    prev_season = f"{int(season.split('-')[0]) - 1}-{season.split('-')[1]}"
    logs = load_enhanced_team_logs(prev_season)
    if logs.empty: return None
    team_ortg = logs.groupby("TEAM_ABBREVIATION")["ortg"].mean()
    team_drtg = logs.groupby("TEAM_ABBREVIATION")["drtg"].mean()
    team_poss = logs.groupby("TEAM_ABBREVIATION")["poss"].mean()
    games = logs.groupby('GAME_ID')
    features_list, totals = [], []
    for gid, group in games:
        if len(group) != 2: continue
        teams = group['TEAM_ABBREVIATION'].unique()
        if len(teams) != 2: continue
        matchup_home = group.iloc[0]['MATCHUP']
        home = group.iloc[0]['TEAM_ABBREVIATION'] if 'vs' in matchup_home else group.iloc[1]['TEAM_ABBREVIATION']
        away = group.iloc[1]['TEAM_ABBREVIATION'] if 'vs' in matchup_home else group.iloc[0]['TEAM_ABBREVIATION']
        home_row = group[group['TEAM_ABBREVIATION'] == home].iloc[0]
        away_row = group[group['TEAM_ABBREVIATION'] == away].iloc[0]
        actual_total = home_row['PTS'] + away_row['PTS']
        game_date = home_row['GAME_DATE'].date()
        rest_home = get_rest_days(home, logs, game_date); rest_away = get_rest_days(away, logs, game_date); rest_d = rest_home - rest_away
        ortg_home_pre = float(team_ortg.get(home, 110.0)); ortg_away_pre = float(team_ortg.get(away, 110.0))
        drtg_home_pre = float(team_drtg.get(home, 110.0)); drtg_away_pre = float(team_drtg.get(away, 110.0))
        poss_avg = float((team_poss.get(home, 98.0) + team_poss.get(away, 98.0)) / 2)
        features = [ortg_home_pre, ortg_away_pre, drtg_home_pre, drtg_away_pre, poss_avg, 0.0, 0.0, rest_d]
        features_list.append(features); totals.append(actual_total)
    if len(totals) < 50: return None
    X = np.array(features_list); y = np.array(totals)
    try:
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        return model
    except ImportError:
        return None
@st.cache_data(show_spinner=False)
def get_espn_game_summary(event_id: str) -> dict:
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200: return r.json()
    except: pass
    return {}
@st.cache_data(show_spinner=False)
def load_player_impact(season: str) -> pd.DataFrame:
    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(season=season, season_type_all_star="Regular Season", per_mode_detailed="PerGame", measure_type_detailed_defense="Advanced", timeout=60).get_data_frames()[0]
    except: return pd.DataFrame()
    if df.empty: return df
    df["PLAYER_NAME_UPPER"] = df["PLAYER_NAME"].str.upper()
    mins = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)
    net = pd.to_numeric(df.get("NET_RATING", 0), errors="coerce").fillna(0)
    minute_factor = mins / 24.0; net_factor = 1 + (net / 20.0)
    impact_score = (minute_factor * net_factor).clip(lower=0.2, upper=2.0)
    df["IMPACT_SCORE"] = impact_score
    return df[["PLAYER_NAME_UPPER", "TEAM_ABBREVIATION", "IMPACT_SCORE", "MIN", "NET_RATING", "GP"]]
@st.cache_data(show_spinner=False)
def load_league_player_logs_upper(season: str) -> pd.DataFrame:
    logs = get_league_player_logs(season)
    if logs is None or logs.empty: return pd.DataFrame()
    logs = logs.copy()
    logs["PLAYER_NAME_UPPER"] = logs["PLAYER_NAME"].str.upper() if "PLAYER_NAME" in logs.columns else ""
    return logs
def extract_injuries_from_summary(summary: dict, home_abbr: str, away_abbr: str, game_date: date, player_impacts: pd.DataFrame | None, logs_team: pd.DataFrame, league_logs_upper: pd.DataFrame) -> tuple:
    inj_home, inj_away = [], []
    adjust_home_nrtg, adjust_away_nrtg = 0.0, 0.0
    adjust_home_ortg, adjust_away_ortg = 0.0, 0.0
    adjust_home_drtg, adjust_away_drtg = 0.0, 0.0
    injuries_top = summary.get("injuries", []); abbr_map = ABBREV_MAP
    if player_impacts is not None and not player_impacts.empty:
        impact_lookup = player_impacts.set_index("PLAYER_NAME_UPPER")["IMPACT_SCORE"].to_dict()
        gp_lookup = player_impacts.set_index("PLAYER_NAME_UPPER")["GP"].to_dict()
    else: impact_lookup, gp_lookup = {}, {}
    impact_scale_nrtg = 1.5; matchup_factor_home = matchup_factor_away = 1.0
    team_grouped = dict(tuple(logs_team.groupby("TEAM_ABBREVIATION"))) if not logs_team.empty else {}
    def get_player_impact(name_upper: str, team_abbr: str) -> float:
        if not impact_lookup: return 1.0
        gp = gp_lookup.get(name_upper, 0)
        team_games = team_grouped.get(team_abbr, pd.DataFrame())
        team_total_games = team_games["GAME_ID"].nunique() if not team_games.empty else 0
        min_games_threshold = max(1, int(0.1 * team_total_games))
        if gp < min_games_threshold: return 0.0
        return float(impact_lookup.get(name_upper, 1.0))
    def compute_team_ortg_delta(team_abbr: str, player_name_upper: str) -> float:
        if logs_team.empty or league_logs_upper.empty: return 0.0
        team_games = team_grouped.get(team_abbr, pd.DataFrame())
        if team_games.empty: return 0.0
        team_total_games = team_games["GAME_ID"].nunique()
        min_games_threshold = max(1, int(0.1 * team_total_games))
        plog = league_logs_upper[(league_logs_upper["TEAM_ABBREVIATION"] == team_abbr) & (league_logs_upper["PLAYER_NAME_UPPER"] == player_name_upper)]
        if plog.empty: return 0.0
        games_with = set(plog["GAME_ID"].unique()); n_with = len(games_with)
        if n_with < min_games_threshold: return 0.0
        with_df = team_games[team_games["GAME_ID"].isin(games_with)]; without_df = team_games[~team_games["GAME_ID"].isin(games_with)]
        n_without = without_df["GAME_ID"].nunique()
        if n_without < 3: return 0.0
        ortg_with = with_df["ortg"].mean(); ortg_without = without_df["ortg"].mean()
        raw_delta = ortg_without - ortg_with
        slack_factor = 1.0 - (0.5 * (1 - min(n_without / 15.0, 1.0))); weight = min(n_without / 15.0, 1.0)
        return float(raw_delta * weight * slack_factor)
    def compute_team_drtg_delta(team_abbr: str, player_name_upper: str) -> float:
        if logs_team.empty or league_logs_upper.empty: return 0.0
        team_games = team_grouped.get(team_abbr, pd.DataFrame())
        if team_games.empty: return 0.0
        team_total_games = team_games["GAME_ID"].nunique()
        min_games_threshold = max(1, int(0.1 * team_total_games))
        plog = league_logs_upper[(league_logs_upper["TEAM_ABBREVIATION"] == team_abbr) & (league_logs_upper["PLAYER_NAME_UPPER"] == player_name_upper)]
        if plog.empty: return 0.0
        games_with = set(plog["GAME_ID"].unique()); n_with = len(games_with)
        if n_with < min_games_threshold: return 0.0
        with_df = team_games[team_games["GAME_ID"].isin(games_with)]; without_df = team_games[~team_games["GAME_ID"].isin(games_with)]
        n_without = without_df["GAME_ID"].nunique()
        if n_without < 3: return 0.0
        drtg_with = with_df["drtg"].mean(); drtg_without = without_df["drtg"].mean()
        raw_delta = drtg_without - drtg_with
        slack_factor = 1.0 - (0.5 * (1 - min(n_without / 15.0, 1.0))); weight = min(n_without / 15.0, 1.0)
        return float(raw_delta * weight * slack_factor)
    for team_inj_item in injuries_top:
        team_abbr_api = team_inj_item["team"]["abbreviation"]; team_abbr = abbr_map.get(team_abbr_api, team_abbr_api)
        team_inj_list, team_nrtg_adj, team_ortg_adj, team_drtg_adj = [], 0.0, 0.0, 0.0
        rest_multiplier = 1.0; matchup_factor = matchup_factor_home if team_abbr == home_abbr else matchup_factor_away
        for inj in team_inj_item.get("injuries", []):
            athlete = inj["athlete"]; player_name = athlete["displayName"]; player_name_upper = player_name.upper()
            details = inj.get("details", {}); fantasy_status = details.get("fantasyStatus", {}).get("description", ""); injury_type = details.get("type", "")
            detail = details.get("detail", ""); full_injury = f"{injury_type} ({detail})" if detail else injury_type
            return_date_str = details.get("returnDate"); return_date = None
            if return_date_str:
                try: return_date = date.fromisoformat(return_date_str)
                except: pass
            is_out = return_date is None or return_date > game_date
            if not is_out: continue
            status_lower = fantasy_status.lower()
            status_weight = 1.0 if "out" in status_lower else (0.5 if any(term in status_lower for term in ["day-to-day", "questionable", "gtd"]) else 0.75)
            player_importance = get_player_impact(player_name_upper, team_abbr)
            base_adjust = -impact_scale_nrtg * player_importance * status_weight
            team_nrtg_adj += base_adjust * rest_multiplier
            ortg_delta = compute_team_ortg_delta(team_abbr, player_name_upper); team_ortg_adj += ortg_delta * status_weight
            drtg_delta = compute_team_drtg_delta(team_abbr, player_name_upper); team_drtg_adj += drtg_delta * status_weight
            team_nrtg_adj += (ortg_delta - drtg_delta) * status_weight * rest_multiplier
            team_inj_list.append({"name": player_name, "status": fantasy_status, "injury": full_injury, "return_date": return_date, "impact_score": round(player_importance, 2), "status_weight": status_weight, "ortg_delta": round(ortg_delta, 2), "drtg_delta": round(drtg_delta, 2)})
        team_nrtg_adj = float(np.clip(team_nrtg_adj, -12, 12)); team_ortg_adj = float(np.clip(team_ortg_adj, -10, 10)); team_drtg_adj = float(np.clip(team_drtg_adj, -10, 10))
        if team_abbr == home_abbr:
            inj_home = team_inj_list; adjust_home_nrtg = team_nrtg_adj; adjust_home_ortg = team_ortg_adj; adjust_home_drtg = team_drtg_adj
        elif team_abbr == away_abbr:
            inj_away = team_inj_list; adjust_away_nrtg = team_nrtg_adj; adjust_away_ortg = team_ortg_adj; adjust_away_drtg = team_drtg_adj
    return inj_home, inj_away, adjust_home_nrtg, adjust_away_nrtg, adjust_home_ortg, adjust_away_ortg, adjust_home_drtg, adjust_away_drtg
def team_html(team):
    team_key = ABBREV_MAP.get(team, team); logo = TEAM_LOGOS.get(team_key, "")
    return f"<span style=\"display:inline-flex; align-items:center; gap:6px; vertical-align:middle;\"><img src=\"{logo}\" width=\"20\" style=\"border-radius:3px; vertical-align:middle;\" /><span style=\"vertical-align:middle;\">{team}</span></span>"
tab_builder, tab_breakeven, tab_mc, tab_injury, tab_me, tab_matchups, tab_qh, tab_ml = st.tabs(["🧮 Parlay Builder", "🧷 Breakeven", "🎲 Monte Carlo Sim", "🩹 Injury Impact", "🔥 Matchup Exploiter","🛡️ Team Defense", "💨 Quick Hits", "💵 ML, Spread, & Totals"])
with tab_builder:
    fc1, fc2, fc3 = st.columns([1.2, 1, 1])
    with fc1:
        seasons = st.multiselect("Seasons", ["2025-26","2024-25","2023-24","2022-23"], default=["2025-26"], key="seasons_builder")
        include_playoffs = st.checkbox("Include Playoffs", value=False, key="pb_playoffs")
        only_playoffs = st.checkbox("Only Playoffs", value=False, key="pb_only_playoffs")
        nba_cup_only = st.checkbox("NBA Cup Games Only", key="cup_only_builder")
    with fc2: min_minutes = st.slider("Min Minutes", 0, 40, 20, 1, key="min_minutes_builder")
    with fc3: last_n_games = st.slider("Last N Games", 5, 100, 20, 1, key="parlay_lastn")
    if "legs" not in st.session_state: st.session_state.legs = []
    if "awaiting_input" not in st.session_state: st.session_state.awaiting_input = True
    c1, c2 = st.columns([1,1])
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("+ Add Leg"):
            st.session_state.awaiting_input = True
    with c2:
        if st.button("− Remove Last Leg") and st.session_state.legs:
            st.session_state.legs.pop()
            st.rerun()
    st.write("**Input bet**")
    if st.session_state.legs:
        for i, leg in enumerate(st.session_state.legs):
            leg_no = i + 1; dir_short = "O" if leg["dir"] == "Over" else "U"
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
                    if st.button(f"Remove Leg {leg_no}", key=f"remove_{i}"):
                        st.session_state.legs.pop(i)
                        st.rerun()
    if st.session_state.awaiting_input:
        bet_text = st.text_input("Input bet", placeholder="Maxey O 24.5 P Away -110", key="freeform_input", label_visibility="collapsed")
        if bet_text.strip():
            parsed = parse_input_line(bet_text)
            if parsed and parsed["player"]: st.session_state.legs.append(parsed); st.session_state.awaiting_input = False; st.rerun()
    parlay_odds = 0
    if len(st.session_state.legs) > 1:
        st.subheader("🎯 Combined Parlay Odds")
        parlay_odds = st.number_input("Parlay Odds (+300, -150)", value=0, step=5, key="parlay_odds")
    if st.session_state.legs and st.button("Compute"):
        st.write("---")
        rows, probs_for_parlay = [], []
        plt.rcParams.update({"axes.facecolor": "#1e1f22", "figure.facecolor": "#1e1f22", "text.color": "#ffffff", "axes.labelcolor": "#e5e7eb", "xtick.color": "#e5e7eb", "ytick.color": "#e5e7eb", "grid.color": "#374151"})
        for leg in st.session_state.legs:
            name, pid = leg["player"], get_player_id(leg["player"])
            stat, direction, thr, loc, rng, odds = leg["stat"], leg["dir"], float(leg["thr"]), leg["loc"], leg.get("range","FULL"), int(leg["odds"])
            df = fetch_gamelog(pid, seasons, include_playoffs, only_playoffs) if pid else pd.DataFrame()
            if df.empty: st.warning(f"Could not compute for **{name}**"); continue
            d = df.copy(); d = d[d["MIN_NUM"] >= min_minutes]
            if loc == "Home Only": d = d[d["MATCHUP"].astype(str).str.contains("vs")]
            elif loc == "Away": d = d[d["MATCHUP"].astype(str).str.contains("@")]
            if nba_cup_only: d = d[d["GAME_DATE_DT"].isin(NBA_CUP_DATES)]
            d = d.sort_values("GAME_DATE_DT", ascending=False)
            if rng == "L10": d = d.head(10)
            elif rng == "L20": d = d.head(20)
            else: d = d.head(last_n_games)
            p, hits, total = leg_probability(d, stat, direction, thr)
            fair = prob_to_american(p); book_prob = american_to_implied(odds); ev = (p - book_prob) * 100.0 if book_prob is not None else None
            if p > 0: probs_for_parlay.append(p)
            rows.append({"name": name, "stat": stat, "thr": thr, "dir": direction, "loc": loc, "range": rng, "odds": odds, "p": p, "hits": hits, "total": total, "fair": fair, "book_prob": book_prob, "ev": ev, "df": d})
        combined_p = float(np.prod(probs_for_parlay)) if probs_for_parlay else 0.0
        combined_fair = prob_to_american(combined_p) if combined_p > 0 else "N/A"
        entered_prob = american_to_implied(parlay_odds); book_implied_prob = entered_prob * 100 if entered_prob else None
        parlay_ev = (combined_p - entered_prob) * 100.0 if entered_prob is not None else None
        cls = "neutral" if parlay_ev is None else ("pos" if parlay_ev >= 0 else "neg")
        combined_prob_disp = f"{combined_p*100:.2f}%" if combined_p > 0 else "—"
        parlay_ev_disp = "—" if parlay_ev is None else f"{parlay_ev:.2f}%"
        book_implied_str = "—" if book_implied_prob is None else f"{book_implied_prob:.2f}%"
        st.markdown(f"""
<div class="card {cls}">
  <h2>💥 Combined Parlay</h2>
  <div style="font-size:0.95rem;color:#9ca3af;margin-bottom:14px;">Includes all legs with your filters</div>
  <div class="row">
    <div class="m"><div class="lab">Model Parlay Probability</div><div class="val">{combined_prob_disp}</div></div>
    <div class="m"><div class="lab">Model Fair Odds</div><div class="val">{combined_fair}</div></div>
    <div class="m"><div class="lab">Entered Odds</div><div class="val">{parlay_odds}</div></div>
    <div class="m"><div class="lab">Book Implied</div><div class="val">{book_implied_str}</div></div>
    <div class="m"><div class="lab">Expected Value</div><div class="val">{parlay_ev_disp}</div></div>
  </div>
  <div style="margin-top:14px;"><span class="chip">{('🔥 +EV Parlay Detected' if (parlay_ev is not None and parlay_ev >= 0) else '⚠️ Negative EV Parlay')}</span></div>
</div>
""", unsafe_allow_html=True)
        for r in rows:
            pid = get_player_id(r["name"]); head = headshot_url(pid); logo = get_team_logo_from_df(r["df"])
            cls = "neutral" if r["ev"] is None else ("pos" if r["ev"] >= 0 else "neg")
            img_html = f'<img src="{head}" width="72" style="border-radius:12px;">' if head else ""
            logo_html = f'<img src="{logo}" width="38" style="margin-left:10px;">' if logo else ""
            stat_label = STAT_LABELS.get(r["stat"], r["stat"]); dir_word = "O" if r["dir"] == "Over" else "U"
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
  <div style="margin-top:10px;"><span class="chip">{('🔥 +EV Play Detected' if (r['ev'] is not None and r['ev'] >= 0) else '⚠️ Negative EV Play')}</span></div>
</div>
""", unsafe_allow_html=True)
            if r["stat"] not in ["DOUBDOUB","TRIPDOUB"]:
                ser = compute_stat_series(r["df"], r["stat"]).dropna().astype(int)
                value_counts = ser.value_counts().sort_index()
                data_min, data_max = int(ser.min()), int(ser.max())
                all_vals = list(range(data_min, data_max + 1)); counts = [value_counts.get(v, 0) for v in all_vals]
                label_count = max(1, (data_max - data_min) // 5)
                with st.expander("📊 Show Game Distribution", expanded=False):
                    fig, ax = plt.subplots(figsize=(6.5, 2.4)); fig.patch.set_facecolor("#1e1f22"); ax.set_facecolor("#1e1f22")
                    color = "#00c896" if (r["ev"] is not None and r["ev"] >= 0) else "#e05a5a"
                    ax.bar(all_vals, counts, color=color, alpha=0.75, edgecolor="#d1d5db", linewidth=0.5, width=0.9)
                    ax.axvline(r["thr"], color="white", linestyle="--", linewidth=1.4)
                    ax.set_xticks(all_vals); ax.set_xticklabels([str(v) if ((v - data_min) % label_count == 0 or v == r["thr"]) else "" for v in all_vals], color="#9ca3af", fontsize=8)
                    ax.tick_params(axis="y", colors="#9ca3af", labelsize=8)
                    for spine in ax.spines.values(): spine.set_edgecolor("#4b5563")
                    ax.grid(color="#374151", linestyle="--", linewidth=0.5, alpha=0.55); ax.set_axisbelow(True); ax.set_ylabel(""); ax.set_xlabel("")
                    st.pyplot(fig, use_container_width=True)
with tab_breakeven:
    st.subheader("🔎 Breakeven Finder")
    st.caption(f"See the breakeven (closest to even odds) value for each stat type")
    f1, f2 = st.columns([1.2, 1])
    with f1:
        seasons_b = st.multiselect("Seasons", ["2025-26","2024-25","2023-24","2022-23"], default=["2025-26"], key="seasons_breakeven")
        include_playoffs_b = st.checkbox("Include Playoffs", value=False, key="be_playoffs")
        only_playoffs_b = st.checkbox("Only Playoffs", value=False, key="be_only_playoffs")
        nba_cup_only_b = st.checkbox("NBA Cup Games Only", key="cup_only_break")
    cA, cB, cC, cD = st.columns([2,1,1,1])
    with cA: player_query = st.text_input("Player", placeholder="e.g., Stephen Curry", key="breakeven_player")
    with cB: last_n = st.slider("Last N", 5, 100, 20, 1, key="breakeven_lastn")
    with cC: min_min_b = st.slider("Min Min", 0, 40, 20, 1, key="breakeven_minmin")
    with cD: loc_choice = st.selectbox("Location", ["All","Home Only","Away"], index=0, key="breakeven_loc")
    do_search = st.button("Search", key="breakeven_search")
    if do_search:
        player_name = best_player_match(player_query)
        if not player_name: st.warning("Could not match player.")
        else:
            pid = get_player_id(player_name); df = fetch_gamelog(pid, seasons_b, include_playoffs_b, only_playoffs_b)
            d = df.copy(); d = d[d["MIN_NUM"] >= min_min_b]
            if loc_choice == "Home Only": d = d[d["MATCHUP"].astype(str).str.contains("vs", regex=False)]
            elif loc_choice == "Away": d = d[d["MATCHUP"].astype(str).str.contains("@", regex=False)]
            if nba_cup_only_b: d = d[d["GAME_DATE_DT"].isin(NBA_CUP_DATES)]
            d = d.sort_values("GAME_DATE_DT", ascending=False).head(last_n)
            left, right = st.columns([1,2])
            with left:
                img = headshot_url(pid); logo = get_team_logo_from_df(df)
                img_html = f'<img src="{img}" width="120" style="border-radius:10px;">' if img else ""
                logo_html = f'<img src="{logo}" width="55" style="margin-top:8px;">' if logo else ""
                player_html = f"<div style='text-align:center; padding:10px;'>{img_html}<br>{logo_html}<div style='font-size:1.3rem; font-weight:700; margin-top:6px;'>{player_name}</div></div>"
                st.markdown(player_html, unsafe_allow_html=True)
                st.caption(f"Filters: Seasons {', '.join(seasons_b)} • Last {last_n} • Min {min_min_b}m • {loc_choice}")
            with right:
                stat_list = ["PTS","REB","AST","FG3M","STL","BLK","P+R","P+A","R+A","PRA"]
                rows = []
                for sc in stat_list:
                    out = breakeven_for_stat(compute_stat_series(d, sc))
                    line = out["line"]
                    if line is None: rows.append({"Stat": STAT_LABELS[sc], "Breakeven Line": "—", "Over": "—", "Under": "—"}); continue
                    rows.append({"Stat": STAT_LABELS[sc], "Breakeven Line": fmt_half(line), "Over": f"{out['over_prob']*100:.1f}% ({out['over_odds']})", "Under": f"{out['under_prob']*100:.1f}% ({out['under_odds']})"})
                st.table(pd.DataFrame(rows).set_index("Stat"))
with tab_mc:
    st.subheader("🎲 Monte Carlo Prop Simulator (Predictive)")
    st.caption(f"Simulate a specific player prop to understand their dispersion and confidence intervals")
    mc_text = st.text_input("Prop (e.g., 'LeBron James O 25.5 PTS Home -110')", placeholder="Enter player + O/U + line + stat + location + odds", help="Parser handles shortcuts: O/U, PTS/REB/AST/etc, Home/Away/All, odds like -110 or +120")
    seasons_mc = st.multiselect("Seasons", ["2025-26", "2024-25","2023-24","2022-23","2021-22"], default=["2025-26"])
    last_n_mc = st.slider("Last N Games", 5, 100, 20)
    min_min_mc = st.slider("Min Minutes", 0, 40, 20)
    sims_mc = st.slider("Number of Simulations", 2000, 50000, 15000, 2000)
    if st.button("Run Simulation") and mc_text.strip():
        parsed = parse_input_line(mc_text)
        if not parsed: st.warning("Could not parse the input line. Check format (e.g., 'Curry O 5.5 3PM -110')."); st.stop()
        pid = get_player_id(parsed["player"])
        if not pid: st.warning(f"Player '{parsed['player']}' not found. Try full name (e.g., 'Stephen Curry')."); st.stop()
        df = fetch_gamelog(pid, seasons_mc, include_playoffs=False, only_playoffs=False)
        if df.empty: st.warning("No game log data available for this player in the selected seasons. Try adding more seasons."); st.stop()
        d = df.copy(); d = d[d["MIN_NUM"] >= min_min_mc]
        if parsed["loc"] == "Home Only": d = d[d["MATCHUP"].str.contains("vs", na=False)]
        elif parsed["loc"] == "Away": d = d[d["MATCHUP"].str.contains("@", na=False)]
        d = d.sort_values("GAME_DATE_DT", ascending=False).head(last_n_mc)
        if d.empty: st.warning("No games matching the filters (e.g., min minutes, location). Loosen filters and try again."); st.stop()
        ser = compute_stat_series(d, parsed["stat"]).dropna()
        if ser.empty: st.warning("No valid stat history after filters."); st.stop()
        draws = monte_carlo_predictive(ser, sims_mc)
        thr = parsed["thr"]; direction = parsed["dir"]
        hit_prob = float((draws <= thr).mean()) if direction == "Under" else float((draws >= thr).mean())
        fair_odds = prob_to_american(hit_prob); book_odds = parsed["odds"]; book_prob = american_to_implied(book_odds)
        edge_pct = (hit_prob - book_prob) * 100 if book_prob is not None else None
        stat_name = STAT_LABELS.get(parsed["stat"], parsed["stat"])
        st.markdown(render_mc_result_card(parsed["player"], direction, thr, stat_name, parsed["loc"], last_n_mc, hit_prob, fair_odds, book_odds, edge_pct), unsafe_allow_html=True)
        st.subheader("📊 Distribution Summary")
        mean_val = float(np.mean(draws)); median_val = float(np.median(draws)); stdev = float(np.std(draws))
        p10 = float(np.percentile(draws, 10)); p90 = float(np.percentile(draws, 90))
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Mean", f"{mean_val:.1f}"); st.metric("Median", f"{median_val:.1f}")
        with col2: st.metric("Std Dev", f"{stdev:.2f}"); st.metric("10th %ile", f"{p10:.1f}")
        with col3: st.metric("90th %ile", f"{p90:.1f}"); st.metric("Sim Hit %", f"{hit_prob*100:.1f}%")
        st.caption("Based on smoothed Monte Carlo draws from historical data.")
        hist_color = "#00c896" if edge_pct is not None and edge_pct >= 0 else "#e05a5a"
        fig, ax = plt.subplots(figsize=(6, 3)); fig.patch.set_facecolor("#1e1f22"); ax.set_facecolor("#1e1f22")
        ax.hist(draws, bins=25, color=hist_color, alpha=0.75, edgecolor="#d1d5db", linewidth=0.4)
        ax.axvline(thr, color="#ff6666" if direction == "Over" else "#00c896", linestyle="--", linewidth=1.8)
        ax.set_xlabel(f"{stat_name} ({direction} {thr})", color="#e5e7eb"); ax.set_ylabel("Frequency", color="#e5e7eb")
        ax.tick_params(colors="#9ca3af")
        for s in ax.spines.values(): s.set_edgecolor("#4b5563")
        st.pyplot(fig, use_container_width=True)
        st.caption(f"💡 Simulations use historical variance + smoothing noise. Rerun for new random draws. Edge >0% = +EV bet.")
with tab_injury:
    st.subheader("🩹 Injury Impact Analyzer")
    st.caption(f"Search for an injured player to see how their team fares without them")
    colL, colR = st.columns([1.2, 2])
    with colL:
        season_inj = st.selectbox("Season", ["2025-26","2024-25","2023-24"], index=0, key="season_inj")
        team_inj = st.selectbox("Team", TEAM_ABBRS, index=TEAM_ABBRS.index("PHI") if "PHI" in TEAM_ABBRS else 0)
        roster_df = get_team_roster(season_inj, team_inj)
        if roster_df.empty: 
            st.warning("Could not load roster for this team/season.")
            injured_name = None
            injured_id = None
        else:
            injured_name = st.selectbox("Injured / Missing Player", roster_df["PLAYER"].tolist(), key="inj_player")
            injured_id = int(roster_df.loc[roster_df["PLAYER"] == injured_name, "PLAYER_ID"].iloc[0]) if injured_name else None
        stat_inj = st.selectbox("Stat", ["PTS","REB","AST","PRA"], index=0, key="stat_inj")
        min_games_without = st.slider("Min games without to include", 1, 15, 3, 1, key="min_g_without")
        double_injury = st.checkbox("Analyze double injury impact?", key="double_inj")
        second_injured_name = None
        second_injured_id = None
        if double_injury and not roster_df.empty and injured_name:
            available_players = [p for p in roster_df["PLAYER"].tolist() if p != injured_name]
            if available_players:
                second_injured_name = st.selectbox("Second Injured / Missing Player", available_players, key="inj_player2")
                second_injured_id = int(roster_df.loc[roster_df["PLAYER"] == second_injured_name, "PLAYER_ID"].iloc[0])
            else:
                st.warning("No other players available.")
                double_injury = False  # Disable if no options
        run_inj = st.button("Analyze Impact", key="run_inj")
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
            if stat_inj == "PRA": 
                team_logs["PRA"] = team_logs["PTS"].fillna(0) + team_logs["REB"].fillna(0) + team_logs["AST"].fillna(0)
            inj_logs1 = team_logs[team_logs["PLAYER_ID"] == injured_id]
            if inj_logs1.empty: 
                st.warning(f"{injured_name} has no logged games this season.")
                st.stop()
            games_with1 = set(inj_logs1["GAME_ID"].unique())
            all_games = set(team_logs["GAME_ID"].unique())
            if double_injury and second_injured_id:
                inj_logs2 = team_logs[team_logs["PLAYER_ID"] == second_injured_id]
                if inj_logs2.empty: 
                    st.warning(f"{second_injured_name} has no logged games this season.")
                    st.stop()
                games_with2 = set(inj_logs2["GAME_ID"].unique())
                games_both_in = games_with1 & games_with2
                games_at_least_one_in = games_with1 | games_with2
                games_both_out = all_games - games_at_least_one_in
                games_with = games_both_in  # Games where both played (w/ both)
                games_without = games_both_out  # Games where neither played (w/o both)
                injured_players = [injured_name, second_injured_name]
                exclude_ids = [injured_id, second_injured_id]
                without_msg = f"both {injured_name} and {second_injured_name} were OUT"
            else:
                games_with = games_with1
                games_without = all_games - games_with1
                injured_players = [injured_name]
                exclude_ids = [injured_id]
                without_msg = f"{injured_name} was OUT"
            if not games_without: 
                st.warning(f"No games where {without_msg}.")
                st.stop()
            exclude_condition = team_logs["PLAYER_ID"].isin(exclude_ids)
            with_df = team_logs[(team_logs["GAME_ID"].isin(games_with)) & (~exclude_condition)].copy()
            without_df = team_logs[(team_logs["GAME_ID"].isin(games_without)) & (~exclude_condition)].copy()
            stat_col = stat_inj
            g_with = with_df.groupby("PLAYER_ID")[stat_col].mean()
            g_without = without_df.groupby("PLAYER_ID")[stat_col].mean()
            n_without = without_df.groupby("PLAYER_ID")["GAME_ID"].nunique()
            rows = []
            idx = sorted(set(g_with.index) | set(g_without.index))
            inj_str = " & ".join(injured_players)
            for pid in idx:
                if pid in exclude_ids: 
                    continue
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
                    f"{stat_col} w/ {inj_str}": w, 
                    f"{stat_col} w/o {inj_str}": wo, 
                    "Games w/o": nwo, 
                    "Delta": delta
                })
            if not rows: 
                st.warning("No players met the filters.")
                st.stop()
            impact_df = pd.DataFrame(rows).sort_values("Delta", ascending=False).reset_index(drop=True)
            def apply_delta_color(val):
                if val > 0: 
                    return "color:#7CFCBE; font-weight:700;"
                elif val < 0: 
                    return "color:#FF6B6B; font-weight:700;"
                else: 
                    return "color:#e5e7eb;"
            html = """
            <style>
                table.custom-table {border-collapse: collapse; width: 100%; border-radius: 10px; overflow: hidden; margin-top: 10px;}
                table.custom-table th {background-color: #1f2125; color: #f9fafb; padding: 10px; font-weight: 700; font-size: 0.9rem; border-bottom: 1px solid #333;}
                table.custom-table td {background-color: #131417; color: #e5e7eb; padding: 8px 10px; border-bottom: 1px solid #2a2d31; font-size: 0.9rem;}
            </style>
            <table class="custom-table"><thead><tr>"""
            for col in impact_df.columns: 
                html += f"<th>{col}</th>"
            html += "</tr></thead><tbody>"
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
            st.caption(f"Positive Delta = player gains production when **{inj_str}** {'are' if len(injured_players)>1 else 'is'} OUT.")
            st.markdown(html, unsafe_allow_html=True)
with tab_me:
    st.subheader("🔥 Matchup Exploiter — Auto-Detected Game Edges")
    c1, c2 = st.columns([1.2, 1])
    with c1: season_me = st.selectbox("Season for stats", ["2025-26", "2024-25", "2023-24", "2022-23"], index=0)
    with c2: last_n_me = st.slider("Last N games (primary form window)", 5, 25, 10)
    exclude_low_usage = st.checkbox("Exclude low-usage players (Season ≥15 MPG or L5 ≥18 MPG)", value=False)
    game_date = st.date_input("Games for date", value=today)
    run_matchups = st.button("Scan Matchups")
    if run_matchups:
        date_str = game_date.strftime("%Y%m%d"); sb = fetch_scoreboard_cached(date_str); games = extract_games_from_scoreboard(sb)
        if not games: st.warning("No games available for that date."); st.stop()
        logs = get_league_player_logs(season_me)
        if logs.empty: st.warning("No player logs available for that season."); st.stop()
        if "MIN_NUM" not in logs.columns: logs["MIN_NUM"] = logs["MIN"].apply(to_minutes)
        if "GAME_DATE_DT" not in logs.columns: logs["GAME_DATE_DT"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
        if "PRA" not in logs.columns: logs["PRA"] = logs["PTS"].fillna(0) + logs["REB"].fillna(0) + logs["AST"].fillna(0)
        base_def = get_team_defense_table(season_me).copy(); base_def["PRA_allowed"] = base_def["PTS_allowed"] + base_def["REB_allowed"] + base_def["AST_allowed"]
        num_teams = len(base_def)
        for stat in ["PTS", "REB", "AST", "PRA", "FG3M"]:
            col = f"{stat}_allowed"; vals = base_def[col]; ranks = vals.rank(ascending=False, method="min").astype(int)
            base_def[f"{stat}_rank"] = ranks
        team_def_overall = {}
        for _, row in base_def.iterrows():
            team = row["Team"]; team_def_overall[team] = {}
            for stat in ["PTS", "REB", "AST", "PRA", "FG3M"]:
                col = f"{stat}_allowed"; rcol = f"{stat}_rank"
                team_def_overall[team][stat] = {"allowed": float(row[col]), "rank": int(row[rcol])}
        team_pos_def = build_team_positional_defense(season_me)
        max_window = max(10, last_n_me); ALL_STATS = ["PTS", "REB", "AST", "PRA", "FG3M"]
        for g in games:
            home, away, status = g["home"], g["away"], g.get("status", "")
            if home not in team_def_overall or away not in team_def_overall: continue
            overs_candidates, fades_candidates = [], []
            for team_abbr, opp_team, side_label in [(away, home, "Away"), (home, away, "Home")]:
                team_logs = logs[logs["TEAM_ABBREVIATION"] == team_abbr].copy()
                if team_logs.empty: continue
                team_logs = team_logs.sort_values("GAME_DATE_DT", ascending=False).groupby("PLAYER_ID").head(max_window)
                grp = team_logs.groupby(["PLAYER_ID", "PLAYER_NAME"])
                for (pid, name), sub in grp:
                    min_series = pd.to_numeric(sub["MIN_NUM"], errors="coerce").dropna()
                    if len(min_series) < 5: continue
                    season_min = float(min_series.mean()); l5_min = float(min_series.head(5).mean()); min_diff = l5_min - season_min
                    if exclude_low_usage and (season_min < 15) and (l5_min < 18): continue
                    player_pos = get_player_position(pid)
                    pos_def = team_pos_def[opp_team][player_pos] if opp_team in team_pos_def and player_pos in team_pos_def[opp_team] else None
                    stat_std_cache = {}
                    for stat in ALL_STATS:
                        stat_series = pd.to_numeric(sub[stat], errors="coerce").dropna()
                        if len(stat_series) < 7: continue
                        season_avg = float(stat_series.mean()); stat_std = float(stat_series.std(ddof=0) or 0.0); stat_std_cache[stat] = stat_std
                        window_avgs = {}
                        if len(stat_series) >= 3: window_avgs["L3"] = float(stat_series.head(3).mean())
                        if len(stat_series) >= 5: window_avgs["L5"] = float(stat_series.head(5).mean())
                        if len(stat_series) >= 10: window_avgs["L10"] = float(stat_series.head(10).mean())
                        if not window_avgs: continue
                        diffs = {w: avg - season_avg for w, avg in window_avgs.items()}
                        best_window, best_diff = max(diffs.items(), key=lambda x: x[1]); best_window_avg = window_avgs[best_window]
                        window_size = int(best_window[1:]); thresholds = EDGE_THRESHOLDS[stat]; strong_thr = thresholds["strong"]; mild_thr = thresholds["mild"]
                        overall_def = team_def_overall[opp_team][stat]; overall_allowed = overall_def["allowed"]; overall_rank = overall_def["rank"]
                        pos_allowed, pos_rank = (pos_def[f"{stat}_allowed"], pos_def[f"{stat}_rank"]) if pos_def is not None else (overall_allowed, overall_rank)
                        combo_rank = min(overall_rank, pos_rank); mid_rank = (num_teams + 1) / 2.0
                        def_factor = (mid_rank - combo_rank) / (mid_rank - 1)
                        denom = stat_std if stat_std > 0.75 else 0.75; trend_z = best_diff / denom
                        usage_factor = np.clip(season_min / 36.0, 0.0, 1.0)
                        edge_score = 0.55 * trend_z + 0.30 * def_factor + 0.15 * np.clip(min_diff / 6.0, -1.0, 1.0) + 0.15 * usage_factor
                        projection = season_avg + 0.6 * best_diff + 0.4 * def_factor * denom; proj_delta = projection - season_avg
                        proj_z = proj_delta / denom if denom > 0 else 0.0; prob_over = logistic_prob(proj_z); prob_under = 1.0 - prob_over
                        is_strong_trend = best_diff >= strong_thr; is_mild_trend = best_diff >= mild_thr
                        weak_def = combo_rank <= 7; neutral_to_weak_def = combo_rank <= 12; strong_def = combo_rank >= (num_teams - 4)
                        stat_label = STAT_LABELS.get(stat, stat)
                        if is_mild_trend and neutral_to_weak_def and edge_score > 0:
                            strength_tag = "strong" if (is_strong_trend and weak_def and edge_score > 1.0) else "mild"
                            weak_ord_overall = ordinal(overall_rank); weak_ord_pos = ordinal(pos_rank)
                            text = f"{name} ({side_label} {team_abbr}) projected {projection:.1f} {stat_label} vs {season_avg:.1f} season baseline over last {window_size} ({best_window_avg:.1f} vs {season_avg:.1f}). {opp_team} allows about {overall_allowed:.1f} {stat_label} per game overall ({weak_ord_overall}-highest), and roughly {pos_allowed:.1f} to {player_pos}s ({weak_ord_pos}-highest by position). Estimated chance to beat that baseline is around {prob_over*100:,.0f}%."
                            if abs(min_diff) >= 1.0: text += f" Minutes trend: {min_diff:+.1f} (L5 vs season)."
                            prefix = player_prefix(name, stat)
                            line_html = f"""
<div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:12px; background:#2a2a2a; padding:10px; border-radius:8px; border-left:4px solid #ff6b35;">
  <img src="{get_player_headshot(pid)}" width="40" style="border-radius:6px;" />
  <div style="color:#fff; font-size:14px;">
    <span style="font-weight:700; font-size:15px; letter-spacing:0.03em; margin-right:4px;">{prefix}</span>
    <span>{text}</span>
  </div>
</div>
"""
                            overs_candidates.append({"score": edge_score, "strength": strength_tag, "html": line_html})
                        if (best_diff <= -mild_thr) and strong_def and edge_score < 0:
                            strong_ord_overall = ordinal(num_teams - overall_rank + 1); strong_ord_pos = ordinal(num_teams - pos_rank + 1)
                            text = f"{name} ({side_label} {team_abbr}) projected {projection:.1f} {stat_label} vs {season_avg:.1f} season baseline, with only about {prob_over*100:,.0f}% chance to beat that mark given trend and matchup. {opp_team} allows just {overall_allowed:.1f} {stat_label} per game overall ({strong_ord_overall}-lowest), and about {pos_allowed:.1f} to {player_pos}s ({strong_ord_pos}-lowest by position)."
                            if abs(min_diff) >= 1.0: text += f" Minutes trend: {min_diff:+.1f} (L5 vs season)."
                            prefix = player_prefix(name, stat)
                            line_html = f"""
<div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:12px; background:#2a2a2a; padding:10px; border-radius:8px; border-left:4px solid #666;">
  <img src="{get_player_headshot(pid)}" width="40" style="border-radius:6px;" />
  <div style="color:#fff; font-size:14px;">
    <span style="font-weight:700; font-size:15px; letter-spacing:0.03em; margin-right:4px;">{prefix}</span>
    <span>{text}</span>
  </div>
</div>
"""
                            fades_candidates.append({"score": abs(edge_score), "html": line_html})
            away_logo = TEAM_LOGOS.get(away, "https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/default.png")
            home_logo = TEAM_LOGOS.get(home, "https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/default.png")
            best_over_score = max(e["score"] for e in overs_candidates) if overs_candidates else 0
            best_fade_score = max(e["score"] for e in fades_candidates) if fades_candidates else 0
            game_heat = 92 if best_over_score > 1.2 else 86 if best_over_score > 0.8 else 78 if best_over_score > 0.4 or best_fade_score > 0.7 else 70 if best_over_score > 0 or best_fade_score > 0 else 55
            header_html = f"""
<div style="background:#1e1e1e; padding:12px; border-radius:10px; border:1px solid #333; box-shadow:0 2px 8px rgba(0,0,0,0.3); margin-bottom:12px;">
  <div style="display:flex; align-items:center; gap:12px; font-size:18px; font-weight:600; color:#fff;">
    <img src="{away_logo}" width="32" style="border-radius:6px;" />
    <span>{away}</span>
    <span style="opacity:0.7; font-size:16px;">@</span>
    <span>{home}</span>
    <img src="{home_logo}" width="32" style="border-radius:6px;" />
  </div>
  <div style="margin-top:4px; opacity:0.8; color:#aaa; font-size:12px;">{status}</div>
  <div style="margin-top:8px; font-size:16px; font-weight:600; color:#ff6b35;">Matchup Heat: {game_heat} / 100</div>
</div>
"""
            st.markdown(header_html, unsafe_allow_html=True)
            overs_candidates.sort(key=lambda x: x["score"], reverse=True); fades_candidates.sort(key=lambda x: x["score"], reverse=True)
            top_overs = overs_candidates[:3]; top_fades = fades_candidates[:3]
            if top_overs:
                st.markdown("### 🔥 Best Overs (All Stats)")
                for item in top_overs: st.markdown(item["html"], unsafe_allow_html=True)
            if top_fades:
                st.markdown("### 🚫 Tough Spots / Fade Candidates")
                for item in top_fades: st.markdown(item["html"], unsafe_allow_html=True)
            if not top_overs and not top_fades:
                st.markdown("<div style='color:#ccc; font-size:13px;'>No clear edges detected for this matchup. ⚠️</div>", unsafe_allow_html=True)
            st.markdown("---")
with tab_matchups:
    st.subheader("📈 Team Defense — Defensive Averages (Per Game)")
    st.caption("Based on NBA team game logs. Sorted from weakest (top) to strongest (bottom) defense.")
    season = get_current_season_str(); df = load_team_logs(season)
    if df.empty: st.warning("No data yet for this season."); st.stop()
    stats = ["PTS", "REB", "AST", "FG3M"]; cols = st.columns(len(stats))
    for i, stat in enumerate(stats):
        allowed = df.groupby("OPP", as_index=False)[stat].mean().rename(columns={stat: f"{stat}_ALLOWED_PER_GAME"}).sort_values(f"{stat}_ALLOWED_PER_GAME", ascending=False)
        with cols[i]:
            st.markdown(f"### {stat} Allowed / Game")
            for _, row in allowed.iterrows():
                team, val = row["OPP"], row[f"{stat}_ALLOWED_PER_GAME"]
                color = get_team_color(team); bg = soft_bg(color, 0.18); border_color = color + "99"
                st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between; margin-bottom:5px;padding:6px 12px;border-radius:8px; background-color:{bg}; border:1px solid {border_color};'>
  <span style='font-weight:700;color:#FFFFFF;font-size:1rem; text-shadow:0 0 6px {color}99;'>{team}</span>
  <span style='font-size:0.95rem;color:#FFFFFF;font-weight:600; text-shadow:0 0 6px {color}66;'>{val:.1f}</span>
</div>
""", unsafe_allow_html=True)
    st.divider()
    st.caption(f"Season {season} • Source: NBA Stats API • Regular-season team logs (per-game averages)")
with tab_qh:
    st.subheader("⚡ Quick Hits")
    st.caption("Player averages for first quarter or first 3 minutes over last N games")
  
    col1, col2 = st.columns([1, 2])
    with col1:
        season_qh = st.selectbox("Season", ["2025-26", "2024-25", "2023-24"], index=0, key="season_qh")
        team_qh = st.selectbox("Team", TEAM_ABBRS, index=TEAM_ABBRS.index("PHI") if "PHI" in TEAM_ABBRS else 0, key="team_qh")
    with col2:
        last_n_qh = st.slider("Last N games", 5, 30, 10, key="last_n_qh")
        time_filter_qh = st.selectbox("Time Filter", ["First Quarter", "First 3 Minutes"], index=0, key="time_qh")
  
    run_qh = st.button("Load Quick Hits", key="run_qh")
  
    if run_qh and team_qh:
        team_id = team_abbrev_to_id.get(team_qh, 1610612745)  # Default PHI
        try:
            # Get last N game IDs using teamgamelog (lowercase 'g')
            gamelog = teamgamelog.TeamGameLog(season_nullable=season_qh.replace('-', ''), team_id=team_id)
            df_gamelog = gamelog.get_data_frames()[0]
            df_gamelog['GAME_DATE'] = pd.to_datetime(df_gamelog['GAME_DATE'])
            recent_games = df_gamelog.nlargest(last_n_qh, 'GAME_DATE')['GAME_ID'].tolist()
          
            # Aggregate stats
            player_stats = {'PLAYER_ID': [], 'PLAYER_NAME': [], 'PTS': [], 'REB': [], 'AST': []}
            total_games = len(recent_games)
          
            for game_id in recent_games:
                # Use boxscoretraditionalv2 (lowercase 'b')
                bs = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
                # Access via .get_data_frames() or direct attributes if available
                player_stats_home = bs.home_players_stats if hasattr(bs, 'home_players_stats') else pd.DataFrame()
                player_stats_away = bs.away_players_stats if hasattr(bs, 'away_players_stats') else pd.DataFrame()
                all_players = pd.concat([player_stats_home, player_stats_away], ignore_index=True)
                all_players = all_players[all_players['TEAM_ABBREVIATION'] == team_qh.upper()]
              
                if time_filter_qh == "First Quarter":
                    # Correct column names: PTS_QTR1, REB_QTR1, AST_QTR1 (per NBA API)
                    for _, row in all_players.iterrows():
                        pid = row['PLAYER_ID']
                        name = row['PLAYER_NAME']
                        pts = pd.to_numeric(row.get('PTS_QTR1', 0), errors='coerce') or 0
                        reb = pd.to_numeric(row.get('REB_QTR1', 0), errors='coerce') or 0
                        ast = pd.to_numeric(row.get('AST_QTR1', 0), errors='coerce') or 0
                        player_stats['PLAYER_ID'].append(pid)
                        player_stats['PLAYER_NAME'].append(name)
                        player_stats['PTS'].append(pts)
                        player_stats['REB'].append(reb)
                        player_stats['AST'].append(ast)
                else:  # First 3 Minutes - Basic PBP aggregation
                    # Use playbyplayv2 (lowercase 'p')
                    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id).get_data_frames()[0]
                    # Filter for Period=1 and clock > 9:00 (first 3 min of Q1)
                    first3_events = pbp[(pbp['PERIOD'] == 1) & (pd.to_numeric(pbp['PCTIMESTRING'].str[:2], errors='coerce') > 9)].copy()
                    # Simple aggregation: sum actions by PLAYER1_ID (e.g., made shots for PTS, etc.)
                    # Note: This is a stub—full logic needs event_type grouping (1=shot, 5=rebound, 8=assist)
                    player_first3 = {}
                    for _, event in first3_events.iterrows():
                        player_id = event.get('PLAYER1_ID')
                        if pd.isna(player_id): continue
                        if player_id not in player_first3: player_first3[player_id] = {'PTS': 0, 'REB': 0, 'AST': 0}
                        # Example: If made FG/3PT, add to PTS; REB if REBOUND event; AST if assist
                        if event.get('EVENTMSGTYPE') in [1, 3]:  # Shot made
                            player_first3[player_id]['PTS'] += 2 if event.get('EVENTNUM') % 2 == 0 else 3  # Simplified
                        # Add REB/AST logic similarly...
                    # Match to all_players for name
                    for _, row in all_players.iterrows():
                        pid = row['PLAYER_ID']
                        name = row['PLAYER_NAME']
                        stats = player_first3.get(pid, {'PTS': 0, 'REB': 0, 'AST': 0})
                        player_stats['PLAYER_ID'].append(pid)
                        player_stats['PLAYER_NAME'].append(name)
                        player_stats['PTS'].append(stats['PTS'])
                        player_stats['REB'].append(stats['REB'])
                        player_stats['AST'].append(stats['AST'])
          
            if not player_stats['PLAYER_ID']:
                st.warning("No player data found.")
                st.stop()
          
            df_stats = pd.DataFrame(player_stats)
            df_avg = df_stats.groupby(['PLAYER_ID', 'PLAYER_NAME']).agg({
                'PTS': 'mean', 'REB': 'mean', 'AST': 'mean'
            }).round(1).reset_index()
          
            # Team header
            team_logo = TEAM_LOGOS.get(team_qh, "")
            st.markdown(f"""
            <div style='text-align: center; margin-bottom: 20px;'>
                <img src='{team_logo}' width='60' height='60' style='border-radius: 50%;' />
                <h3 style='color: #fff; margin: 10px 0;'>{team_qh}</h3>
                <p style='color: #aaa; font-size: 0.9rem;'>Averages over {total_games} games ({time_filter_qh})</p>
            </div>
            """, unsafe_allow_html=True)
          
            # Custom HTML table with headshots
            html_table = """
            <style>
                table.quick-hits {border-collapse: collapse; width: 100%; background: #1e1e1e; border-radius: 8px; overflow: hidden;}
                th {background: #333; color: #fff; padding: 12px; text-align: left; font-weight: 600;}
                td {padding: 10px; border-bottom: 1px solid #444; color: #fff;}
                img {border-radius: 4px; width: 40px; height: 40px;}
            </style>
            <table class='quick-hits'>
                <thead><tr><th>Player</th><th>PTS</th><th>REB</th><th>AST</th></tr></thead>
                <tbody>
            """
            for _, row in df_avg.iterrows():
                pid = row['PLAYER_ID']
                name = row['PLAYER_NAME']
                pts = row['PTS']
                reb = row['REB']
                ast = row['AST']
                headshot = get_player_headshot(pid)
                html_table += f"""
                <tr>
                    <td><img src='{headshot}' onerror="this.src='https://a.espncdn.com/i/headshots/nba/players/full/{pid}.png'" /> {name}</td>
                    <td>{pts}</td><td>{reb}</td><td>{ast}</td>
                </tr>
                """
            html_table += "</tbody></table>"
            st.markdown(html_table, unsafe_allow_html=True)
          
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("Ensure nba_api is installed: pip install nba_api")
with tab_ml:
    st.subheader("💵 ML, Spread, & Totals Analyzer")
    st.caption("Projections for all games on the selected date. Click expanders for detailed efficiencies & injuries.")
    ml_date = st.date_input("Game Date", value=today, key="ml_date")
    date_str = ml_date.strftime("%Y%m%d")
    scoreboard_ml = fetch_scoreboard_cached(date_str)
    games_ml = extract_games_from_scoreboard(scoreboard_ml)
    if not games_ml:
        st.warning("No games found for this date.")
        st.stop()
    season = get_current_season_str()
    @st.cache_data(ttl=1800) # Cache for 30min
    def get_game_data(season):
        logs_team = load_enhanced_team_logs(season)
        player_impacts = load_player_impact(season)
        league_logs_upper = load_league_player_logs_upper(season)
        margin_model = train_margin_model(season)
        total_model = train_total_model(season)
        return logs_team, player_impacts, league_logs_upper, margin_model, total_model
    logs_team, player_impacts, league_logs_upper, margin_model, total_model = get_game_data(season)
    if logs_team.empty:
        st.warning("No data available for this season yet.")
        st.stop()
    
    # Convert GAME_DATE to datetime for filtering
    logs_team['GAME_DATE'] = pd.to_datetime(logs_team['GAME_DATE'])
    league_logs_upper['GAME_DATE'] = pd.to_datetime(league_logs_upper['GAME_DATE'])
    
    # Compute recent metrics based on last 10 games before ml_date
    import pandas as pd
    ml_date_pd = pd.to_datetime(ml_date)
    logs_recent = logs_team[logs_team['GAME_DATE'] < ml_date_pd].sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'], ascending=[True, False])
    last_10_games = logs_recent.groupby('TEAM_ABBREVIATION').head(10)
    team_ortg = last_10_games.groupby("TEAM_ABBREVIATION")["ortg"].mean()
    team_drtg = last_10_games.groupby("TEAM_ABBREVIATION")["drtg"].mean()
    team_nrtg = team_ortg - team_drtg
    team_poss = last_10_games.groupby("TEAM_ABBREVIATION")["poss"].mean()
    
    # Filter for last 20 games for injury metrics
    last_20_games = logs_recent.groupby('TEAM_ABBREVIATION').head(20)
    
    def get_impact_from_mpg(mpg):
        if mpg < 10:
            return 0.15
        elif mpg < 15:
            return 0.25
        elif mpg < 20:
            return 0.45
        elif mpg < 25:
            return 0.6
        elif mpg < 30:
            return 0.85
        elif mpg < 35:
            return 1.0
        else:
            return 1.0
    
    # Fallback for zero deltas (e.g., rookies/low-sample): modest imputation by position/role
    def get_fallback_deltas(player_name):
        # Simple heuristic: bigs/centers like Sarr/Hartenstein get small team shifts; guards/wings less
        if 'sarr' in player_name.lower() or 'hartenstein' in player_name.lower():
            return -1.5, +2.0  # Slight O hit, D anchor loss
        # Add more players/roles as needed
        return 0.0, 0.0
    
    for game_idx, game in enumerate(games_ml):
        home_raw = game["home"]
        away_raw = game["away"]
        home = ABBREV_MAP.get(home_raw, home_raw)
        away = ABBREV_MAP.get(away_raw, away_raw)
        event_id = game["event_id"]
        status = game.get("status", "")
        # Rest days
        rest_home = get_rest_days(home, logs_team, ml_date)
        rest_away = get_rest_days(away, logs_team, ml_date)
        rest_diff = rest_home - rest_away
        # Team stats
        ortg_home = float(team_ortg.get(home, 110.0))
        drtg_home = float(team_drtg.get(home, 110.0))
        nrtg_home = float(team_nrtg.get(home, 0.0))
        ortg_away = float(team_ortg.get(away, 110.0))
        drtg_away = float(team_drtg.get(away, 110.0))
        nrtg_away = float(team_nrtg.get(away, 0.0))
        poss_home = float(team_poss.get(home, 98.0))
        poss_away = float(team_poss.get(away, 98.0))
        proj_possessions = (poss_home + poss_away) / 2.0
        # Injuries
        summary = get_espn_game_summary(event_id)
        inj_home, inj_away, adjust_home_nrtg, adjust_away_nrtg, adjust_home_ortg, adjust_away_ortg, adjust_home_drtg, adjust_away_drtg = extract_injuries_from_summary(summary, home, away, ml_date, player_impacts, last_20_games, league_logs_upper)
        
        # Get recent game dates for home and away
        recent_game_dates_home = last_20_games[last_20_games['TEAM_ABBREVIATION'] == home]['GAME_DATE'].unique()[:20]
        recent_game_dates_away = last_20_games[last_20_games['TEAM_ABBREVIATION'] == away]['GAME_DATE'].unique()[:20]
        
        # Override impacts based on recent MPG (including 0 for missed games) and re-compute adjustments
        for i in inj_home:
            player_name = i['name']
            player_logs = league_logs_upper[
                (league_logs_upper['TEAM_ABBREVIATION'] == home) & 
                (league_logs_upper['PLAYER_NAME'] == player_name) & 
                (league_logs_upper['GAME_DATE'].isin(recent_game_dates_home))
            ]
            played_mins = player_logs['MIN'].tolist() if not player_logs.empty else []
            total_mins = sum(played_mins) + 0 * (20 - len(played_mins))
            avg_mpg = total_mins / 20.0 if len(recent_game_dates_home) > 0 else 0.0
            i['impact_score'] = get_impact_from_mpg(avg_mpg)
            # Debug: st.write(f"Debug: {player_name} (home) avg MPG: {avg_mpg:.1f}, impact: {i['impact_score']:.2f}")
            
            # Fallback if deltas are zero
            if i.get('ortg_delta', 0.0) == 0 and i.get('drtg_delta', 0.0) == 0:
                fallback_ortg, fallback_drtg = get_fallback_deltas(player_name)
                i['ortg_delta'] = fallback_ortg
                i['drtg_delta'] = fallback_drtg
        
        adjust_home_ortg = sum(i['impact_score'] * i.get('ortg_delta', 0.0) for i in inj_home)
        adjust_home_drtg = sum(i['impact_score'] * i.get('drtg_delta', 0.0) for i in inj_home)
        adjust_home_nrtg = adjust_home_ortg - adjust_home_drtg
        
        for i in inj_away:
            player_name = i['name']
            player_logs = league_logs_upper[
                (league_logs_upper['TEAM_ABBREVIATION'] == away) & 
                (league_logs_upper['PLAYER_NAME'] == player_name) & 
                (league_logs_upper['GAME_DATE'].isin(recent_game_dates_away))
            ]
            played_mins = player_logs['MIN'].tolist() if not player_logs.empty else []
            total_mins = sum(played_mins) + 0 * (20 - len(played_mins))
            avg_mpg = total_mins / 20.0 if len(recent_game_dates_away) > 0 else 0.0
            i['impact_score'] = get_impact_from_mpg(avg_mpg)
            # Debug: st.write(f"Debug: {player_name} (away) avg MPG: {avg_mpg:.1f}, impact: {i['impact_score']:.2f}")
            
            # Fallback if deltas are zero
            if i.get('ortg_delta', 0.0) == 0 and i.get('drtg_delta', 0.0) == 0:
                fallback_ortg, fallback_drtg = get_fallback_deltas(player_name)
                i['ortg_delta'] = fallback_ortg
                i['drtg_delta'] = fallback_drtg
        
        adjust_away_ortg = sum(i['impact_score'] * i.get('ortg_delta', 0.0) for i in inj_away)
        adjust_away_drtg = sum(i['impact_score'] * i.get('drtg_delta', 0.0) for i in inj_away)
        adjust_away_nrtg = adjust_away_ortg - adjust_away_drtg
        
        nrtg_home_adj = nrtg_home + adjust_home_nrtg
        nrtg_away_adj = nrtg_away + adjust_away_nrtg
        nrtg_diff_adj = nrtg_home_adj - nrtg_away_adj
        # Margin projection
        hca = 2.7
        est_margin_base = round(nrtg_diff_adj + hca, 1)
        ml_margin = est_margin_base
        if margin_model is not None:
            ortg_diff = ortg_home - ortg_away
            drtg_diff = drtg_home - drtg_away
            pace_avg = proj_possessions
            feat_vec = np.array([[ortg_diff, drtg_diff, pace_avg, adjust_home_nrtg, adjust_away_nrtg, hca, rest_diff]])
            xgb_margin = margin_model.predict(feat_vec)[0]
            ml_margin = 0.5 * est_margin_base + 0.5 * xgb_margin
        est_margin = round(ml_margin, 1)
        est_spread_home = round(-est_margin, 1)
        win_prob_home = 1 / (1 + np.exp(-est_margin / 7.5))
        win_prob_away = 1 - win_prob_home
        ml_home = prob_to_ml(win_prob_home)
        ml_away = prob_to_ml(win_prob_away)
        # Total projection
        ortg_home_total = ortg_home + adjust_home_ortg
        ortg_away_total = ortg_away + adjust_away_ortg
        drtg_home_total = drtg_home + adjust_home_drtg
        drtg_away_total = drtg_away + adjust_away_drtg
        exp_pts_home = ((ortg_home_total + drtg_away_total) / 2.0) * (proj_possessions / 100.0)
        exp_pts_away = ((ortg_away_total + drtg_home_total) / 2.0) * (proj_possessions / 100.0)
        projected_total_base = round(exp_pts_home + exp_pts_away, 1)
        projected_total = projected_total_base
        if total_model is not None:
            total_feat = [ortg_home, ortg_away, drtg_home, drtg_away, proj_possessions, adjust_home_ortg, adjust_away_ortg, rest_diff]
            xgb_total = total_model.predict(np.array([total_feat]))[0]
            projected_total = 0.5 * projected_total_base + 0.5 * xgb_total
        projected_total = round(projected_total, 1)
        # Logos
        away_logo = TEAM_LOGOS.get(away, "")
        home_logo = TEAM_LOGOS.get(home, "")
        # Render compact predictions with logos
        game_header = f"""
<div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; padding: 10px; background: #1e1e1e; border-radius: 8px; border: 1px solid #333;'>
  <div style='display: flex; align-items: center; gap: 8px;'>
    <img src='{away_logo}' width='24' height='24' style='border-radius: 4px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{away.lower()}.png'" />
    <strong style='color: #fff;'>{away}</strong>
    <span style='color: #aaa;'>@</span>
    <img src='{home_logo}' width='24' height='24' style='border-radius: 4px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{home.lower()}.png'" />
    <strong style='color: #fff;'>{home}</strong>
  </div>
  <div style='color: #aaa; font-size: 0.9rem;'>{status}</div>
</div>
"""
        st.markdown(game_header, unsafe_allow_html=True)
        projected_html = textwrap.dedent(f"""
<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;'>
  <div style='border:1px solid #333; padding:10px; border-radius:8px; background:#1e1e1e; text-align:center;'>
    <div style='margin-bottom:6px; font-weight:600; font-size:0.9rem;'>Spread</div>
    <div style='display: flex; align-items: center; justify-content: center; gap: 4px;'>
      <img src='{home_logo}' width='16' height='16' style='border-radius: 2px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{home.lower()}.png'" />
      <span style='font-size:1.1rem; font-weight:700;'>{est_spread_home:+.1f}</span>
    </div>
  </div>
  <div style='border:1px solid #333; padding:10px; border-radius:8px; background:#1e1e1e; text-align:center;'>
    <div style='margin-bottom:6px; font-weight:600; font-size:0.9rem;'>Total</div>
    <div style='display: flex; align-items: center; justify-content: center; gap: 8px;'>
      <img src='{away_logo}' width='16' height='16' style='border-radius: 2px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{away.lower()}.png'" />
      <span style='font-size:1.1rem; font-weight:700;'>{projected_total:.1f}</span>
      <img src='{home_logo}' width='16' height='16' style='border-radius: 2px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{home.lower()}.png'" />
    </div>
  </div>
  <div style='border:1px solid #333; padding:10px; border-radius:8px; background:#1e1e1e; text-align:center;'>
    <div style='margin-bottom:6px; font-weight:600; font-size:0.9rem;'>Win Prob</div>
    <div style='display: flex; flex-direction: column; gap: 4px; font-size:0.9rem;'>
      <div style='display: flex; align-items: center; justify-content: center; gap: 4px;'>
        <img src='{home_logo}' width='16' height='16' style='border-radius: 2px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{home.lower()}.png'" />
        <span>{home}: {win_prob_home*100:.0f}%</span>
      </div>
      <div style='display: flex; align-items: center; justify-content: center; gap: 4px;'>
        <img src='{away_logo}' width='16' height='16' style='border-radius: 2px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{away.lower()}.png'" />
        <span>{away}: {win_prob_away*100:.0f}%</span>
      </div>
    </div>
  </div>
  <div style='border:1px solid #333; padding:10px; border-radius:8px; background:#1e1e1e; text-align:center;'>
    <div style='margin-bottom:6px; font-weight:600; font-size:0.9rem;'>Fair ML</div>
    <div style='display: flex; flex-direction: column; gap: 4px; font-size:0.9rem;'>
      <div style='display: flex; align-items: center; justify-content: center; gap: 4px;'>
        <img src='{home_logo}' width='16' height='16' style='border-radius: 2px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{home.lower()}.png'" />
        <span>{home}: {ml_home}</span>
      </div>
      <div style='display: flex; align-items: center; justify-content: center; gap: 4px;'>
        <img src='{away_logo}' width='16' height='16' style='border-radius: 2px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{away.lower()}.png'" />
        <span>{away}: {ml_away}</span>
      </div>
    </div>
  </div>
</div>
""")
        st.markdown(projected_html, unsafe_allow_html=True)
        with st.expander(f"Expand: Efficiencies & Injuries ({len(inj_home) + len(inj_away)} total)"):
            # Team strengths with logos
            strength_html = textwrap.dedent(f"""
<div style='margin-top:10px;'>
  <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
    <div style='display: flex; align-items: center; gap: 10px;'>
      <img src='{home_logo}' width='32' height='32' style='border-radius: 4px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{home.lower()}.png'" />
      <div>
        <div style='font-weight:600; margin-bottom:8px;'>Home: {home}</div>
        <div style='display:flex;'><span style='width:60px;'>ORTG:</span> <span style='font-weight:600;'>{ortg_home:.1f}</span></div>
        <div style='display:flex;'><span style='width:60px;'>DRTG:</span> <span style='font-weight:600;'>{drtg_home:.1f}</span></div>
        <div style='display:flex;'><span style='width:60px;'>NRTG:</span> <span style='font-weight:600; color:#00c896;'>{nrtg_home_adj:.1f}</span> <span style='color:#aaa; font-size:0.8rem;'>(adj {adjust_home_nrtg:+.1f})</span></div>
      </div>
    </div>
    <div style='display: flex; align-items: center; gap: 10px;'>
      <img src='{away_logo}' width='32' height='32' style='border-radius: 4px;' onerror="this.src='https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{away.lower()}.png'" />
      <div>
        <div style='font-weight:600; margin-bottom:8px;'>Away: {away}</div>
        <div style='display:flex;'><span style='width:60px;'>ORTG:</span> <span style='font-weight:600;'>{ortg_away:.1f}</span></div>
        <div style='display:flex;'><span style='width:60px;'>DRTG:</span> <span style='font-weight:600;'>{drtg_away:.1f}</span></div>
        <div style='display:flex;'><span style='width:60px;'>NRTG:</span> <span style='font-weight:600; color:#00c896;'>{nrtg_away_adj:.1f}</span> <span style='color:#aaa; font-size:0.8rem;'>(adj {adjust_away_nrtg:+.1f})</span></div>
      </div>
    </div>
  </div>
  <div style='margin-top:10px; font-weight:600; color:#00c896; font-size:1.1rem;'>Diff (Adj): {nrtg_diff_adj:+.1f} | Margin (w/ HCA): {est_margin:+.1f}</div>
</div>
""")
            st.markdown(strength_html, unsafe_allow_html=True)
            # Injuries (logos already in header, no need here)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{home} Out ({len(inj_home)} players):**")
                if inj_home:
                    for i in inj_home:
                        ret_str = i['return_date'].strftime('%b %d') if i['return_date'] else 'TBD'
                        st.write(f"- {i['name']} (impact {i.get('impact_score', 1.0):.2f}, ΔORTG/ΔDRTG {i.get('ortg_delta', 0.0):+.1f}/{i.get('drtg_delta', 0.0):+.1f}, {i['status']}, {i['injury']}, return {ret_str})")
                else:
                    st.write("No key injuries.")
                st.write(f"*NRTG Adj: {adjust_home_nrtg:+.1f} | ORTG/DRTG Adj: {adjust_home_ortg:+.1f}/{adjust_home_drtg:+.1f}*")
            with col2:
                st.write(f"**{away} Out ({len(inj_away)} players):**")
                if inj_away:
                    for i in inj_away:
                        ret_str = i['return_date'].strftime('%b %d') if i['return_date'] else 'TBD'
                        st.write(f"- {i['name']} (impact {i.get('impact_score', 1.0):.2f}, ΔORTG/ΔDRTG {i.get('ortg_delta', 0.0):+.1f}/{i.get('drtg_delta', 0.0):+.1f}, {i['status']}, {i['injury']}, return {ret_str})")
                else:
                    st.write("No key injuries.")
                st.write(f"*NRTG Adj: {adjust_away_nrtg:+.1f} | ORTG/DRTG Adj: {adjust_away_ortg:+.1f}/{adjust_away_drtg:+.1f}*")
        if game_idx < len(games_ml) - 1:
            st.markdown("---")
