# data.py
# NBA AlgoHub — Data Collection via nba_api
#
# Pulls: team game logs, team ratings, player game logs,
#        schedule/upcoming games, injury report

import time
import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamelog,
    leaguedashteamstats,
    leaguedashplayerstats,
    playergamelogs,
    teamgamelogs,
    scoreboardv2,
)
from nba_api.stats.static import teams as nba_teams_static
from config import CURRENT_SEASON, ROLLING_WINDOW

# nba_api rate-limit pause (seconds)
_PAUSE = 0.6


def _pause():
    time.sleep(_PAUSE)


# ── Team helpers ───────────────────────────────────────────────────────────────

def get_all_teams() -> pd.DataFrame:
    """Return static table of all NBA teams."""
    teams = nba_teams_static.get_teams()
    return pd.DataFrame(teams)   # id, full_name, abbreviation, nickname, city, state, year_founded


def get_team_id(abbreviation: str) -> int:
    df = get_all_teams()
    row = df[df["abbreviation"] == abbreviation.upper()]
    if row.empty:
        raise ValueError(f"Unknown team abbreviation: {abbreviation}")
    return int(row.iloc[0]["id"])


# ── Team game logs ─────────────────────────────────────────────────────────────

def get_team_game_logs(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    All team game logs for a season.
    Returns one row per team per game with box score stats.
    """
    gl = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="T",
        season_type_all_star="Regular Season",
    )
    _pause()
    df = gl.get_data_frames()[0]
    df.columns = [c.lower() for c in df.columns]
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    return df


# ── Team ratings (OffRtg / DefRtg / Pace) ─────────────────────────────────────

def get_team_ratings(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Season-level advanced team ratings: OffRtg, DefRtg, NetRtg, Pace, etc.
    """
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    _pause()
    df = stats.get_data_frames()[0]
    df.columns = [c.lower() for c in df.columns]
    return df


# ── Player game logs ───────────────────────────────────────────────────────────

def get_player_game_logs(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    All player game logs for the season.
    """
    gl = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star="Regular Season",
    )
    _pause()
    df = gl.get_data_frames()[0]
    df.columns = [c.lower() for c in df.columns]
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return df


# ── Player season averages ─────────────────────────────────────────────────────

def get_player_season_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Season-level per-game averages for all players.
    Includes: pts, reb, ast, min, usg_pct, etc.
    """
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
    )
    _pause()
    df = stats.get_data_frames()[0]
    df.columns = [c.lower() for c in df.columns]
    return df


def get_player_advanced_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Advanced per-game stats: usage %, off/def rating, etc.
    """
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    _pause()
    df = stats.get_data_frames()[0]
    df.columns = [c.lower() for c in df.columns]
    return df


# ── Today's schedule ──────────────────────────────────────────────────────────

def get_todays_games() -> pd.DataFrame:
    """
    Returns today's NBA schedule with game_id, home/away team IDs.
    Uses ScoreboardV3 (ScoreboardV2 has known issues with 2025-26 season).
    """
    import datetime
    from nba_api.stats.endpoints import scoreboardv3
    
    today = datetime.date.today().strftime("%Y-%m-%d")
    sb = scoreboardv3.ScoreboardV3(game_date=today, league_id="00")
    _pause()
    
    games_df = sb.get_data_frames()[1]   # Frame 1 = game header
    teams_df = sb.get_data_frames()[2]   # Frame 2 = team info
    
    if games_df.empty:
        return pd.DataFrame()
    
    rows = []
    for _, game in games_df.iterrows():
        game_id = game["gameId"]
        game_teams = teams_df[teams_df["gameId"] == game_id]
        if len(game_teams) < 2:
            continue
        # home team has higher seed number in away context — use teamId directly
        home = game_teams[game_teams["teamId"].isin(
            [t for t in game_teams["teamId"]]
        )].iloc[0]
        away = game_teams.iloc[1]
        rows.append({
            "game_id":        game_id,
            "game_date_est":  game.get("gameEt", today),
            "home_team_id":   int(game_teams.iloc[0]["teamId"]),
            "visitor_team_id":int(game_teams.iloc[1]["teamId"]),
            "arena_name":     "",
        })
    
    return pd.DataFrame(rows).drop_duplicates(subset=["game_id"]).reset_index(drop=True)

# ── Rolling features helper ────────────────────────────────────────────────────

def add_rolling_team_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Given a team game log DataFrame (one row per team per game, sorted by date),
    compute rolling averages for key stats.
    Returns df with new rolling_* columns.
    """
    stat_cols = ["pts", "fg_pct", "fg3_pct", "ft_pct", "ast", "reb", "tov", "plus_minus"]
    # Only use columns that actually exist
    stat_cols = [c for c in stat_cols if c in df.columns]

    df = df.sort_values(["team_id", "game_date"])
    for col in stat_cols:
        df[f"rolling_{col}"] = (
            df.groupby("team_id")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        )
    return df


def add_rolling_player_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Rolling averages for player game logs.
    """
    stat_cols = ["pts", "reb", "ast", "min", "fga", "fg_pct", "fg3a", "fta", "tov", "plus_minus"]
    stat_cols = [c for c in stat_cols if c in df.columns]

    df = df.sort_values(["player_id", "game_date"])
    for col in stat_cols:
        df[f"rolling_{col}"] = (
            df.groupby("player_id")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        )
    return df


def get_rest_days(team_logs: pd.DataFrame, upcoming_games: pd.DataFrame) -> dict:
    """
    Calculate days rest for each team playing today.
    Pulls playoff game logs to get accurate rest during postseason.
    Capped at 7 (anything more = well rested, treat same).
    """
    today = pd.Timestamp.today().normalize()

    # Also pull playoff logs to get most recent game date
    all_logs = team_logs.copy()
    try:
        playoff_gl = leaguegamelog.LeagueGameLog(
            season=CURRENT_SEASON,
            player_or_team_abbreviation="T",
            season_type_all_star="Playoffs",
        )
        _pause()
        playoff_df = playoff_gl.get_data_frames()[0]
        playoff_df.columns = [c.lower() for c in playoff_df.columns]
        playoff_df["game_date"] = pd.to_datetime(playoff_df["game_date"])
        all_logs = pd.concat([all_logs, playoff_df], ignore_index=True)
    except Exception:
        pass

    last_game = (
        all_logs.sort_values("game_date")
        .groupby("team_id")["game_date"]
        .last()
        .reset_index()
    )
    last_game.columns = ["team_id", "last_game_date"]

    playing_today = set()
    for _, game in upcoming_games.iterrows():
        playing_today.add(int(game["home_team_id"]))
        playing_today.add(int(game["visitor_team_id"]))

    rest_map = {}
    for _, row in last_game.iterrows():
        tid = int(row["team_id"])
        if tid not in playing_today:
            continue
        last = pd.Timestamp(row["last_game_date"]).normalize()
        delta = (today - last).days - 1
        rest_map[tid] = min(max(0, delta), 7)

    return rest_map




def get_injury_report() -> pd.DataFrame:
    """
    Scrape today's NBA injury report from the NBA Stats API via nba_api.
    Returns a DataFrame with columns:
      player_name, team_id, team_abbreviation, status, reason
    Status values: 'Out', 'Doubtful', 'Questionable', 'Available'
    """
    try:
        from nba_api.stats.endpoints import leagueinjurysink
        report = leagueinjurysink.LeagueInjurySink()
        _pause()
        df = report.get_data_frames()[0]
        df.columns = [c.lower() for c in df.columns]

        # Normalize column names across nba_api versions
        rename_map = {}
        for col in df.columns:
            if "player" in col and "name" in col:
                rename_map[col] = "player_name"
            elif col in ("team_id", "teamid"):
                rename_map[col] = "team_id"
            elif "abbr" in col or "abbreviation" in col:
                rename_map[col] = "team_abbreviation"
            elif "status" in col:
                rename_map[col] = "status"
            elif "reason" in col or "comment" in col:
                rename_map[col] = "reason"
        df = df.rename(columns=rename_map)

        keep = [c for c in ["player_name", "team_id", "team_abbreviation", "status", "reason"] if c in df.columns]
        return df[keep]

    except Exception as e:
        print(f"  Warning: Could not fetch injury report via nba_api ({e}). Trying ESPN fallback...")
        return _get_injury_report_espn()


def _get_injury_report_espn() -> pd.DataFrame:
    """
    Fallback: scrape injury data from ESPN's public API.
    """
    import requests
    rows = []
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # ESPN returns a list under different keys depending on version
        entries = data.get("injuries", data.get("items", []))

        for team_entry in entries:
            # Team abbreviation lives in team_entry["team"]["abbreviation"]
            team_info = team_entry.get("team", {})
            team_abbr = team_info.get("abbreviation", "")

            injuries = team_entry.get("injuries", team_entry.get("items", []))
            for injury in injuries:
                athlete = injury.get("athlete", {})
                status  = injury.get("status", injury.get("type", {}).get("description", ""))
                detail  = injury.get("details", {})
                reason  = detail.get("detail", detail.get("type", "")) if isinstance(detail, dict) else ""

                pname = athlete.get("displayName", athlete.get("fullName", ""))
                if pname and team_abbr:
                    rows.append({
                        "player_name":       pname,
                        "team_abbreviation": team_abbr,
                        "status":            status,
                        "reason":            reason,
                    })

    except Exception as e:
        print(f"  Warning: ESPN injury fallback also failed ({e}). No injury data available.")

    if rows:
        df = pd.DataFrame(rows)
        # Only keep actual out/doubtful — filter noise
        df = df[df["status"].str.lower().isin(["out", "doubtful", "injured reserve", "day-to-day"])]
        return df

    return pd.DataFrame(columns=["player_name", "team_abbreviation", "status", "reason"])


def get_key_injuries(
    injury_report: pd.DataFrame,
    player_stats:  pd.DataFrame,
    top_n:         int = 3,
) -> dict:
    """
    Identify teams with key players (top N by usage/pts) listed as Out or Doubtful.
    Returns dict: {team_abbreviation: [list of out player names]}
    """
    if injury_report.empty or player_stats.empty:
        return {}

    # Players listed Out or Doubtful
    out_statuses = ["Out", "Doubtful", "out", "doubtful", "OUT", "DOUBTFUL"]
    out_players  = injury_report[injury_report["status"].isin(out_statuses)].copy()
    if out_players.empty:
        return {}

    # Get top players per team by points scored
    pts_col = "pts" if "pts" in player_stats.columns else None
    if pts_col is None:
        return {}

    top_players = (
        player_stats.sort_values(pts_col, ascending=False)
        .groupby("team_id")
        .head(top_n)
    )

    # Build set of top player names (lowercase for matching)
    top_names = set(top_players["player_name"].str.lower().tolist())

    key_out = {}
    for _, row in out_players.iterrows():
        pname = row.get("player_name", "")
        if pname.lower() in top_names:
            team = row.get("team_abbreviation", row.get("team_id", "UNK"))
            key_out.setdefault(str(team), []).append(pname)

    return key_out


# ── Multi-season data loader ───────────────────────────────────────────────────

def get_team_game_logs_multi(seasons: list) -> pd.DataFrame:
    """
    Pull team game logs across multiple seasons and combine.
    Gives the model far more training examples.
    """
    dfs = []
    for season in seasons:
        print(f"  Fetching team logs: {season}...")
        try:
            gl = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="T",
                season_type_all_star="Regular Season",
            )
            _pause()
            df = gl.get_data_frames()[0]
            df.columns = [c.lower() for c in df.columns]
            df["game_date"] = pd.to_datetime(df["game_date"])
            df["season"] = season
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not fetch {season} logs: {e}")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    return combined.sort_values(["team_id", "game_date"]).reset_index(drop=True)


# ── Lineup scraper ─────────────────────────────────────────────────────────────

def get_tonights_lineups() -> dict:
    """
    Scrape tonight's projected starting lineups from Rotowire.
    Returns dict: {team_abbreviation: [list of starting player names]}
    """
    import requests
    from bs4 import BeautifulSoup

    lineups = {}
    try:
        url = "https://www.rotowire.com/basketball/nba-lineups.php"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Each game box has team lineups
        game_boxes = soup.find_all("div", class_="lineup__main")
        for box in game_boxes:
            teams = box.find_all("ul", class_="lineup__list")
            team_labels = box.find_all("div", class_="lineup__team")

            for i, team_ul in enumerate(teams):
                if i >= len(team_labels):
                    continue
                abbr_el = team_labels[i].find("div", class_="lineup__abbr")
                if not abbr_el:
                    continue
                abbr = abbr_el.text.strip().upper()
                players = []
                for li in team_ul.find_all("li", class_="lineup__player"):
                    name_el = li.find("a")
                    if name_el:
                        players.append(name_el.text.strip())
                if players:
                    lineups[abbr] = players

    except ImportError:
        print("  Warning: beautifulsoup4 not installed. Run: pip install beautifulsoup4")
    except Exception as e:
        print(f"  Warning: Could not scrape lineups from Rotowire ({e})")

    return lineups


def get_lineup_strength(
    lineups: dict,
    player_stats: pd.DataFrame,
    team_lookup: pd.DataFrame,
) -> dict:
    """
    For each team playing tonight, compute the sum of pts/game for starters.
    Returns dict: {team_abbreviation: lineup_pts_sum}
    Used to adjust team offensive rating based on who's actually playing.
    """
    abbr_to_id = dict(zip(team_lookup["abbreviation"], team_lookup["id"]))
    strength = {}

    for abbr, starters in lineups.items():
        total_pts = 0.0
        found = 0
        for pname in starters:
            match = player_stats[
                player_stats["player_name"].str.lower() == pname.lower()
            ]
            if not match.empty:
                total_pts += match.iloc[0].get("pts", 0)
                found += 1
        if found > 0:
            strength[abbr] = total_pts
    return strength


# ── Head-to-head features ──────────────────────────────────────────────────────

def get_h2h_features(
    team_logs: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    n_games: int = 5,
) -> dict:
    """
    Compute head-to-head features between two teams from this season's logs.
    Returns dict with h2h win rate, avg point diff, avg total for home team.
    """
    # Find games where these two teams played each other
    home_games = team_logs[
        (team_logs["team_id"] == home_team_id) &
        (team_logs["matchup"].str.contains("vs\\.|@", regex=True))
    ].copy()

    # Filter to games vs this opponent using matchup string
    # matchup looks like "MIA vs. CHA" or "MIA @ CHA"
    teams_df = pd.DataFrame(nba_teams_static.get_teams())
    away_abbr_rows = teams_df[teams_df["id"] == away_team_id]
    if away_abbr_rows.empty:
        return {"h2h_win_rate": 0.5, "h2h_avg_diff": 0.0, "h2h_avg_total": 220.0, "h2h_games": 0}

    away_abbr = away_abbr_rows.iloc[0]["abbreviation"]

    h2h = home_games[home_games["matchup"].str.contains(away_abbr, case=False, na=False)]
    h2h = h2h.sort_values("game_date").tail(n_games)

    if h2h.empty:
        return {"h2h_win_rate": 0.5, "h2h_avg_diff": 0.0, "h2h_avg_total": 220.0, "h2h_games": 0}

    # Get opponent pts for each h2h game
    opp_logs = team_logs[
        (team_logs["team_id"] == away_team_id) &
        (team_logs["game_id"].isin(h2h["game_id"]))
    ][["game_id", "pts"]].rename(columns={"pts": "opp_pts"})

    h2h = h2h.merge(opp_logs, on="game_id", how="left")

    wins      = (h2h["wl"] == "W").sum()
    n         = len(h2h)
    avg_diff  = (h2h["pts"] - h2h["opp_pts"]).mean() if "opp_pts" in h2h.columns else 0.0
    avg_total = (h2h["pts"] + h2h["opp_pts"]).mean() if "opp_pts" in h2h.columns else 220.0

    return {
        "h2h_win_rate":  round(wins / n, 3),
        "h2h_avg_diff":  round(avg_diff, 1),
        "h2h_avg_total": round(avg_total, 1),
        "h2h_games":     n,
    }


# ── Closing line storage ───────────────────────────────────────────────────────

def save_closing_lines(odds_df: pd.DataFrame, date: str = None):
    """
    Save today's odds as closing lines for future training use.
    """
    import os, json
    if date is None:
        import datetime
        date = datetime.date.today().isoformat()

    path = os.path.join("logs", f"closing_lines_{date}.json")
    os.makedirs("logs", exist_ok=True)
    odds_df.to_json(path, orient="records")


def load_closing_line(home_team: str, away_team: str, date: str) -> dict:
    """
    Load saved closing line for a specific game.
    Returns dict with home_spread, total_line or empty dict if not found.
    """
    import os, json
    path = os.path.join("logs", f"closing_lines_{date}.json")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_json(path)
        def tlast(n): return str(n).split()[-1].lower()
        match = df[df["home_team"].apply(tlast) == tlast(home_team)]
        if not match.empty:
            m = match.iloc[0]
            return {"home_spread": m.get("home_spread"), "total_line": m.get("total_line")}
    except Exception:
        pass
    return {}



TRAINING_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]


def load_all_data(season: str = CURRENT_SEASON, multi_season: bool = False) -> dict:
    """
    Load all data needed for feature engineering.
    multi_season=True loads 4 seasons for richer model training.
    """
    if multi_season:
        print("Fetching multi-season team game logs (4 seasons)...")
        team_logs = get_team_game_logs_multi(TRAINING_SEASONS)
    else:
        print("Fetching team game logs...")
        team_logs = get_team_game_logs(season)

    # ── Append playoff game logs so rolling features reflect series results ──
    print("Fetching playoff game logs...")
    try:
        playoff_gl = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation="T",
            season_type_all_star="Playoffs",
        )
        _pause()
        playoff_df = playoff_gl.get_data_frames()[0]
        playoff_df.columns = [c.lower() for c in playoff_df.columns]
        playoff_df["game_date"] = pd.to_datetime(playoff_df["game_date"])
        playoff_df["season"] = season
        if not playoff_df.empty:
            team_logs = pd.concat([team_logs, playoff_df], ignore_index=True)
            team_logs = team_logs.sort_values(["team_id", "game_date"]).reset_index(drop=True)
            print(f"  Added {len(playoff_df)} playoff game rows.")
        else:
            print("  No playoff games yet.")
    except Exception as e:
        print(f"  Warning: Could not fetch playoff logs ({e})")

    print("Fetching team ratings...")
    team_ratings = get_team_ratings(season)

    print("Fetching player game logs...")
    player_logs = get_player_game_logs(season)

    print("Fetching player season stats...")
    player_stats = get_player_season_stats(season)

    print("Fetching player advanced stats...")
    player_adv = get_player_advanced_stats(season)

    print("Adding rolling features...")
    team_logs   = add_rolling_team_features(team_logs)
    player_logs = add_rolling_player_features(player_logs)

    print("Fetching tonight's lineups...")
    lineups = get_tonights_lineups()
    teams_df = get_all_teams()
    lineup_strength = get_lineup_strength(lineups, player_stats, teams_df)
    if lineups:
        print(f"  Lineups found for: {list(lineups.keys())}")
    else:
        print("  No lineup data available.")

    print("Fetching injury report...")
    injury_report = get_injury_report()
    key_injuries  = get_key_injuries(injury_report, player_stats)
    if key_injuries:
        print(f"  Key injuries found: {key_injuries}")
    else:
        print("  No key injuries detected.")

    print("Done loading data.")
    return {
        "team_logs":        team_logs,
        "team_ratings":     team_ratings,
        "player_logs":      player_logs,
        "player_stats":     player_stats,
        "player_adv":       player_adv,
        "injury_report":    injury_report,
        "key_injuries":     key_injuries,
        "lineups":          lineups,
        "lineup_strength":  lineup_strength,
    }
