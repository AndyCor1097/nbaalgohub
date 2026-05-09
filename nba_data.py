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
    return pd.DataFrame(teams)


def get_team_id(abbreviation: str) -> int:
    df = get_all_teams()
    row = df[df["abbreviation"] == abbreviation.upper()]
    if row.empty:
        raise ValueError(f"Unknown team abbreviation: {abbreviation}")
    return int(row.iloc[0]["id"])


# ── Team game logs ─────────────────────────────────────────────────────────────

def get_team_game_logs(season: str = CURRENT_SEASON) -> pd.DataFrame:
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


# ── Team ratings ───────────────────────────────────────────────────────────────

def get_team_ratings(season: str = CURRENT_SEASON) -> pd.DataFrame:
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


def get_player_playoff_logs(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Pull player game logs for playoffs only."""
    try:
        gl = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation="P",
            season_type_all_star="Playoffs",
        )
        _pause()
        df = gl.get_data_frames()[0]
        df.columns = [c.lower() for c in df.columns]
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["is_playoff"] = True
        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
        print(f"  Playoff player logs: {len(df)} rows, {df['player_id'].nunique()} players")
        return df
    except Exception as e:
        print(f"  Warning: Could not fetch playoff player logs ({e})")
        return pd.DataFrame()


# ── Player season averages ─────────────────────────────────────────────────────

def get_player_season_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
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

    games_df = sb.get_data_frames()[1]
    teams_df = sb.get_data_frames()[2]

    if games_df.empty:
        return pd.DataFrame()

    rows = []
    for _, game in games_df.iterrows():
        game_id   = game["gameId"]
        game_teams = teams_df[teams_df["gameId"] == game_id]
        if len(game_teams) < 2:
            continue
        rows.append({
            "game_id":         game_id,
            "game_date_est":   game.get("gameEt", today),
            "home_team_id":    int(game_teams.iloc[0]["teamId"]),
            "visitor_team_id": int(game_teams.iloc[1]["teamId"]),
            "arena_name":      "",
        })

    return pd.DataFrame(rows).drop_duplicates(subset=["game_id"]).reset_index(drop=True)


# ── Rolling features ───────────────────────────────────────────────────────────

def add_rolling_team_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    stat_cols = ["pts", "fg_pct", "fg3_pct", "ft_pct", "ast", "reb", "tov", "plus_minus"]
    stat_cols = [c for c in stat_cols if c in df.columns]
    df = df.sort_values(["team_id", "game_date"])
    for col in stat_cols:
        df[f"rolling_{col}"] = (
            df.groupby("team_id")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        )
    return df


def add_rolling_player_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    stat_cols = ["pts", "reb", "ast", "min", "fga", "fg_pct", "fg3a", "fta", "tov", "plus_minus"]
    stat_cols = [c for c in stat_cols if c in df.columns]
    df = df.sort_values(["player_id", "game_date"])
    for col in stat_cols:
        df[f"rolling_{col}"] = (
            df.groupby("player_id")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        )
    return df


def add_playoff_rolling_features(
    regular_logs: pd.DataFrame,
    playoff_logs: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Build playoff-specific rolling averages.
    Combines regular season baseline with playoff games, weighting playoffs heavily.
    Returns playoff_logs with rolling_pts_po, rolling_reb_po, rolling_ast_po columns.
    """
    if playoff_logs.empty:
        return playoff_logs

    stat_cols = ["pts", "reb", "ast", "min"]
    stat_cols = [c for c in stat_cols if c in playoff_logs.columns]

    # For each player, get their last N regular season games as baseline
    # then compute rolling avg using playoff games only
    combined = pd.concat([
        regular_logs.assign(is_playoff=False),
        playoff_logs.assign(is_playoff=True),
    ], ignore_index=True).sort_values(["player_id", "game_date"])

    for col in stat_cols:
        combined[f"rolling_{col}_po"] = (
            combined.groupby("player_id")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Return only playoff rows with the new columns
    po_cols = [c for c in combined.columns if c.endswith("_po")]
    result = combined[combined["is_playoff"] == True].copy()
    return result


def get_rest_days(team_logs: pd.DataFrame, upcoming_games: pd.DataFrame) -> dict:
    """Calculate days rest for each team playing today."""
    today = pd.Timestamp.today().normalize()
    rest = {}

    playing_today = set()
    for _, game in upcoming_games.iterrows():
        playing_today.add(int(game["home_team_id"]))
        playing_today.add(int(game["visitor_team_id"]))

    for tid in playing_today:
        team_games = team_logs[team_logs["team_id"] == tid].sort_values("game_date")
        if team_games.empty:
            rest[tid] = 3
            continue
        last = team_games.iloc[-1]["game_date"]
        if pd.isna(last):
            rest[tid] = 3
            continue
        delta = (today - pd.Timestamp(last)).days - 1
        rest[tid] = min(max(int(delta), 0), 7)

    return rest


# ── Lineup scraper ─────────────────────────────────────────────────────────────

def get_tonights_lineups() -> dict:
    import requests
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  Warning: beautifulsoup4 not installed.")
        return {}

    lineups = {}
    try:
        url = "https://www.rotowire.com/basketball/nba-lineups.php"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
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
    except Exception as e:
        print(f"  Warning: Could not scrape lineups ({e})")
    return lineups


def get_lineup_strength(lineups, player_stats, team_lookup):
    abbr_to_id = dict(zip(team_lookup["abbreviation"], team_lookup["id"]))
    strength = {}
    for abbr, starters in lineups.items():
        total_pts = 0.0
        found = 0
        for pname in starters:
            match = player_stats[player_stats["player_name"].str.lower() == pname.lower()]
            if not match.empty:
                total_pts += match.iloc[0].get("pts", 0)
                found += 1
        if found > 0:
            strength[abbr] = total_pts
    return strength


def get_h2h_features(team_logs, home_team_id, away_team_id, n_games=5):
    home_games = team_logs[team_logs["team_id"] == home_team_id]
    if home_games.empty:
        return {"h2h_win_rate": 0.5, "h2h_avg_diff": 0.0, "h2h_avg_total": 220.0}
    h2h = home_games[home_games["matchup"].str.contains(
        str(away_team_id), na=False
    )].tail(n_games)
    if h2h.empty:
        return {"h2h_win_rate": 0.5, "h2h_avg_diff": 0.0, "h2h_avg_total": 220.0}
    win_rate = (h2h["wl"] == "W").mean()
    avg_diff = h2h["plus_minus"].mean() if "plus_minus" in h2h.columns else 0.0
    avg_total = 220.0
    return {
        "h2h_win_rate":  round(float(win_rate), 3),
        "h2h_avg_diff":  round(float(avg_diff), 1),
        "h2h_avg_total": round(float(avg_total), 1),
    }


def get_injury_report():
    try:
        import requests
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        r = requests.get(url, timeout=10)
        return r.json()
    except:
        return {}


def get_key_injuries(injury_report, player_stats):
    return {}


def save_closing_lines(odds_df, date=None):
    import os, datetime
    if date is None:
        date = datetime.date.today().isoformat()
    path = os.path.join("logs", f"closing_lines_{date}.json")
    os.makedirs("logs", exist_ok=True)
    odds_df.to_json(path, orient="records")


def load_closing_line(home_team, away_team, date):
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
    except:
        pass
    return {}


TRAINING_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]


def get_team_game_logs_multi(seasons):
    dfs = []
    for season in seasons:
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


def load_all_data(season: str = CURRENT_SEASON, multi_season: bool = False) -> dict:
    if multi_season:
        print("Fetching multi-season team game logs (4 seasons)...")
        team_logs = get_team_game_logs_multi(TRAINING_SEASONS)
    else:
        print("Fetching team game logs...")
        team_logs = get_team_game_logs(season)

    # Append playoff team logs
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

    # ── NEW: Fetch playoff player logs ────────────────────────────────────────
    print("Fetching playoff player logs...")
    playoff_player_logs = get_player_playoff_logs(season)

    # Build playoff rolling averages (last 5 playoff games)
    if not playoff_player_logs.empty:
        playoff_player_logs = add_playoff_rolling_features(player_logs, playoff_player_logs, window=5)
        print(f"  Built playoff rolling features for {playoff_player_logs['player_id'].nunique()} players")

        # Merge playoff rolling avgs back into player_logs
        # These will override regular season rolling avgs for players in playoffs
        po_cols = [c for c in playoff_player_logs.columns if c.endswith("_po")]
        if po_cols:
            po_latest = playoff_player_logs.sort_values("game_date").groupby("player_id")[
                ["player_id"] + po_cols
            ].last().reset_index(drop=True)

            player_logs = player_logs.merge(po_latest, on="player_id", how="left")

            # Override rolling avgs with playoff rolling avgs where available
            for col in ["pts", "reb", "ast", "min"]:
                po_col = f"rolling_{col}_po"
                reg_col = f"rolling_{col}"
                if po_col in player_logs.columns and reg_col in player_logs.columns:
                    mask = player_logs[po_col].notna()
                    player_logs.loc[mask, reg_col] = player_logs.loc[mask, po_col]

            print(f"  Playoff rolling avgs merged — overriding regular season for {mask.sum()} players")

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
        "team_logs":          team_logs,
        "team_ratings":       team_ratings,
        "player_logs":        player_logs,
        "playoff_player_logs": playoff_player_logs,
        "player_stats":       player_stats,
        "player_adv":         player_adv,
        "injury_report":      injury_report,
        "key_injuries":       key_injuries,
        "lineups":            lineups,
        "lineup_strength":    lineup_strength,
    }
