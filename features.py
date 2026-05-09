# features.py
# NBA AlgoHub — Feature Engineering
#
# Builds model-ready feature matrices for:
#   1. Game predictions  (home_score, away_score, spread, total)
#   2. Player props      (pts, reb, ast)

import pandas as pd
import numpy as np
import requests
from config import ODDS_API_KEY, ODDS_BASE_URL, ODDS_SPORT, ODDS_REGION, CURRENT_SEASON

# ── Odds API ───────────────────────────────────────────────────────────────────

def fetch_nba_odds_bovada() -> pd.DataFrame:
    """
    Scrape NBA odds from Bovada — adapted from working NHL scraper.
    No API key required.
    """
    NBA_NAME_MAP = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    }

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.bovada.lv/sports/basketball/nba",
    }
    URL = "https://www.bovada.lv/services/sports/event/coupon/events/A/description/basketball/nba?lang=en"

    def parse_american(price_str) -> int:
        try:
            return int(float(str(price_str).replace("EVEN", "100").strip()))
        except:
            return 0

    rows = []
    try:
        resp = requests.get(URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for group in data:
            for event in group.get("events", []):
                if event.get("live", False):
                    continue

                competitors = event.get("competitors", [])
                home_name = away_name = ""
                for c in competitors:
                    name = c.get("name", "")
                    if c.get("home"):
                        home_name = name
                    else:
                        away_name = name

                home_abbr = NBA_NAME_MAP.get(home_name, "")
                away_abbr = NBA_NAME_MAP.get(away_name, "")
                if not home_abbr or not away_abbr:
                    continue

                row = {
                    "game_id":       event.get("id", ""),
                    "home_team":     home_name,
                    "away_team":     away_name,
                    "home_spread":   np.nan,
                    "total_line":    np.nan,
                    "home_ml":       np.nan,
                    "away_ml":       np.nan,
                    "line_movement": 0.0,
                    "sharp_side":    "none",
                }

                for dg in event.get("displayGroups", []):
                    for market in dg.get("markets", []):
                        desc = market.get("description", "").strip()
                        outcomes = market.get("outcomes", [])

                        if desc == "Moneyline":
                            for o in outcomes:
                                ml = parse_american(o.get("price", {}).get("american", 0))
                                oname = o.get("description", "")
                                if NBA_NAME_MAP.get(oname) == home_abbr:
                                    row["home_ml"] = ml
                                elif NBA_NAME_MAP.get(oname) == away_abbr:
                                    row["away_ml"] = ml

                        elif desc == "Point Spread":
                            for o in outcomes:
                                oname = o.get("description", "")
                                if NBA_NAME_MAP.get(oname) == home_abbr:
                                    handicap = o.get("price", {}).get("handicap", 0)
                                    try:
                                        row["home_spread"] = float(handicap)
                                    except:
                                        pass

                        elif desc == "Total":
                            for o in outcomes:
                                oname = o.get("description", "").lower()
                                if "over" in oname:
                                    handicap = o.get("price", {}).get("handicap", 0)
                                    try:
                                        line = float(handicap)
                                        # NBA full game totals are 180-280, filter out quarters/halves
                                        if 180 <= line <= 280:
                                            row["total_line"] = line
                                    except:
                                        pass

                rows.append(row)

    except Exception as e:
        print(f"  Warning: Bovada NBA scrape failed ({e})")

    if rows:
        print(f"  Bovada: found {len(rows)} games")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_nba_odds() -> pd.DataFrame:
    """
    Pull today's NBA odds. Tries Odds API first, falls back to Bovada.
    """
    # Try Odds API first
    if ODDS_API_KEY and ODDS_API_KEY != "YOUR_ODDS_API_KEY_HERE":
        try:
            url = f"{ODDS_BASE_URL}/sports/{ODDS_SPORT}/odds"
            params = {
                "apiKey":  ODDS_API_KEY,
                "regions": ODDS_REGION,
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "bookmakers": "pinnacle,draftkings,fanduel,betmgm",
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            rows = []
            for game in data:
                row = {
                    "game_id":   game["id"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "commence":  game["commence_time"],
                }

                # Collect lines from each bookmaker separately
                book_spreads = {}
                book_totals  = {}
                book_home_ml = {}
                book_away_ml = {}

                for book in game.get("bookmakers", []):
                    bname = book["key"]
                    for market in book.get("markets", []):
                        if market["key"] == "h2h":
                            for outcome in market["outcomes"]:
                                if outcome["name"] == game["home_team"]:
                                    book_home_ml[bname] = outcome["price"]
                                else:
                                    book_away_ml[bname] = outcome["price"]
                        elif market["key"] == "spreads":
                            for outcome in market["outcomes"]:
                                if outcome["name"] == game["home_team"]:
                                    book_spreads[bname] = outcome["point"]
                        elif market["key"] == "totals":
                            if market["outcomes"]:
                                book_totals[bname] = market["outcomes"][0]["point"]

                # Consensus line = average across books
                row["home_spread"] = np.mean(list(book_spreads.values())) if book_spreads else np.nan
                row["total_line"]  = np.mean(list(book_totals.values()))  if book_totals  else np.nan
                row["home_ml"]     = list(book_home_ml.values())[0] if book_home_ml else np.nan
                row["away_ml"]     = list(book_away_ml.values())[0] if book_away_ml else np.nan

                # Line movement: Pinnacle vs public books
                pin_spread  = book_spreads.get("pinnacle", np.nan)
                pub_spreads = [v for k, v in book_spreads.items() if k != "pinnacle"]
                pub_avg     = np.mean(pub_spreads) if pub_spreads else np.nan

                if not np.isnan(pin_spread) and not np.isnan(pub_avg):
                    row["line_movement"] = round(pin_spread - pub_avg, 1)
                    row["sharp_side"] = "home" if row["line_movement"] > 0.5 else (
                                        "away" if row["line_movement"] < -0.5 else "none")
                else:
                    row["line_movement"] = 0.0
                    row["sharp_side"]    = "none"

                # Use Pinnacle as primary if available
                if not np.isnan(pin_spread):
                    row["home_spread"] = pin_spread
                if "pinnacle" in book_totals:
                    row["total_line"] = book_totals["pinnacle"]
                if "pinnacle" in book_home_ml:
                    row["home_ml"] = book_home_ml["pinnacle"]
                if "pinnacle" in book_away_ml:
                    row["away_ml"] = book_away_ml["pinnacle"]

                rows.append(row)

            return pd.DataFrame(rows)

        except Exception as e:
            print(f"  Odds API failed ({e}), falling back to Bovada...")

    # Bovada fallback
    print("  Fetching odds from Bovada...")
    return fetch_nba_odds_bovada()


def fetch_nba_props(player_names: list) -> pd.DataFrame:
    """
    Pull player prop odds (pts/reb/ast) for today's NBA games.
    Returns a DataFrame with one row per player per prop market.
    """
    url = f"{ODDS_BASE_URL}/sports/{ODDS_SPORT}/events"
    params = {"apiKey": ODDS_API_KEY, "regions": ODDS_REGION}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    events = resp.json()

    rows = []
    prop_markets = ["player_points", "player_rebounds", "player_assists"]

    for event in events:
        event_id = event["id"]
        props_url = f"{ODDS_BASE_URL}/sports/{ODDS_SPORT}/events/{event_id}/odds"
        props_params = {
            "apiKey":  ODDS_API_KEY,
            "regions": ODDS_REGION,
            "markets": ",".join(prop_markets),
            "oddsFormat": "american",
        }
        try:
            pr = requests.get(props_url, params=props_params, timeout=10)
            pr.raise_for_status()
            pdata = pr.json()
        except Exception:
            continue

        for book in pdata.get("bookmakers", []):
            for market in book.get("markets", []):
                prop_type = market["key"]  # e.g. player_points
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "game_id":    event_id,
                        "home_team":  pdata.get("home_team"),
                        "away_team":  pdata.get("away_team"),
                        "player":     outcome["description"],
                        "prop_type":  prop_type,
                        "line":       outcome.get("point"),
                        "over_odds":  outcome["price"] if outcome["name"] == "Over" else None,
                        "under_odds": outcome["price"] if outcome["name"] == "Under" else None,
                    })
            break  # first bookmaker

    df = pd.DataFrame(rows)
    return df


# ── Implied probability helpers ────────────────────────────────────────────────

def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def prob_to_american(prob: float) -> int:
    """Convert probability to American odds."""
    if prob >= 0.5:
        return int(-prob / (1 - prob) * 100)
    else:
        return int((1 - prob) / prob * 100)


def compute_rolling_ratings(team_logs: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Compute rolling advanced-style ratings from raw game logs.
    Uses last N games instead of full-season averages — reflects post-trade reality.

    Derived metrics:
      - roll_off_rating  : points scored per 100 possessions (approx)
      - roll_def_rating  : points allowed per 100 possessions (approx)
      - roll_net_rating  : off - def
      - roll_pace        : estimated possessions per game
      - roll_ts_pct      : true shooting %

    These replace leaguedashteamstats season-level ratings.
    """
    # Need opponent points — find them by matching game_id across teams
    logs = team_logs.sort_values(["team_id", "game_date"]).copy()

    # Build opponent pts lookup: for each (game_id, team_id) → opp pts
    game_team_pts = logs[["game_id", "team_id", "pts"]].copy()
    game_team_pts = game_team_pts.rename(columns={"team_id": "opp_team_id", "pts": "opp_pts"})

    logs = logs.merge(
        game_team_pts,
        on="game_id",
        how="left"
    )
    # Remove self-match
    logs = logs[logs["team_id"] != logs["opp_team_id"]]

    # Approximate possessions: FGA - OREB + TOV + 0.44*FTA
    poss_cols = ["fga", "oreb", "tov", "fta"]
    has_poss = all(c in logs.columns for c in poss_cols)

    ratings = {}
    for tid, grp in logs.groupby("team_id"):
        grp = grp.sort_values("game_date")

        # Rolling window using ewm for recency weighting
        pts_scored  = grp["pts"].shift(1).ewm(span=window, min_periods=5).mean()
        pts_allowed = grp["opp_pts"].shift(1).ewm(span=window, min_periods=5).mean()

        if has_poss:
            poss = (grp["fga"] - grp["oreb"] + grp["tov"] + 0.44 * grp["fta"])
            poss_roll = poss.shift(1).ewm(span=window, min_periods=5).mean()
            off_rtg = (pts_scored / poss_roll * 100).iloc[-1]
            def_rtg = (pts_allowed / poss_roll * 100).iloc[-1]
            pace    = poss_roll.iloc[-1] * 2  # both teams
        else:
            # Fallback: use raw pts as proxy
            off_rtg = pts_scored.iloc[-1]
            def_rtg = pts_allowed.iloc[-1]
            pace    = 98.0  # NBA average

        net_rtg = off_rtg - def_rtg

        # True shooting %: pts / (2 * (FGA + 0.44*FTA))
        if "fga" in grp.columns and "fta" in grp.columns:
            ts_num  = grp["pts"].shift(1).ewm(span=window, min_periods=5).mean()
            ts_den  = (2 * (grp["fga"] + 0.44 * grp["fta"])).shift(1).ewm(span=window, min_periods=5).mean()
            ts_pct  = (ts_num / ts_den).iloc[-1]
        else:
            ts_pct = np.nan

        ratings[tid] = {
            "team_id":    tid,
            "off_rating": round(off_rtg, 2),
            "def_rating": round(def_rtg, 2),
            "net_rating": round(net_rtg, 2),
            "pace":       round(pace, 2),
            "ts_pct":     round(ts_pct, 4) if not np.isnan(ts_pct) else np.nan,
        }

    return pd.DataFrame(list(ratings.values()))


# ── Game feature builder ───────────────────────────────────────────────────────

def build_game_features(
    team_logs: pd.DataFrame,
    team_ratings: pd.DataFrame,
    upcoming_games: pd.DataFrame,
    team_lookup: pd.DataFrame,
    rest_days: dict = None,
    lineup_strength: dict = None,    # {team_abbr: starter_pts_sum}
) -> pd.DataFrame:
    """
    Build a feature row for each upcoming game.
    Uses rolling 30-game ratings instead of full-season — reflects current roster.
    """
    # ── Recency-weighted rolling stats (7-game EWM) ───────────────────────────
    stat_cols = ["pts", "fg_pct", "fg3_pct", "ft_pct", "ast", "reb", "tov", "plus_minus"]
    stat_cols = [c for c in stat_cols if c in team_logs.columns]

    team_logs = team_logs.sort_values(["team_id", "game_date"])

    latest_weighted = {}
    for tid, grp in team_logs.groupby("team_id"):
        row = {"team_id": tid}
        for col in stat_cols:
            if col in grp.columns:
                val = grp[col].shift(1).ewm(span=7, min_periods=3).mean()
                row[f"rolling_{col}"] = val.iloc[-1] if len(val) > 0 else np.nan
        latest_weighted[tid] = row

    latest_df = pd.DataFrame(list(latest_weighted.values()))
    rolling_cols = [c for c in latest_df.columns if c.startswith("rolling_")]

    # ── Rolling 30-game advanced ratings — post-trade accurate ────────────────
    print("  Computing rolling team ratings (last 30 games)...")
    ratings = compute_rolling_ratings(team_logs, window=30)

    team_profile = latest_df.merge(ratings, on="team_id", how="left")

    rows = []
    for _, game in upcoming_games.iterrows():
        home_id = int(game["home_team_id"])
        away_id = int(game["visitor_team_id"])

        home = team_profile[team_profile["team_id"] == home_id]
        away = team_profile[team_profile["team_id"] == away_id]

        if home.empty or away.empty:
            continue

        home = home.iloc[0]
        away = away.iloc[0]

        row = {"game_id": game["game_id"]}

        # Rolling stat differentials
        for col in rolling_cols:
            stat = col.replace("rolling_", "")
            row[f"diff_{stat}"] = home.get(col, np.nan) - away.get(col, np.nan)
            row[f"home_{stat}"] = home.get(col, np.nan)
            row[f"away_{stat}"] = away.get(col, np.nan)

        # Advanced ratings
        for metric in ["off_rating", "def_rating", "net_rating", "pace", "ts_pct"]:
            row[f"home_{metric}"] = home.get(metric, np.nan)
            row[f"away_{metric}"] = away.get(metric, np.nan)
            row[f"diff_{metric}"] = home.get(metric, np.nan) - away.get(metric, np.nan)

        row["avg_pace"] = (home.get("pace", 98) + away.get("pace", 98)) / 2

        # ── Rest days ─────────────────────────────────────────────────────────
        home_rest = rest_days.get(home_id, 2) if rest_days else 2
        away_rest = rest_days.get(away_id, 2) if rest_days else 2
        row["home_rest_days"]  = home_rest
        row["away_rest_days"]  = away_rest
        row["rest_advantage"]  = home_rest - away_rest
        row["home_b2b"]        = int(home_rest == 0)
        row["away_b2b"]        = int(away_rest == 0)

        # ── Home court advantage — arena-calibrated + series-aware ──────────────
        # The home_team_id from the schedule IS the actual home team for this game.
        # In a playoff series this switches — Game 3 MIN is home even though DEN
        # had home court in Games 1&2. The schedule always reflects the correct
        # home team so we just need to look up the right arena.
        ARENA_HCA = {
            1610612743: 3.8,   # Denver Nuggets — altitude
            1610612749: 3.5,   # Milwaukee Bucks
            1610612738: 3.2,   # Boston Celtics
            1610612760: 3.2,   # OKC Thunder
            1610612750: 3.0,   # Minnesota Timberwolves
            1610612748: 2.5,   # Miami Heat
            1610612766: 2.5,   # Charlotte Hornets
            1610612741: 2.5,   # Chicago Bulls
            1610612739: 2.5,   # Cleveland Cavaliers
            1610612757: 2.3,   # Portland Trail Blazers
            1610612756: 2.2,   # Phoenix Suns
            1610612746: 1.8,   # LA Clippers
            1610612747: 1.8,   # LA Lakers
        }
        base_hca = ARENA_HCA.get(home_id, 2.5)
        hca = base_hca - (1.5 * row["home_b2b"]) + (1.0 * row["away_b2b"])
        row["home_advantage"] = hca

        # Series momentum: did the home team win their last game?
        # Gives a small boost to the team that won last
        home_last = team_logs[team_logs["team_id"] == home_id].sort_values("game_date")
        away_last = team_logs[team_logs["team_id"] == away_id].sort_values("game_date")
        home_won_last = int(home_last.iloc[-1]["wl"] == "W") if not home_last.empty else 0
        away_won_last = int(away_last.iloc[-1]["wl"] == "W") if not away_last.empty else 0
        row["home_momentum"] = home_won_last - away_won_last  # +1 home won, -1 away won, 0 split

        row["home_team_id"] = home_id
        row["away_team_id"] = away_id

        # ── Lineup strength adjustment ─────────────────────────────────────────
        abbr_map = dict(zip(team_lookup["id"], team_lookup["abbreviation"]))
        home_abbr = abbr_map.get(home_id, "")
        away_abbr = abbr_map.get(away_id, "")

        if lineup_strength:
            home_ls = lineup_strength.get(home_abbr, np.nan)
            away_ls = lineup_strength.get(away_abbr, np.nan)
            row["home_lineup_pts"]  = home_ls if not pd.isna(home_ls) else 0
            row["away_lineup_pts"]  = away_ls if not pd.isna(away_ls) else 0
            row["lineup_pts_diff"]  = row["home_lineup_pts"] - row["away_lineup_pts"]
        else:
            row["home_lineup_pts"] = 0
            row["away_lineup_pts"] = 0
            row["lineup_pts_diff"] = 0

        # ── Head-to-head features ──────────────────────────────────────────────
        from data import get_h2h_features
        h2h = get_h2h_features(team_logs, home_id, away_id)
        row["h2h_win_rate"]  = h2h["h2h_win_rate"]
        row["h2h_avg_diff"]  = h2h["h2h_avg_diff"]
        row["h2h_avg_total"] = h2h["h2h_avg_total"]

        rows.append(row)

    features_df = pd.DataFrame(rows)

    # ── Inject market line as a feature ───────────────────────────────────────
    # The market line is the single best predictor — model learns to predict
    # deviations from it rather than raw scores from scratch.
    try:
        odds_df = fetch_nba_odds()
        if not odds_df.empty:
            def team_last(name):
                return str(name).split()[-1].lower()

            id_to_name = dict(zip(team_lookup["id"], team_lookup["full_name"]))

            for idx, frow in features_df.iterrows():
                home_name = id_to_name.get(int(frow["home_team_id"]), "")
                away_name = id_to_name.get(int(frow["away_team_id"]), "")

                match = odds_df[
                    odds_df["home_team"].apply(team_last) == team_last(home_name)
                ]
                if not match.empty and not pd.isna(match.iloc[0].get("away_team")):
                    match = match[
                        match["away_team"].apply(team_last) == team_last(away_name)
                    ]

                if not match.empty:
                    m = match.iloc[0]
                    # Market spread: negative = home favored
                    mspread = m.get("home_spread", np.nan)
                    mtotal  = m.get("total_line", np.nan)
                    # Convert spread to implied home margin
                    features_df.at[idx, "market_spread"]       = mspread if not pd.isna(mspread) else 0.0
                    features_df.at[idx, "market_total"]        = mtotal  if not pd.isna(mtotal)  else 220.0
                    features_df.at[idx, "market_home_implied"] = -mspread if not pd.isna(mspread) else 0.0
                else:
                    features_df.at[idx, "market_spread"]       = 0.0
                    features_df.at[idx, "market_total"]        = 220.0
                    features_df.at[idx, "market_home_implied"] = 0.0
    except Exception as e:
        print(f"  Warning: Could not inject market line into features ({e})")
        features_df["market_spread"]       = 0.0
        features_df["market_total"]        = 220.0
        features_df["market_home_implied"] = 0.0

    return features_df


# ── Training feature builder ───────────────────────────────────────────────────

def build_training_game_features(team_logs: pd.DataFrame, team_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Build historical game-level feature rows for model training.
    Uses rolling 30-game ratings instead of full-season — post-trade accurate.
    """
    if "matchup" not in team_logs.columns:
        return pd.DataFrame()

    home_games = team_logs[team_logs["matchup"].str.contains("vs\\.", regex=True)].copy()
    away_games = team_logs[team_logs["matchup"].str.contains("@")].copy()

    # ── Pre-compute rolling ratings per team per game date ────────────────────
    # For each game, use ratings from the 30 games BEFORE that game
    print("  Pre-computing rolling ratings for training...")
    logs_sorted = team_logs.sort_values(["team_id", "game_date"]).copy()

    # Merge opponent pts for def rating calculation
    game_pts = logs_sorted[["game_id", "team_id", "pts"]].rename(
        columns={"team_id": "opp_team_id", "pts": "opp_pts"}
    )
    logs_sorted = logs_sorted.merge(game_pts, on="game_id", how="left")
    logs_sorted = logs_sorted[logs_sorted["team_id"] != logs_sorted["opp_team_id"]]

    has_poss = all(c in logs_sorted.columns for c in ["fga", "oreb", "tov", "fta"])

    # Rolling per-team rating history
    rating_history = {}
    for tid, grp in logs_sorted.groupby("team_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        pts_roll = grp["pts"].shift(1).ewm(span=30, min_periods=5).mean()
        opp_roll = grp["opp_pts"].shift(1).ewm(span=30, min_periods=5).mean()

        if has_poss:
            poss = grp["fga"] - grp["oreb"] + grp["tov"] + 0.44 * grp["fta"]
            poss_roll = poss.shift(1).ewm(span=30, min_periods=5).mean()
            off_rtg = (pts_roll / poss_roll * 100)
            def_rtg = (opp_roll / poss_roll * 100)
            pace    = poss_roll * 2
        else:
            off_rtg = pts_roll
            def_rtg = opp_roll
            pace    = pd.Series([98.0] * len(grp))

        if "fga" in grp.columns and "fta" in grp.columns:
            ts_num = grp["pts"].shift(1).ewm(span=30, min_periods=5).mean()
            ts_den = (2 * (grp["fga"] + 0.44 * grp["fta"])).shift(1).ewm(span=30, min_periods=5).mean()
            ts_pct = ts_num / ts_den
        else:
            ts_pct = pd.Series([np.nan] * len(grp))

        grp["r_off_rating"] = off_rtg
        grp["r_def_rating"] = def_rtg
        grp["r_net_rating"] = off_rtg - def_rtg
        grp["r_pace"]       = pace
        grp["r_ts_pct"]     = ts_pct
        rating_history[tid] = grp.set_index("game_id")[["r_off_rating","r_def_rating","r_net_rating","r_pace","r_ts_pct"]]

    # Rest days
    logs_rest = team_logs.sort_values(["team_id", "game_date"]).copy()
    logs_rest["prev_game_date"] = logs_rest.groupby("team_id")["game_date"].shift(1)
    logs_rest["rest_days"] = (logs_rest["game_date"] - logs_rest["prev_game_date"]).dt.days - 1
    logs_rest["rest_days"] = logs_rest["rest_days"].fillna(3).clip(0, 7)
    rest_lookup = logs_rest.set_index(["team_id", "game_id"])["rest_days"].to_dict()

    rows = []
    for _, hg in home_games.iterrows():
        game_id = hg["game_id"]
        ag_rows = away_games[away_games["game_id"] == game_id]
        if ag_rows.empty:
            continue
        ag = ag_rows.iloc[0]

        rolling_cols = [c for c in hg.index if c.startswith("rolling_")]

        home_rest = rest_lookup.get((hg["team_id"], game_id), 2)
        away_rest = rest_lookup.get((ag["team_id"], game_id), 2)

        # Get rolling ratings at game time
        def get_rating(tid, gid, col):
            hist = rating_history.get(tid)
            if hist is None or gid not in hist.index:
                return np.nan
            return hist.loc[gid, col]

        ARENA_HCA = {
            1610612743: 3.8, 1610612749: 3.5, 1610612738: 3.2,
            1610612760: 3.2, 1610612750: 3.0, 1610612748: 2.5,
            1610612766: 2.5, 1610612757: 2.3, 1610612756: 2.2,
            1610612746: 1.8, 1610612747: 1.8,
        }
        base_hca = ARENA_HCA.get(int(hg["team_id"]), 2.5)
        hca = base_hca - (1.5 * int(home_rest == 0)) + (1.0 * int(away_rest == 0))

        row = {
            "game_id":        game_id,
            "game_date":      hg["game_date"],
            "home_pts":       hg["pts"],
            "away_pts":       ag["pts"],
            "total_pts":      hg["pts"] + ag["pts"],
            "home_win":       int(hg["wl"] == "W"),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_advantage": home_rest - away_rest,
            "home_b2b":       int(home_rest == 0),
            "away_b2b":       int(away_rest == 0),
            "home_advantage": hca,
        }

        for col in rolling_cols:
            stat = col.replace("rolling_", "")
            row[f"home_{stat}"] = hg.get(col, np.nan)
            row[f"away_{stat}"] = ag.get(col, np.nan)
            row[f"diff_{stat}"] = hg.get(col, np.nan) - ag.get(col, np.nan)

        for metric in ["off_rating", "def_rating", "net_rating", "pace", "ts_pct"]:
            h_val = get_rating(hg["team_id"], game_id, f"r_{metric}")
            a_val = get_rating(ag["team_id"], game_id, f"r_{metric}")
            row[f"home_{metric}"] = h_val
            row[f"away_{metric}"] = a_val
            row[f"diff_{metric}"] = h_val - a_val if not (np.isnan(h_val) or np.isnan(a_val)) else np.nan

        row["avg_pace"] = (
            get_rating(hg["team_id"], game_id, "r_pace") +
            get_rating(ag["team_id"], game_id, "r_pace")
        ) / 2

        # Market spread proxy: net rating diff → implied spread
        h_net = get_rating(hg["team_id"], game_id, "r_net_rating")
        a_net = get_rating(ag["team_id"], game_id, "r_net_rating")
        net_diff = (h_net - a_net) if not (np.isnan(h_net) or np.isnan(a_net)) else 0.0
        row["market_home_implied"] = (net_diff / 2.5) + hca
        row["market_spread"]       = -row["market_home_implied"]
        row["market_total"]        = 220.0

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    non_target = [c for c in df.columns if c not in ["game_id","game_date","home_pts","away_pts","total_pts","home_win"]]
    df = df.dropna(subset=non_target)
    return df


# ── Player prop feature builder ────────────────────────────────────────────────

def build_prop_features(
    player_logs:   pd.DataFrame,
    player_stats:  pd.DataFrame,
    player_adv:    pd.DataFrame,
    team_ratings:  pd.DataFrame,
    upcoming_games: pd.DataFrame,
    player_team_map: dict,           # player_id -> team_id
    opponent_map:    dict,           # team_id -> opponent team_id for today
) -> pd.DataFrame:
    """
    Build feature rows for player prop predictions.
    One row per player per stat (pts, reb, ast).
    """
    # Advanced: usage rate is key
    adv_slim = player_adv[["player_id", "usg_pct", "off_rating", "def_rating"]].copy()
    base_slim = player_stats[[
        "player_id", "player_name", "team_id",
        "pts", "reb", "ast", "min", "fga", "fg_pct", "fg3a", "fta", "gp",
    ]].copy()

    merged = base_slim.merge(adv_slim, on="player_id", how="left")

    # Defensive rating of opponent (proxy for matchup difficulty)
    opp_def = team_ratings[["team_id", "def_rating"]].copy()
    opp_def.columns = ["opp_team_id", "opp_def_rating"]

    rows = []
    for _, player in merged.iterrows():
        pid     = player["player_id"]
        tid     = player["team_id"]
        opp_id  = opponent_map.get(int(tid) if not pd.isna(tid) else -1, None)
        if opp_id is None:
            continue   # player's team not playing today

        opp_dr = opp_def[opp_def["opp_team_id"] == opp_id]
        opp_def_rating = opp_dr.iloc[0]["opp_def_rating"] if not opp_dr.empty else 110.0

        # Rolling stats for this player
        plogs = player_logs[player_logs["player_id"] == pid].sort_values("game_date")
        if plogs.empty:
            continue

        last = plogs.iloc[-1]
        rolling_pts = last.get("rolling_pts", player["pts"])
        rolling_reb = last.get("rolling_reb", player["reb"])
        rolling_ast = last.get("rolling_ast", player["ast"])
        rolling_min = last.get("rolling_min", player["min"])

        for stat in ["pts", "reb", "ast"]:
            row = {
                "player_id":        pid,
                "player_name":      player["player_name"],
                "team_id":          tid,
                "opp_team_id":      opp_id,
                "stat":             stat,
                # Season averages
                "season_avg":       player[stat],
                "season_min":       player["min"],
                "season_usg":       player.get("usg_pct", np.nan),
                "season_fga":       player.get("fga", np.nan),
                "season_fg_pct":    player.get("fg_pct", np.nan),
                "season_fta":       player.get("fta", np.nan),
                "gp":               player.get("gp", np.nan),
                # Rolling averages
                "rolling_pts":      rolling_pts,
                "rolling_reb":      rolling_reb,
                "rolling_ast":      rolling_ast,
                "rolling_min":      rolling_min,
                # Matchup
                "opp_def_rating":   opp_def_rating,
                # Derived
                "pts_per_min":      player["pts"] / max(player["min"], 1),
                "reb_per_min":      player["reb"] / max(player["min"], 1),
                "ast_per_min":      player["ast"] / max(player["min"], 1),
            }
            rows.append(row)

    return pd.DataFrame(rows)
