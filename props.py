# props.py
# NBA AlgoHub — Player Prop Predictions
#
# Predicts pts/reb/ast for players in today's games
# and compares projections to market prop lines

import pandas as pd
import numpy as np

from features import fetch_nba_props, build_prop_features
from models import load_prop_models
from config import PROP_MIN_EDGE, MIN_GAMES


STAT_LABELS = {
    "pts":  "Points",
    "reb":  "Rebounds",
    "ast":  "Assists",
}


def build_opponent_map(upcoming_games: pd.DataFrame) -> dict:
    """
    Given today's game schedule, return a dict mapping each team_id
    to its opponent's team_id.
    """
    opp_map = {}
    for _, game in upcoming_games.iterrows():
        h = int(game["home_team_id"])
        a = int(game["visitor_team_id"])
        opp_map[h] = a
        opp_map[a] = h
    return opp_map


def generate_prop_picks(
    player_logs:    pd.DataFrame,
    player_stats:   pd.DataFrame,
    player_adv:     pd.DataFrame,
    team_ratings:   pd.DataFrame,
    upcoming_games: pd.DataFrame,
    min_games:      int = MIN_GAMES,
) -> pd.DataFrame:
    """
    Returns a DataFrame of player prop projections and picks for today.
    """
    prop_models, feat_cols = load_prop_models()
    if not prop_models:
        print("No prop models found. Run training first.")
        return pd.DataFrame()

    # Filter players with enough games
    player_stats = player_stats[player_stats["gp"] >= min_games].copy()

    # Build opponent map
    opponent_map = build_opponent_map(upcoming_games)
    if not opponent_map:
        print("No upcoming games found — no prop picks.")
        return pd.DataFrame()

    # Build features
    prop_features = build_prop_features(
        player_logs    = player_logs,
        player_stats   = player_stats,
        player_adv     = player_adv,
        team_ratings   = team_ratings,
        upcoming_games = upcoming_games,
        player_team_map = {},
        opponent_map   = opponent_map,
    )

    if prop_features.empty:
        print("No prop features built.")
        return pd.DataFrame()

    # Fetch prop lines
    try:
        prop_lines = fetch_nba_props(player_stats["player_name"].tolist())
    except Exception as e:
        print(f"Warning: Could not fetch prop lines ({e}). Showing projections only.")
        prop_lines = pd.DataFrame()

    results = []
    for stat, model in prop_models.items():
        df_stat = prop_features[prop_features["stat"] == stat].copy()
        if df_stat.empty:
            continue

        # Align features
        X = df_stat[[c for c in feat_cols if c in df_stat.columns]].fillna(0)
        missing = [c for c in feat_cols if c not in df_stat.columns]
        for col in missing:
            X[col] = 0
        X = X[feat_cols]

        projections = model.predict(X)
        df_stat = df_stat.copy()
        df_stat["projection"] = np.round(projections, 1)

        # Match to prop lines
        for _, row in df_stat.iterrows():
            pname = row["player_name"]
            proj  = row["projection"]
            season_avg = row["season_avg"]

            pick_row = {
                "player":      pname,
                "stat":        STAT_LABELS[stat],
                "projection":  proj,
                "season_avg":  round(season_avg, 1),
                "line":        np.nan,
                "edge":        np.nan,
                "direction":   None,
                "over_odds":   np.nan,
                "under_odds":  np.nan,
            }

            # Find matching prop line
            if not prop_lines.empty:
                match = prop_lines[
                    (prop_lines["player"].str.lower() == pname.lower()) &
                    (prop_lines["prop_type"] == f"player_{stat}")
                ]
                if not match.empty:
                    m = match.iloc[0]
                    line = m.get("line", np.nan)
                    if not pd.isna(line):
                        pick_row["line"]      = line
                        pick_row["edge"]      = round(abs(proj - line), 1)
                        pick_row["direction"] = "Over" if proj > line else "Under"
                        pick_row["over_odds"]  = m.get("over_odds", -110)
                        pick_row["under_odds"] = m.get("under_odds", -110)

            results.append(pick_row)

    df = pd.DataFrame(results)
    return df


def filter_top_props(props_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return only props with edge >= PROP_MIN_EDGE, sorted by edge desc."""
    if props_df.empty:
        return props_df
    filtered = props_df[
        props_df["edge"].notna() & (props_df["edge"] >= PROP_MIN_EDGE)
    ].copy()
    return filtered.sort_values("edge", ascending=False).head(top_n)


def print_props(props_df: pd.DataFrame):
    """Pretty-print player prop picks — with fallback to raw projections if no lines available."""
    print("\n" + "=" * 60)
    print("🏀  NBA ALGOHUB — PLAYER PROP PROJECTIONS")
    print("=" * 60)

    if props_df.empty:
        print("\n  No prop data available.")
        print("=" * 60)
        return

    top = filter_top_props(props_df)

    if not top.empty:
        # Has market lines with edge
        print()
        for _, row in top.iterrows():
            direction = row.get("direction", "")
            line      = row.get("line", "?")
            edge      = row.get("edge", "?")
            odds_key  = "over_odds" if direction == "Over" else "under_odds"
            odds      = row.get(odds_key, -110)
            print(f"  {row['player']}  —  {row['stat']}")
            print(f"    Projection: {row['projection']}  |  Season Avg: {row['season_avg']}  |  Line: {line}")
            print(f"    ✅ {direction}  (Edge: +{edge})  Odds: {odds}")
            print()
    else:
        # No market lines — show top projections vs season avg as reference
        print("\n  ℹ️  No prop lines available (requires paid Odds API tier).")
        print("  Showing top model projections vs season averages:\n")

        for stat_label in ["Points", "Rebounds", "Assists"]:
            subset = props_df[props_df["stat"] == stat_label].copy()
            if subset.empty:
                continue
            # Show players where projection differs most from season avg
            subset["variance"] = abs(subset["projection"] - subset["season_avg"])
            top_var = subset.nlargest(5, "variance")
            print(f"  ── {stat_label} ──")
            for _, row in top_var.iterrows():
                diff = row["projection"] - row["season_avg"]
                arrow = "▲" if diff > 0 else "▼"
                print(f"    {row['player']:<25}  Proj: {row['projection']:.1f}  "
                      f"Avg: {row['season_avg']:.1f}  {arrow} {abs(diff):.1f}")
            print()

    print("=" * 60)
