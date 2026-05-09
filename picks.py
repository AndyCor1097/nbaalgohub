# picks.py
# NBA AlgoHub — Daily Game Picks
#
# Generates ML, spread, and total picks with edge % vs. Vegas lines

import pandas as pd
import numpy as np

from features import fetch_nba_odds, american_to_prob, prob_to_american, build_game_features
from models import load_game_models
from config import MIN_EDGE_PCT


# ── Playoff direct projection ─────────────────────────────────────────────────

def project_scores_playoff(
    team_logs:      pd.DataFrame,
    home_id:        int,
    away_id:        int,
    rest_days:      dict = None,
    team_lookup:    pd.DataFrame = None,
) -> tuple:
    """
    Bypass XGBoost entirely in playoffs.
    Uses last 5 games rolling avg (including playoff games) + HCA + momentum.
    Much more accurate for small-sample postseason series.
    """
    ARENA_HCA = {
        1610612743: 3.8, 1610612749: 3.5, 1610612738: 3.2,
        1610612760: 3.2, 1610612750: 3.0, 1610612748: 2.5,
        1610612741: 2.5, 1610612739: 2.5, 1610612757: 2.3,
        1610612756: 2.2, 1610612746: 1.8, 1610612747: 1.8,
    }

    def get_rolling(tid, window=5):
        logs = team_logs[team_logs["team_id"] == tid].sort_values("game_date")
        if len(logs) < 3:
            return None, None
        recent = logs.tail(window)
        pts_avg = recent["pts"].mean()
        # Get opponent pts for defensive rating proxy
        opp_logs = team_logs[
            (team_logs["game_id"].isin(recent["game_id"])) &
            (team_logs["team_id"] != tid)
        ]
        opp_pts = opp_logs["pts"].mean() if not opp_logs.empty else 112.0
        last_wl = logs.iloc[-1]["wl"] if "wl" in logs.columns else "W"
        return pts_avg, opp_pts, last_wl

    home_stats = get_rolling(home_id)
    away_stats = get_rolling(away_id)

    if home_stats[0] is None or away_stats[0] is None:
        return None, None

    home_pts_avg, home_opp_avg, home_last_wl = home_stats
    away_pts_avg, away_opp_avg, away_last_wl = away_stats

    # Blend team's own scoring with opponent's defensive avg
    # 60% own offense, 40% opponent defense
    home_proj = (home_pts_avg * 0.6) + (away_opp_avg * 0.4)
    away_proj = (away_pts_avg * 0.6) + (home_opp_avg * 0.4)

    # HCA — home team is whoever is listed as home in the schedule
    base_hca = ARENA_HCA.get(home_id, 2.5)
    home_rest = rest_days.get(home_id, 2) if rest_days else 2
    away_rest = rest_days.get(away_id, 2) if rest_days else 2
    hca = base_hca - (1.5 * int(home_rest == 0)) + (1.0 * int(away_rest == 0))
    home_proj += hca / 2
    away_proj -= hca / 2

    # Momentum: +1.5 pts to team that won last game
    if home_last_wl == "W":
        home_proj += 1.5
    if away_last_wl == "W":
        away_proj += 1.5

    # Playoff pace shade — lower scoring than regular season
    home_proj -= 6.0
    away_proj -= 6.0

    return round(home_proj, 1), round(away_proj, 1)


# ── Score projection ──────────────────────────────────────────────────────────

def project_scores(X: pd.DataFrame, home_model, away_model) -> tuple:
    """Return (home_proj, away_proj) arrays."""
    home_proj = home_model.predict(X)
    away_proj = away_model.predict(X)
    return home_proj, away_proj


# ── Edge calculation ───────────────────────────────────────────────────────────

def calc_ml_edge(model_prob: float, market_odds: float) -> float:
    """
    Edge % = model implied prob - market implied prob.
    Positive = model thinks this side is more likely than the market.
    """
    market_prob = american_to_prob(market_odds)
    if np.isnan(market_prob):
        return np.nan
    return (model_prob - market_prob) * 100


def calc_spread_edge(proj_diff: float, market_spread: float) -> float:
    """
    proj_diff  = home_proj - away_proj
    market_spread = home spread (negative = home favored)
    Edge = proj_diff - (-market_spread)  [convert spread to 'home beats by' framing]
    """
    return proj_diff - (-market_spread)


def calc_total_edge(proj_total: float, market_total: float) -> float:
    """Returns positive = model over, negative = model under."""
    return proj_total - market_total


# ── Main picks generator ───────────────────────────────────────────────────────

def generate_game_picks(
    team_logs:        pd.DataFrame,
    team_ratings:     pd.DataFrame,
    upcoming_games:   pd.DataFrame,
    team_lookup:      pd.DataFrame,
    key_injuries:     dict = None,
    rest_days:        dict = None,
    lineup_strength:  dict = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame of today's picks with projected scores and edge %.
    In playoff mode, bypasses XGBoost and uses direct rolling avg calculation.
    """
    import datetime
    today = datetime.date.today()
    is_playoffs = today.month in [4, 5, 6] and today.day >= 13

    # Load models (still used for regular season)
    home_model, away_model, feat_cols = load_game_models()

    # Build features
    features_df = build_game_features(team_logs, team_ratings, upcoming_games, team_lookup, rest_days=rest_days, lineup_strength=lineup_strength)
    if features_df.empty:
        print("No game features built — no picks today.")
        return pd.DataFrame()

    if is_playoffs:
        print("  🏆 PLAYOFF MODE: Using direct rolling avg projection (bypassing XGBoost)")
        home_proj_arr = []
        away_proj_arr = []
        for _, frow in features_df.iterrows():
            hp, ap = project_scores_playoff(
                team_logs   = team_logs,
                home_id     = int(frow["home_team_id"]),
                away_id     = int(frow["away_team_id"]),
                rest_days   = rest_days,
                team_lookup = team_lookup,
            )
            home_proj_arr.append(hp if hp is not None else 112.0)
            away_proj_arr.append(ap if ap is not None else 109.0)
        home_proj = np.array(home_proj_arr)
        away_proj = np.array(away_proj_arr)
    else:
        # Align feature columns
        X = features_df[[c for c in feat_cols if c in features_df.columns]].fillna(0)
        missing = [c for c in feat_cols if c not in features_df.columns]
        for col in missing:
            X[col] = 0
        X = X[feat_cols]
        home_proj, away_proj = project_scores(X, home_model, away_model)
        # Regular season shade not needed
        PLAYOFF_TOTAL_SHADE = 0.0

    # Win probability derived directly from score projections
    diff = np.array(home_proj) - np.array(away_proj)
    win_probs = 1 / (1 + np.exp(-diff / 6.0))

    features_df = features_df.copy()
    features_df["home_proj"] = np.round(home_proj, 1)
    features_df["away_proj"] = np.round(away_proj, 1)
    features_df["total_proj"] = np.round(home_proj + away_proj, 1)
    features_df["home_win_prob"] = np.round(win_probs, 3)
    features_df["away_win_prob"] = np.round(1 - win_probs, 3)

    # ── Sanity check: flag rows where score diff and win prob magnitude conflict ──
    # If score model projects a big margin but win prob is on wrong side,
    # the feature row likely has home/away flipped — suppress ML/spread picks for it.
    def _prob_consistent(home_proj, away_proj, home_win_prob):
        score_says_home = home_proj > away_proj
        prob_says_home  = home_win_prob > 0.5
        if score_says_home != prob_says_home:
            return False
        # Also check magnitude: 10+ pt margin should give 80%+ win prob
        margin = abs(home_proj - away_proj)
        expected_prob = 1 / (1 + np.exp(-margin / 6.0))
        actual_prob = home_win_prob if score_says_home else (1 - home_win_prob)
        if margin >= 8 and actual_prob < 0.65:
            return False
        return True

    features_df["_consistent"] = features_df.apply(
        lambda r: _prob_consistent(r["home_proj"], r["away_proj"], r["home_win_prob"]), axis=1
    )

    # Fetch odds
    try:
        odds_df = fetch_nba_odds()
    except Exception as e:
        print(f"Warning: Could not fetch odds ({e}). Proceeding without edge calc.")
        odds_df = pd.DataFrame()

    # Build output rows
    rows = []
    id_to_abbr = dict(zip(team_lookup["id"], team_lookup["abbreviation"]))
    id_to_name = dict(zip(team_lookup["id"], team_lookup["full_name"]))

    for i, row in features_df.iterrows():
        home_id  = int(row["home_team_id"])
        away_id  = int(row["away_team_id"])
        home_abbr = id_to_abbr.get(home_id, str(home_id))
        away_abbr = id_to_abbr.get(away_id, str(away_id))
        home_name = id_to_name.get(home_id, home_abbr)
        away_name = id_to_name.get(away_id, away_abbr)

        pick = {
            "matchup":        f"{away_abbr} @ {home_abbr}",
            "home_team":      home_name,
            "away_team":      away_name,
            "home_proj":      row["home_proj"],
            "away_proj":      row["away_proj"],
            "total_proj":     row["total_proj"],
            "home_win_prob":  row["home_win_prob"],
            "away_win_prob":  row["away_win_prob"],
            "home_ml_edge":   np.nan,
            "away_ml_edge":   np.nan,
            "spread_edge":    np.nan,
            "total_edge":     np.nan,
            "market_total":   np.nan,
            "home_spread":    np.nan,
            "home_ml":        np.nan,
            "away_ml":        np.nan,
            "_injury_flag":   [],     # list of injury warnings for this game
        }

        # Check for key injuries affecting this game
        if key_injuries:
            for abbr in [home_abbr, away_abbr]:
                if abbr in key_injuries:
                    pick["_injury_flag"].extend(
                        [f"{p} ({abbr} OUT)" for p in key_injuries[abbr]]
                    )

        # Match odds — require BOTH home and away team names to match
        if not odds_df.empty:
            def team_last(name):
                return name.split()[-1].lower()

            home_last = team_last(home_name)
            away_last = team_last(away_name)

            match = odds_df[
                odds_df["home_team"].apply(lambda t: team_last(str(t))) == home_last
            ]
            # Also verify away team matches to prevent wrong-game attachment
            if not match.empty and not pd.isna(match.iloc[0].get("away_team")):
                match = match[
                    match["away_team"].apply(lambda t: team_last(str(t))) == away_last
                ]

            if not match.empty:
                m = match.iloc[0]
                market_spread = m.get("home_spread", np.nan)
                proj_diff = row["home_proj"] - row["away_proj"]

                pick["home_ml"]       = m.get("home_ml", np.nan)
                pick["away_ml"]       = m.get("away_ml", np.nan)
                pick["home_spread"]   = market_spread
                pick["market_total"]  = m.get("total_line", np.nan)
                pick["line_movement"] = m.get("line_movement", 0.0)
                pick["sharp_side"]    = m.get("sharp_side", "none")

                pick["home_ml_edge"] = calc_ml_edge(row["home_win_prob"], m.get("home_ml", np.nan))
                pick["away_ml_edge"] = calc_ml_edge(row["away_win_prob"], m.get("away_ml", np.nan))

                if not pd.isna(market_spread):
                    spread_edge = round(calc_spread_edge(proj_diff, market_spread), 1)
                    if abs(market_spread) <= (abs(proj_diff) * 2 + 5):
                        pick["spread_edge"] = spread_edge
                    else:
                        pick["spread_edge"] = np.nan

                if not pd.isna(m.get("total_line")):
                    pick["total_edge"] = round(calc_total_edge(row["total_proj"], m["total_line"]), 1)

        rows.append(pick)

    picks_df = pd.DataFrame(rows)

    # ── Consistency filter ─────────────────────────────────────────────────────
    def pick_side(row):
        results = []

        score_says_home = row["home_proj"] > row["away_proj"]
        prob_says_home  = row["home_win_prob"] > 0.5
        consistent      = row.get("_consistent", True)
        injured         = len(row.get("_injury_flag", [])) > 0

        # ML pick — suppress if key injury detected (model doesn't know)
        home_edge = row.get("home_ml_edge", np.nan)
        away_edge = row.get("away_ml_edge", np.nan)
        if consistent and not injured and not pd.isna(home_edge) and home_edge >= MIN_EDGE_PCT:
            if score_says_home and prob_says_home:
                results.append({"market": "ML", "pick": row["home_team"],
                                 "edge": round(home_edge, 1), "odds": row.get("home_ml")})
        if consistent and not injured and not pd.isna(away_edge) and away_edge >= MIN_EDGE_PCT:
            if not score_says_home and not prob_says_home:
                results.append({"market": "ML", "pick": row["away_team"],
                                 "edge": round(away_edge, 1), "odds": row.get("away_ml")})

        # Spread pick — suppress if key injury detected
        spread_edge = row.get("spread_edge", np.nan)
        if consistent and not injured and not pd.isna(spread_edge) and abs(spread_edge) >= MIN_EDGE_PCT:
            if spread_edge > 0 and score_says_home and prob_says_home:
                results.append({"market": "Spread", "pick": f"{row['home_team']} {row['home_spread']}",
                                 "edge": round(spread_edge, 1), "odds": -110})
            elif spread_edge < 0 and not score_says_home and not prob_says_home:
                away_spread = -row["home_spread"] if not pd.isna(row["home_spread"]) else np.nan
                results.append({"market": "Spread", "pick": f"{row['away_team']} {away_spread}",
                                 "edge": round(abs(spread_edge), 1), "odds": -110})

        # Total pick — still show even with injuries, but flag it
        total_edge = row.get("total_edge", np.nan)
        if not pd.isna(total_edge) and abs(total_edge) >= MIN_EDGE_PCT:
            direction = "Over" if total_edge > 0 else "Under"
            results.append({"market": "Total", "pick": f"{direction} {row['market_total']}",
                             "edge": round(abs(total_edge), 1), "odds": -110,
                             "injury_warning": injured})

        return results

    picks_df["picks"] = picks_df.apply(pick_side, axis=1)
    return picks_df


def print_picks(picks_df: pd.DataFrame, rest_days: dict = None):
    """Pretty-print the daily picks."""
    print("\n" + "=" * 60)
    print("🏀  NBA ALGOHUB — DAILY PICKS")
    print("=" * 60)

    has_picks = False
    for _, row in picks_df.iterrows():
        print(f"\n{row['matchup']}")

        # Rest context — look up directly from rest_days dict using team IDs
        home_id_int = int(row["home_team_id"]) if "home_team_id" in row else None
        away_id_int = int(row["away_team_id"]) if "away_team_id" in row else None

        if rest_days and home_id_int and away_id_int:
            home_rest_val = rest_days.get(home_id_int, "?")
            away_rest_val = rest_days.get(away_id_int, "?")
        else:
            home_rest_val = row.get("home_rest_days", "?")
            away_rest_val = row.get("away_rest_days", "?")

        rest_str = f"Rest: {row['home_team'].split()[-1]} {home_rest_val}d / {row['away_team'].split()[-1]} {away_rest_val}d"

        # Sharp money
        sharp    = row.get("sharp_side", "none")
        movement = row.get("line_movement", 0.0)
        if sharp != "none" and abs(movement) >= 0.5:
            sharp_team = row["home_team"].split()[-1] if sharp == "home" else row["away_team"].split()[-1]
            sharp_str = f"  📐 Sharp: {sharp_team} ({movement:+.1f})"
        else:
            sharp_str = ""

        print(f"  {rest_str}{sharp_str}")
        print(f"  Projected Score:  {row['away_team']} {row['away_proj']:.1f} — "
              f"{row['home_team']} {row['home_proj']:.1f}  (Total: {row['total_proj']:.1f})")
        print(f"  Win Probability:  {row['home_team']} {row['home_win_prob']*100:.1f}%  |  "
              f"{row['away_team']} {row['away_win_prob']*100:.1f}%")

        if row["picks"]:
            has_picks = True
            for pick in row["picks"]:
                inj_warn = "  ⚠️  INJURY WARNING" if pick.get("injury_warning") else ""
                print(f"  ✅ PICK [{pick['market']}]: {pick['pick']}  "
                      f"(Edge: +{pick['edge']}%  Odds: {pick['odds']}){inj_warn}")
        elif row.get("_injury_flag"):
            print(f"  🚨 ML/Spread suppressed — key injury: {', '.join(row['_injury_flag'])}")
        elif not row.get("_consistent", True):
            print("  ⚠️  Skipped ML/Spread — feature conflict detected")
        else:
            print("  — No picks with sufficient edge")

    if not has_picks:
        print("\n  No picks meeting minimum edge threshold today.")

    print("\n" + "=" * 60)
