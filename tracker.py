# tracker.py
# NBA AlgoHub — Pick Logging, Result Grading, ROI Dashboard
#
# Logs picks → grades results via NBA Stats API → tracks ROI

import os
import json
import datetime
import pandas as pd
import numpy as np
import requests

from config import LOGS_DIR, RETRAIN_AFTER, CURRENT_SEASON

PICKS_LOG   = os.path.join(LOGS_DIR, "nba_picks_log.jsonl")
RESULTS_LOG = os.path.join(LOGS_DIR, "nba_results.csv")


# ── Logging ────────────────────────────────────────────────────────────────────

def log_picks(picks_df: pd.DataFrame, props_df: pd.DataFrame = None, date: str = None):
    """
    Append today's picks to the picks log (JSONL format).
    """
    if date is None:
        date = datetime.date.today().isoformat()

    entries = []

    # Game picks
    for _, row in picks_df.iterrows():
        for pick in row.get("picks", []):
            entries.append({
                "date":        date,
                "type":        "game",
                "matchup":     row["matchup"],
                "market":      pick["market"],
                "pick":        pick["pick"],
                "edge":        pick["edge"],
                "odds":        pick.get("odds", -110),
                "home_proj":   row["home_proj"],
                "away_proj":   row["away_proj"],
                "total_proj":  row["total_proj"],
                "result":      None,   # filled in later
                "graded":      False,
            })

    # Prop picks
    if props_df is not None and not props_df.empty:
        from props import filter_top_props
        top_props = filter_top_props(props_df)
        for _, row in top_props.iterrows():
            entries.append({
                "date":       date,
                "type":       "prop",
                "player":     row["player"],
                "stat":       row["stat"],
                "direction":  row["direction"],
                "line":       row.get("line"),
                "projection": row["projection"],
                "edge":       row["edge"],
                "odds":       row.get("over_odds" if row["direction"] == "Over" else "under_odds", -110),
                "result":     None,
                "graded":     False,
            })

    with open(PICKS_LOG, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Logged {len(entries)} picks to {PICKS_LOG}")


# ── Grading ────────────────────────────────────────────────────────────────────

def fetch_game_scores(game_date: str) -> dict:
    """
    Fetch final scores for a given date using NBA Stats API.
    Returns dict: {game_id: {"home_pts": X, "away_pts": Y, "home_team": ..., "away_team": ...}}
    """
    from nba_api.stats.endpoints import scoreboardv2
    import time

    sb = scoreboardv2.ScoreboardV2(game_date=game_date)
    time.sleep(0.6)

    line_score = sb.get_data_frames()[1]   # LineScore
    line_score.columns = [c.lower() for c in line_score.columns]

    games = {}
    for game_id, group in line_score.groupby("game_id"):
        if len(group) < 2:
            continue
        home_row = group[group["team_abbreviation"].notna()].iloc[0]
        away_row = group[group["team_abbreviation"].notna()].iloc[1]
        games[game_id] = {
            "home_team": home_row["team_abbreviation"],
            "away_team": away_row["team_abbreviation"],
            "home_pts":  home_row.get("pts", None),
            "away_pts":  away_row.get("pts", None),
        }
    return games


def grade_pick(pick: dict, game_scores: dict) -> str:
    """
    Grade a single pick as 'W', 'L', or 'P' (push).
    Returns result string.
    """
    if pick["type"] != "game":
        return "UNGRADED"   # prop grading requires box score lookup

    market = pick.get("market", "")
    pick_str = pick.get("pick", "")

    # Find matching game
    matchup = pick.get("matchup", "")
    parts = matchup.split(" @ ")
    if len(parts) != 2:
        return "UNGRADED"
    away_abbr, home_abbr = parts[0].strip(), parts[1].strip()

    matched_game = None
    for gid, scores in game_scores.items():
        if (scores["home_team"] == home_abbr and scores["away_team"] == away_abbr):
            matched_game = scores
            break

    if matched_game is None:
        return "UNGRADED"

    home_pts = matched_game["home_pts"]
    away_pts = matched_game["away_pts"]
    if home_pts is None or away_pts is None:
        return "UNGRADED"

    actual_total = home_pts + away_pts
    home_won = home_pts > away_pts

    if market == "ML":
        if home_abbr in pick_str:
            return "W" if home_won else "L"
        else:
            return "W" if not home_won else "L"

    elif market == "Spread":
        # parse spread line from pick string
        import re
        nums = re.findall(r"[-+]?\d+\.?\d*", pick_str.replace(home_abbr, "").replace(away_abbr, ""))
        if not nums:
            return "UNGRADED"
        spread_line = float(nums[-1])
        if home_abbr in pick_str:
            cover = (home_pts + spread_line) > away_pts
            push  = (home_pts + spread_line) == away_pts
        else:
            cover = (away_pts + spread_line) > home_pts
            push  = (away_pts + spread_line) == home_pts
        if push:
            return "P"
        return "W" if cover else "L"

    elif market == "Total":
        if "Over" in pick_str:
            total_line = float(pick_str.replace("Over", "").strip())
            return "W" if actual_total > total_line else ("P" if actual_total == total_line else "L")
        elif "Under" in pick_str:
            total_line = float(pick_str.replace("Under", "").strip())
            return "W" if actual_total < total_line else ("P" if actual_total == total_line else "L")

    return "UNGRADED"


def grade_picks_for_date(grade_date: str = None):
    """
    Grade all ungraded picks for a given date.
    Updates the picks log in place.
    """
    if grade_date is None:
        grade_date = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    print(f"Grading picks for {grade_date}...")
    game_scores = fetch_game_scores(grade_date)

    # Load picks log
    picks = []
    with open(PICKS_LOG, "r") as f:
        for line in f:
            picks.append(json.loads(line.strip()))

    updated = 0
    for pick in picks:
        if pick.get("graded") or pick.get("date") != grade_date:
            continue
        result = grade_pick(pick, game_scores)
        pick["result"] = result
        pick["graded"] = result != "UNGRADED"
        if pick["graded"]:
            updated += 1

    # Rewrite log
    with open(PICKS_LOG, "w") as f:
        for pick in picks:
            f.write(json.dumps(pick) + "\n")

    print(f"  Graded {updated} picks.")
    return picks


# ── ROI Dashboard ──────────────────────────────────────────────────────────────

def calc_roi(odds: float, result: str) -> float:
    """Return profit/loss per $1 unit wagered."""
    if result == "P":
        return 0.0
    if result == "W":
        if odds > 0:
            return odds / 100
        else:
            return 100 / abs(odds)
    return -1.0


def roi_dashboard() -> dict:
    """
    Compute hit rate, ROI, and per-market breakdown from graded picks.
    """
    if not os.path.exists(PICKS_LOG):
        return {"error": "No picks log found."}

    picks = []
    with open(PICKS_LOG, "r") as f:
        for line in f:
            picks.append(json.loads(line.strip()))

    graded = [p for p in picks if p.get("graded") and p.get("result") in ("W", "L", "P")]
    if not graded:
        return {"graded": 0, "message": "No graded picks yet."}

    df = pd.DataFrame(graded)
    df["profit"] = df.apply(lambda r: calc_roi(r.get("odds", -110), r["result"]), axis=1)

    total    = len(df)
    wins     = (df["result"] == "W").sum()
    losses   = (df["result"] == "L").sum()
    pushes   = (df["result"] == "P").sum()
    roi      = df["profit"].sum() / (total - pushes) * 100 if (total - pushes) > 0 else 0
    hit_rate = wins / (total - pushes) * 100 if (total - pushes) > 0 else 0

    by_market = {}
    for market, group in df.groupby("market") if "market" in df.columns else []:
        w = (group["result"] == "W").sum()
        n = len(group[group["result"] != "P"])
        by_market[market] = {
            "record":   f"{w}-{(group['result']=='L').sum()}",
            "hit_rate": round(w / n * 100, 1) if n > 0 else 0,
            "roi":      round(group["profit"].sum() / n * 100, 1) if n > 0 else 0,
        }

    result = {
        "total_picks":  total,
        "record":       f"{wins}-{losses}-{pushes}",
        "hit_rate":     round(hit_rate, 1),
        "roi_pct":      round(roi, 2),
        "total_profit": round(df["profit"].sum(), 2),
        "by_market":    by_market,
        "needs_retrain": total >= RETRAIN_AFTER,
    }
    return result


def print_dashboard():
    """Print the ROI dashboard to console."""
    stats = roi_dashboard()
    if "error" in stats or "message" in stats:
        print(stats.get("error") or stats.get("message"))
        return

    print("\n" + "=" * 55)
    print("🏀  NBA ALGOHUB — PERFORMANCE DASHBOARD")
    print("=" * 55)
    print(f"  Record:      {stats['record']}")
    print(f"  Hit Rate:    {stats['hit_rate']}%")
    print(f"  ROI:         {stats['roi_pct']}%")
    print(f"  Total Profit (units): {stats['total_profit']}")
    print()
    for market, m in stats.get("by_market", {}).items():
        print(f"  [{market}]  {m['record']}  |  Hit {m['hit_rate']}%  |  ROI {m['roi']}%")
    if stats.get("needs_retrain"):
        print(f"\n  ⚠️  {RETRAIN_AFTER}+ graded picks — consider retraining models.")
    print("=" * 55)
