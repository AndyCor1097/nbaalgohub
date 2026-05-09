"""
nba_daily_run.py — NBA AlgoHub Dashboard Pre-Compute
Runs prop models, fetches Bovada lines, saves data/nba_today.json.

Run every morning:
  python nba_daily_run.py
"""

import json
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_PATH = "data/nba_today.json"

STAT_MAP = {"pts": "Points", "reb": "Rebounds", "ast": "Assists"}

TEAM_ABBR = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN",
    1610612766: "CHA", 1610612741: "CHI", 1610612739: "CLE",
    1610612742: "DAL", 1610612743: "DEN", 1610612765: "DET",
    1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM",
    1610612748: "MIA", 1610612749: "MIL", 1610612750: "MIN",
    1610612740: "NOP", 1610612752: "NYK", 1610612760: "OKC",
    1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS",
    1610612761: "TOR", 1610612762: "UTA", 1610612764: "WAS",
}


def safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except:
        return default


def fetch_bovada_props() -> pd.DataFrame:
    """Pull NBA player props from Bovada public API."""
    import requests
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.bovada.lv/",
    }
    url = "https://www.bovada.lv/services/sports/event/v2/events/A/description/basketball/nba"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()

        rows = []
        for path_group in r.json():
            for event in path_group.get("events", []):
                teams = event.get("competitors", [])
                home = next((t["name"] for t in teams if t.get("home")), "")
                away = next((t["name"] for t in teams if not t.get("home")), "")

                for dg in event.get("displayGroups", []):
                    dg_desc = dg.get("description", "")
                    if "Player Points" in dg_desc:
                        prop_type = "points"
                    elif "Player Rebounds" in dg_desc:
                        prop_type = "rebounds"
                    elif "Assists" in dg_desc:
                        prop_type = "assists"
                    elif "Three" in dg_desc:
                        prop_type = "threes"
                    else:
                        continue

                    for market in dg.get("markets", []):
                        m_desc = market.get("description", "")
                        if " - " not in m_desc:
                            continue
                        player = m_desc.split(" - ", 1)[1]
                        if "(" in player:
                            player = player[:player.rfind("(")].strip()

                        line = None
                        over_odds = under_odds = -110
                        for outcome in market.get("outcomes", []):
                            o_type = outcome.get("type", "").upper()
                            price  = outcome.get("price", {})
                            handicap = price.get("handicap")
                            american = price.get("american", "-110")
                            try:
                                odds_val = int(str(american).replace("+", ""))
                            except:
                                odds_val = -110
                            if handicap is not None and line is None:
                                try:
                                    line = float(handicap)
                                except:
                                    pass
                            if o_type == "O":
                                over_odds = odds_val
                            elif o_type == "U":
                                under_odds = odds_val

                        if player and line is not None:
                            rows.append({
                                "player":     player,
                                "prop_type":  prop_type,
                                "line":       line,
                                "over_odds":  over_odds,
                                "under_odds": under_odds,
                                "home_team":  home,
                                "away_team":  away,
                            })

        df = pd.DataFrame(rows)
        print(f"  Bovada NBA props: {len(df)} lines found")
        return df

    except Exception as e:
        print(f"  Bovada props failed: {e}")
        return pd.DataFrame()


LINE_BOUNDS = {
    "points":   (8.5, 55.0),
    "rebounds": (2.5, 22.0),
    "assists":  (1.5, 14.0),
    "threes":   (0.5, 8.0),
}

def get_prop_line(props_df: pd.DataFrame, player_name: str, prop_type: str) -> dict:
    """Look up a player's over/under line using full name matching."""
    if props_df.empty:
        return {}

    name_lower = player_name.lower().strip()
    name_parts = name_lower.split()

    # Try full name match first
    match = props_df[
        props_df["player"].str.lower().str.strip().str.contains(
            " ".join(name_parts[-2:]) if len(name_parts) >= 2 else name_parts[-1]
        ) &
        (props_df["prop_type"] == prop_type)
    ]

    if match.empty:
        return {}

    # If multiple matches (common last name), pick closest full name
    if len(match) > 1:
        # Score by how many name parts match
        def name_score(bovada_name):
            bn = bovada_name.lower()
            return sum(1 for p in name_parts if p in bn)
        match = match.copy()
        match["score"] = match["player"].apply(name_score)
        match = match.sort_values("score", ascending=False)

    row = match.iloc[0]
    line = row["line"]

    # Sanity check line bounds
    bounds = LINE_BOUNDS.get(prop_type)
    if bounds and not (bounds[0] <= line <= bounds[1]):
        return {}

    return {
        "line":       line,
        "over_odds":  row["over_odds"],
        "under_odds": row["under_odds"],
    }


def main():
    print("=" * 60)
    print(f"  NBA AlgoHub Daily Run — {datetime.today().strftime('%A, %B %d %Y')}")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)

    # 1. Load NBA data
    print("\n[1/5] Loading NBA data...")
    try:
        from data import load_all_data, get_todays_games
        data = load_all_data()
        upcoming = get_todays_games()
        print(f"  {len(upcoming)} games today")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    if upcoming.empty:
        print("  No games today")
        return

    # 2. Build opponent map
    opp_map = {}
    for _, game in upcoming.iterrows():
        h = int(game["home_team_id"])
        a = int(game["visitor_team_id"])
        opp_map[h] = a
        opp_map[a] = h

    # 3. Build prop features
    print("\n[2/5] Building prop features...")
    try:
        from features import build_prop_features
        prop_features = build_prop_features(
            player_logs     = data["player_logs"],
            player_stats    = data["player_stats"],
            player_adv      = data["player_adv"],
            team_ratings    = data["team_ratings"],
            upcoming_games  = upcoming,
            player_team_map = {},
            opponent_map    = opp_map,
        )
        print(f"  {len(prop_features)} player-stat rows")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # 4. Run models
    print("\n[3/5] Running prop models...")
    try:
        from models import load_prop_models
        prop_models, feat_cols = load_prop_models()

        projections  = {}
        season_avgs  = {}
        rolling_avgs = {}
        player_teams = {}
        player_opps  = {}

        for stat, model in prop_models.items():
            df_stat = prop_features[prop_features["stat"] == stat].copy()
            if df_stat.empty:
                continue

            X = df_stat[[c for c in feat_cols if c in df_stat.columns]].fillna(0)
            for col in feat_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[feat_cols]

            preds = model.predict(X)
            df_stat["projection"] = np.round(preds, 1)

            for _, row in df_stat.iterrows():
                pname = row["player_name"]
                tid   = int(row["team_id"]) if not pd.isna(row["team_id"]) else 0
                oid   = int(row["opp_team_id"]) if not pd.isna(row.get("opp_team_id", None)) else 0

                if pname not in projections:
                    projections[pname]  = {}
                    season_avgs[pname]  = {}
                    rolling_avgs[pname] = {}
                    player_teams[pname] = tid
                    player_opps[pname]  = oid

                projections[pname][stat]  = safe_float(row["projection"])
                season_avgs[pname][stat]  = safe_float(row["season_avg"])
                rolling_avgs[pname][stat] = safe_float(row.get(f"rolling_{stat}", row["season_avg"]))

        print(f"  {len(projections)} players projected")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # 5. Fetch Bovada props
    print("\n[4/5] Fetching Bovada prop lines...")
    bovada = fetch_bovada_props()

    prop_type_map = {"pts": "points", "reb": "rebounds", "ast": "assists"}

    # 6. Build output
    print("\n[5/5] Building output JSON...")

    games_output = []
    for _, game in upcoming.iterrows():
        home_id   = int(game["home_team_id"])
        away_id   = int(game["visitor_team_id"])
        home_abbr = TEAM_ABBR.get(home_id, "HOME")
        away_abbr = TEAM_ABBR.get(away_id, "AWAY")

        game_players = []
        for pname, projs in projections.items():
            tid = player_teams.get(pname, 0)
            if tid not in [home_id, away_id]:
                continue

            team_abbr = TEAM_ABBR.get(tid, "")
            opp_abbr  = TEAM_ABBR.get(opp_map.get(tid, 0), "OPP")

            player_props = []
            for stat in ["pts", "reb", "ast"]:
                proj = projs.get(stat)
                if proj is None:
                    continue

                s_avg = season_avgs.get(pname, {}).get(stat, 0)
                r_avg = rolling_avgs.get(pname, {}).get(stat, 0)

                line_data  = get_prop_line(bovada, pname, prop_type_map[stat])
                line       = line_data.get("line")
                over_odds  = line_data.get("over_odds", -110)
                under_odds = line_data.get("under_odds", -110)

                overlay   = round(proj - line, 1) if line is not None else None
                direction = None
                edge_pct  = None
                if overlay is not None:
                    direction = "Over" if overlay > 0 else "Under"
                    edge_pct  = round(abs(overlay) / max(line, 1) * 100, 1)

                form = (
                    "hot"     if r_avg > s_avg * 1.15 else
                    "cold"    if r_avg < s_avg * 0.85 else
                    "neutral"
                )

                player_props.append({
                    "stat":        STAT_MAP[stat],
                    "stat_key":    stat,
                    "projection":  proj,
                    "season_avg":  round(s_avg, 1),
                    "rolling_avg": round(r_avg, 1),
                    "line":        line,
                    "overlay":     overlay,
                    "direction":   direction,
                    "edge_pct":    edge_pct,
                    "over_odds":   over_odds,
                    "under_odds":  under_odds,
                    "form":        form,
                })

            if not player_props:
                continue

            overlays = [abs(p["overlay"]) for p in player_props if p["overlay"] is not None]
            avg_overlay = round(sum(overlays) / len(overlays), 2) if overlays else 0

            game_players.append({
                "player_name": pname,
                "team":        team_abbr,
                "opponent":    opp_abbr,
                "props":       player_props,
                "avg_overlay": avg_overlay,
            })

        game_players.sort(key=lambda x: x["avg_overlay"], reverse=True)

        games_output.append({
            "home_team":    home_abbr,
            "away_team":    away_abbr,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "players":      game_players,
        })

    # Top overlays across all games
    top_overlays = []
    for g in games_output:
        for p in g["players"]:
            for prop in p["props"]:
                if prop["overlay"] is not None and abs(prop["overlay"]) >= 1.5:
                    top_overlays.append({
                        "player_name": p["player_name"],
                        "team":        p["team"],
                        "opponent":    p["opponent"],
                        "game":        f"{g['away_team']} @ {g['home_team']}",
                        "stat":        prop["stat"],
                        "projection":  prop["projection"],
                        "season_avg":  prop["season_avg"],
                        "rolling_avg": prop["rolling_avg"],
                        "line":        prop["line"],
                        "overlay":     prop["overlay"],
                        "direction":   prop["direction"],
                        "edge_pct":    prop["edge_pct"],
                        "over_odds":   prop["over_odds"],
                        "form":        prop["form"],
                    })

    top_overlays.sort(key=lambda x: abs(x["overlay"]) if x["overlay"] else 0, reverse=True)

    output = {
        "date":         datetime.today().strftime("%Y-%m-%d"),
        "generated":    datetime.now().isoformat(),
        "games":        games_output,
        "top_overlays": top_overlays[:20],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, default=str)

    print(f"\n  Saved {len(games_output)} games, {len(top_overlays)} overlays to {OUTPUT_PATH}")

    # Git push
    print("\n[6/6] Pushing to GitHub...")
    try:
        subprocess.run(["git", "add", OUTPUT_PATH], check=True)
        subprocess.run(["git", "commit", "-m", f"NBA data {datetime.today().strftime('%Y-%m-%d')}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("  Pushed ✓")
    except Exception as e:
        print(f"  Git push failed: {e}")

    print(f"\n✓ Done! Share: https://nbaalgohub.streamlit.app")


if __name__ == "__main__":
    main()
