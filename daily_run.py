# daily_run.py
# NBA AlgoHub — Master Daily Runner
#
# Usage:
#   python daily_run.py              → run picks for today
#   python daily_run.py --train      → (re)train models first
#   python daily_run.py --grade      → grade yesterday's picks
#   python daily_run.py --dashboard  → show ROI dashboard
#   python daily_run.py --all        → grade + run picks + dashboard

import argparse
import datetime
import pandas as pd

from data import load_all_data, get_all_teams, get_todays_games
from train import run_training
from picks import generate_game_picks, print_picks
from props import generate_prop_picks, print_props
from tracker import log_picks, grade_picks_for_date, print_dashboard, roi_dashboard
from models import models_exist
from config import CURRENT_SEASON, RETRAIN_AFTER


def run_picks():
    """Load data, generate picks, log them."""
    print("\n📥  Loading NBA data...")
    data = load_all_data()

    print("\n📅  Fetching today's schedule...")
    try:
        upcoming = get_todays_games()
    except Exception as e:
        print(f"Could not fetch today's schedule: {e}")
        upcoming = pd.DataFrame()

    if upcoming.empty:
        print("No games today.")
        return

    teams = get_all_teams()

    # ── Game picks ──
    print("\n🔮  Generating game picks...")
    from data import get_rest_days
    rest_days = get_rest_days(data["team_logs"], upcoming)
    if rest_days:
        print(f"  Rest days: { {k: v for k, v in list(rest_days.items())[:6]} }...")

    picks_df = generate_game_picks(
        team_logs        = data["team_logs"],
        team_ratings     = data["team_ratings"],
        upcoming_games   = upcoming,
        team_lookup      = teams,
        key_injuries     = data.get("key_injuries", {}),
        rest_days        = rest_days,
        lineup_strength  = data.get("lineup_strength", {}),
    )
    print_picks(picks_df, rest_days=rest_days)

    # Save today's lines as closing lines for future training
    try:
        from features import fetch_nba_odds
        from data import save_closing_lines
        odds_df = fetch_nba_odds()
        if not odds_df.empty:
            save_closing_lines(odds_df)
            print("  📁 Closing lines saved.")
    except Exception as e:
        pass

    # ── Prop picks ──
    print("\n🎯  Generating player prop picks...")
    props_df = generate_prop_picks(
        player_logs    = data["player_logs"],
        player_stats   = data["player_stats"],
        player_adv     = data["player_adv"],
        team_ratings   = data["team_ratings"],
        upcoming_games = upcoming,
    )
    print_props(props_df)

    # ── Log picks ──
    log_picks(picks_df, props_df)


def run_training_pipeline():
    """Load multi-season data and train all models."""
    print("\n📥  Loading NBA data for training (4 seasons)...")
    data = load_all_data(multi_season=True)
    print("\n🏋️  Training models...")
    run_training(data, verbose=True)


def maybe_auto_retrain():
    """Auto-retrain if >= RETRAIN_AFTER graded picks accumulated."""
    stats = roi_dashboard()
    if stats.get("needs_retrain"):
        print(f"\n⚠️  Auto-retraining triggered ({RETRAIN_AFTER}+ graded picks)...")
        run_training_pipeline()


def main():
    parser = argparse.ArgumentParser(description="NBA AlgoHub Daily Runner")
    parser.add_argument("--train",     action="store_true", help="Train/retrain models")
    parser.add_argument("--grade",     action="store_true", help="Grade yesterday's picks")
    parser.add_argument("--dashboard", action="store_true", help="Show ROI dashboard")
    parser.add_argument("--all",       action="store_true", help="Grade + picks + dashboard")
    parser.add_argument("--date",      type=str,            help="Grade date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.train:
        run_training_pipeline()
        return

    if args.grade or args.all:
        grade_date = args.date or (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        grade_picks_for_date(grade_date)

    if args.dashboard:
        print_dashboard()
        return

    # Check if models exist; if not, train first
    if not models_exist():
        print("\n⚠️  No trained models found. Running initial training...")
        run_training_pipeline()

    # Auto-retrain check
    maybe_auto_retrain()

    # Run picks
    run_picks()

    if args.all:
        print_dashboard()


if __name__ == "__main__":
    main()
