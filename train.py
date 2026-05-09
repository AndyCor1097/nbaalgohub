# train.py
# NBA AlgoHub — Model Training
#
# Trains three models:
#   1. home_score_model  — predict home team points (regression)
#   2. away_score_model  — predict away team points (regression)
#   3. home_win_model    — predict home win probability (classification)
#
# And per-stat prop models:
#   4. prop_pts_model    — predict player points
#   5. prop_reb_model    — predict player rebounds
#   6. prop_ast_model    — predict player assists

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR, XGB_PARAMS
from features import build_training_game_features, build_prop_features


# ── Feature columns ────────────────────────────────────────────────────────────

GAME_FEAT_COLS = [
    "home_pts", "away_pts",
    "home_reb", "away_reb",
    "home_ast", "away_ast",
    "home_tov", "away_tov",
    "home_plus_minus", "away_plus_minus",
    "diff_pts", "diff_reb", "diff_ast", "diff_tov", "diff_plus_minus",
    "home_off_rating", "home_def_rating", "home_net_rating", "home_pace",
    "away_off_rating", "away_def_rating", "away_net_rating", "away_pace",
    "diff_off_rating", "diff_def_rating", "diff_net_rating", "diff_pace",
    "avg_pace", "home_advantage",
    # Rest / schedule context
    "home_rest_days", "away_rest_days", "rest_advantage",
    "home_b2b", "away_b2b",
    # Market line features
    "market_home_implied", "market_spread", "market_total",
    # Lineup strength
    "home_lineup_pts", "away_lineup_pts", "lineup_pts_diff",
    # Head-to-head
    "h2h_win_rate", "h2h_avg_diff", "h2h_avg_total",
    # Series momentum
    "home_momentum",
]

PROP_FEAT_COLS = [
    "season_avg", "season_min", "season_usg", "season_fga",
    "season_fg_pct", "season_fta", "gp",
    "rolling_pts", "rolling_reb", "rolling_ast", "rolling_min",
    "opp_def_rating", "pts_per_min", "reb_per_min", "ast_per_min",
]


def _safe_cols(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]


# ── Game models ────────────────────────────────────────────────────────────────

def train_game_models(game_features: pd.DataFrame, verbose: bool = True):
    """
    Train home score and away score models.
    Win probability is derived from score projections at inference time.
    Saves models to MODELS_DIR.
    """
    df = game_features.dropna(subset=["home_pts", "away_pts", "total_pts"])

    feat_cols = _safe_cols(df, GAME_FEAT_COLS)
    # Remove target-leaking columns
    feat_cols = [c for c in feat_cols if c not in ("home_pts", "away_pts")]

    X = df[feat_cols].fillna(df[feat_cols].median())

    models = {}
    results = {}

    tscv = TimeSeriesSplit(n_splits=5)

    # ── Home score model ──
    y_home = df["home_pts"]
    reg_home = XGBRegressor(**XGB_PARAMS)
    cv_mae = -cross_val_score(reg_home, X, y_home, cv=tscv, scoring="neg_mean_absolute_error")
    reg_home.fit(X, y_home)
    models["home_score"] = reg_home
    results["home_score_mae"] = cv_mae.mean()
    if verbose:
        print(f"Home Score Model  — CV MAE: {cv_mae.mean():.2f} pts")

    # ── Away score model ──
    y_away = df["away_pts"]
    reg_away = XGBRegressor(**XGB_PARAMS)
    cv_mae_a = -cross_val_score(reg_away, X, y_away, cv=tscv, scoring="neg_mean_absolute_error")
    reg_away.fit(X, y_away)
    models["away_score"] = reg_away
    results["away_score_mae"] = cv_mae_a.mean()
    if verbose:
        print(f"Away Score Model  — CV MAE: {cv_mae_a.mean():.2f} pts")

    # Save models + feature columns
    for name, model in models.items():
        path = os.path.join(MODELS_DIR, f"nba_{name}.pkl")
        joblib.dump(model, path)
    joblib.dump(feat_cols, os.path.join(MODELS_DIR, "nba_game_feat_cols.pkl"))

    if verbose:
        print(f"\nModels saved to {MODELS_DIR}/")

    return models, feat_cols, results


# ── Prop models ────────────────────────────────────────────────────────────────

def train_prop_models(prop_features: pd.DataFrame, verbose: bool = True):
    """
    Train one regression model per stat (pts, reb, ast).
    Target: rolling_<stat> (recent form) rather than season avg —
    this makes the model respond to hot/cold streaks and matchup difficulty.
    Sample weights boost recent games and tough matchups.
    """
    models = {}
    results = {}

    # XGBoost params tuned for props — shallower trees to avoid overfitting
    prop_xgb_params = {
        "n_estimators":     400,
        "max_depth":        4,
        "learning_rate":    0.03,
        "subsample":        0.75,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "random_state":     42,
        "n_jobs":           -1,
    }

    for stat in ["pts", "reb", "ast"]:
        df_stat = prop_features[prop_features["stat"] == stat].copy()

        # Use rolling stat as target — reflects recent form, not just season avg
        rolling_col = f"rolling_{stat}"
        if rolling_col not in df_stat.columns:
            print(f"  No rolling_{stat} column — falling back to season_avg for {stat}")
            target_col = "season_avg"
        else:
            # Drop rows where rolling stat is missing (early season / not enough games)
            df_stat = df_stat.dropna(subset=[rolling_col])
            target_col = rolling_col

        if len(df_stat) < 50:
            print(f"Not enough data for {stat} prop model (n={len(df_stat)}), skipping.")
            continue

        feat_cols = _safe_cols(df_stat, PROP_FEAT_COLS)
        X = df_stat[feat_cols].fillna(df_stat[feat_cols].median())
        y = df_stat[target_col]

        # Sample weights: upweight players facing tough defenses (high opp_def_rating)
        # and players with high usage — these cases are most predictable
        if "opp_def_rating" in df_stat.columns and "season_usg" in df_stat.columns:
            w_matchup = df_stat["opp_def_rating"].fillna(110) / 110
            w_usage   = df_stat["season_usg"].fillna(0.2) / 0.2
            sample_weights = (w_matchup * w_usage).clip(0.5, 2.0).values
        else:
            sample_weights = None

        tscv = TimeSeriesSplit(n_splits=5)
        reg = XGBRegressor(**prop_xgb_params)

        cv_mae = -cross_val_score(reg, X, y, cv=tscv, scoring="neg_mean_absolute_error")

        if sample_weights is not None:
            reg.fit(X, y, sample_weight=sample_weights)
        else:
            reg.fit(X, y)

        models[stat] = reg
        results[f"prop_{stat}_mae"] = cv_mae.mean()
        if verbose:
            print(f"Prop {stat.upper()} Model  — CV MAE: {cv_mae.mean():.2f}  "
                  f"(target: {target_col})")

        path = os.path.join(MODELS_DIR, f"nba_prop_{stat}.pkl")
        joblib.dump(reg, path)

    joblib.dump(
        _safe_cols(prop_features[prop_features["stat"] == "pts"], PROP_FEAT_COLS),
        os.path.join(MODELS_DIR, "nba_prop_feat_cols.pkl"),
    )

    if verbose:
        print(f"Prop models saved to {MODELS_DIR}/")

    return models, results


# ── Master trainer ─────────────────────────────────────────────────────────────

def run_training(data: dict, verbose: bool = True):
    """
    Build features and train all models from loaded data dict.
    """
    print("\n=== Building game training features ===")
    game_feats = build_training_game_features(data["team_logs"], data["team_ratings"])
    if game_feats.empty:
        print("ERROR: No game training features built. Check team_logs matchup column.")
        game_models, game_feat_cols, game_results = {}, [], {}
    else:
        print(f"  {len(game_feats)} game samples")
        print("\n=== Training game models ===")
        game_models, game_feat_cols, game_results = train_game_models(game_feats, verbose=verbose)

    print("\n=== Building prop training features ===")
    # For training, opponent_map is a simplification — maps each team to a dummy opponent
    all_team_ids = data["player_stats"]["team_id"].dropna().unique().tolist()
    dummy_opp_map = {int(t): int(all_team_ids[(i + 1) % len(all_team_ids)])
                     for i, t in enumerate(all_team_ids)}

    prop_feats = build_prop_features(
        player_logs    = data["player_logs"],
        player_stats   = data["player_stats"],
        player_adv     = data["player_adv"],
        team_ratings   = data["team_ratings"],
        upcoming_games = pd.DataFrame(),   # unused for training
        player_team_map = {},
        opponent_map    = dummy_opp_map,
    )

    if prop_feats.empty:
        print("ERROR: No prop features built.")
        prop_models, prop_results = {}, {}
    else:
        print(f"  {len(prop_feats)} player-stat samples")
        print("\n=== Training prop models ===")
        prop_models, prop_results = train_prop_models(prop_feats, verbose=verbose)

    all_results = {**game_results, **prop_results}
    print("\n=== Training complete ===")
    for k, v in all_results.items():
        print(f"  {k}: {v:.3f}")

    return game_models, prop_models, all_results
