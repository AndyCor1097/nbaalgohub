# models.py
# NBA AlgoHub — Model Loading / Saving Utilities

import os
import joblib
from config import MODELS_DIR


def load_game_models() -> tuple:
    """
    Load trained game prediction models.
    Returns (home_score_model, away_score_model, feat_cols)
    Win probability is derived from score projections in picks.py.
    """
    home_score = joblib.load(os.path.join(MODELS_DIR, "nba_home_score.pkl"))
    away_score = joblib.load(os.path.join(MODELS_DIR, "nba_away_score.pkl"))
    feat_cols  = joblib.load(os.path.join(MODELS_DIR, "nba_game_feat_cols.pkl"))
    return home_score, away_score, feat_cols


def load_prop_models() -> tuple:
    """
    Load trained player prop models.
    Returns (models_dict, feat_cols)
    models_dict: {"pts": model, "reb": model, "ast": model}
    """
    models = {}
    for stat in ["pts", "reb", "ast"]:
        path = os.path.join(MODELS_DIR, f"nba_prop_{stat}.pkl")
        if os.path.exists(path):
            models[stat] = joblib.load(path)
    feat_cols = joblib.load(os.path.join(MODELS_DIR, "nba_prop_feat_cols.pkl"))
    return models, feat_cols


def models_exist() -> bool:
    """Check whether trained game models are saved."""
    required = [
        "nba_home_score.pkl",
        "nba_away_score.pkl",
        "nba_game_feat_cols.pkl",
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)
