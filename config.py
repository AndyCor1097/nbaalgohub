# config.py
# NBA AlgoHub Configuration

import os

# ── API Keys ──────────────────────────────────────────────────────────────────
ODDS_API_KEY = "b9b245ea08c555d16924d88b8af17935"   # https://the-odds-api.com

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
MODELS_DIR  = "models"
LOGS_DIR    = "logs"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Season ────────────────────────────────────────────────────────────────────
CURRENT_SEASON = "2025-26"
SEASON_ID      = "22025"          # NBA Stats API format

# ── Model Parameters ──────────────────────────────────────────────────────────
ROLLING_WINDOW  = 7               # games for rolling avg features — tighter = more recent form
MIN_GAMES       = 15              # minimum games played to include player
RETRAIN_AFTER   = 50              # auto-retrain after N graded picks

# ── Odds API ──────────────────────────────────────────────────────────────────
ODDS_SPORT      = "basketball_nba"
ODDS_REGION     = "us"
ODDS_MARKETS    = ["h2h", "spreads", "totals"]
ODDS_BASE_URL   = "https://api.the-odds-api.com/v4"

# ── Edge Thresholds ───────────────────────────────────────────────────────────
MIN_EDGE_PCT    = 3.0             # minimum edge % to output a pick
PROP_MIN_EDGE   = 4.0

# ── XGBoost Defaults ──────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "n_jobs":           -1,
}
