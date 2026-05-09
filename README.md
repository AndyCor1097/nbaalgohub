# NBA AlgoHub

A modular NBA prediction system covering:
- **Game Score Predictions** (ML/moneyline, spread, total)
- **Player Props** (points, rebounds, assists)
- **Self-tracking & auto-retraining** via tracker.py

## Structure

```
nba_algo/
├── data.py           # NBA Stats API data collection
├── features.py       # Feature engineering (team + player)
├── train.py          # Model training (XGBoost)
├── models.py         # Model loading/saving utilities
├── picks.py          # Generate daily picks with edge %
├── props.py          # Player prop predictions
├── tracker.py        # Logging, grading, ROI dashboard
├── daily_run.py      # Master runner
└── config.py         # API keys, paths, constants
```

## Setup

```bash
pip install nba_api xgboost scikit-learn pandas numpy requests joblib
```

Add your Odds API key to `config.py`.

## Usage

```bash
python daily_run.py
```
