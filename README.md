# ⚽ World Cup Match Outcome Predictor

Predict wins, draws, and losses for any World Cup matchup using historical FIFA data (1950–2022).

## Stack
- **Python 3.10+**
- **pandas** – data loading & feature engineering
- **scikit-learn** – Gradient Boosting, Random Forest, Logistic Regression
- **Streamlit** – interactive web UI

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open at http://localhost:8501

## How It Works

1. **Data** – 1,248 World Cup matches from 1930–2022 (jfjelstul/worldcup dataset)
2. **Features** – Rolling 10-match stats per team: win rate, draw rate, goals scored/conceded, goal difference
3. **Models** – Three classifiers compared via 5-fold CV; best model auto-selected
4. **Output** – Win/Draw/Loss probabilities + team form stats + H2H history

## Project Structure

```
worldcup_predictor/
├── app.py           ← Streamlit UI
├── model.py         ← Feature engineering + training + prediction
├── matches.csv      ← Historical World Cup data
├── requirements.txt
└── README.md
```

## Accuracy
~55% on 3-class classification (Home Win / Draw / Away Win).
Baseline (always predict most common class) ≈ 47%. 

## Data Source
[jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) — CC BY 4.0
