import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings("ignore")


DATA_PATH = "matches.csv"


def load_and_engineer(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df[df["replayed"] == 0].copy()          # drop replayed
    df["year"] = df["match_date"].str[:4].astype(int)
    df = df[df["year"] >= 1950].copy()            # enough data era

    # Encode label: 0=home win, 1=draw, 2=away win
    label_map = {"home team win": 0, "draw": 1, "away team win": 2}
    df["label"] = df["result"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # ---- Build team-level rolling stats ----------------------------------------
    # We iterate chronologically and compute, for each team:
    #  - win rate (last N matches)
    #  - draw rate
    #  - goals scored avg
    #  - goals conceded avg
    #  - goal difference avg
    #  - matches played

    WINDOW = 10   # rolling window

    df = df.sort_values("match_date").reset_index(drop=True)

    team_history = {}   # team -> list of {goals_scored, goals_conceded, win, draw}

    rows = []
    for _, row in df.iterrows():
        ht = row["home_team_name"]
        at = row["away_team_name"]

        def get_stats(team):
            hist = team_history.get(team, [])
            recent = hist[-WINDOW:]
            n = len(recent)
            if n == 0:
                return dict(win_rate=0.33, draw_rate=0.25,
                            goals_scored=1.2, goals_conceded=1.2,
                            gd=0.0, matches=0)
            wins   = sum(h["win"]  for h in recent)
            draws  = sum(h["draw"] for h in recent)
            gs     = sum(h["goals_scored"]    for h in recent)
            gc     = sum(h["goals_conceded"]  for h in recent)
            return dict(
                win_rate  = wins / n,
                draw_rate = draws / n,
                goals_scored   = gs / n,
                goals_conceded = gc / n,
                gd = (gs - gc) / n,
                matches = n,
            )

        hs = get_stats(ht)
        as_ = get_stats(at)

        features = {
            "home_win_rate":          hs["win_rate"],
            "home_draw_rate":         hs["draw_rate"],
            "home_goals_scored":      hs["goals_scored"],
            "home_goals_conceded":    hs["goals_conceded"],
            "home_gd":                hs["gd"],
            "home_matches":           hs["matches"],
            "away_win_rate":          as_["win_rate"],
            "away_draw_rate":         as_["draw_rate"],
            "away_goals_scored":      as_["goals_scored"],
            "away_goals_conceded":    as_["goals_conceded"],
            "away_gd":                as_["gd"],
            "away_matches":           as_["matches"],
            # derived
            "win_rate_diff":      hs["win_rate"] - as_["win_rate"],
            "gd_diff":            hs["gd"] - as_["gd"],
            "goals_scored_diff":  hs["goals_scored"] - as_["goals_scored"],
            "stage_knockout":     int(row["knockout_stage"]),
            "year":               row["year"],
        }

        rows.append({**features, "label": row["label"],
                     "home_team": ht, "away_team": at, "match_date": row["match_date"]})

        # Update history AFTER extracting features (no leakage)
        hs_result = dict(
            goals_scored  = row["home_team_score"],
            goals_conceded= row["away_team_score"],
            win  = int(row["home_team_win"]),
            draw = int(row["draw"]),
        )
        as_result = dict(
            goals_scored  = row["away_team_score"],
            goals_conceded= row["home_team_score"],
            win  = int(row["away_team_win"]),
            draw = int(row["draw"]),
        )
        team_history.setdefault(ht, []).append(hs_result)
        team_history.setdefault(at, []).append(as_result)

    feat_df = pd.DataFrame(rows)
    return feat_df, team_history


FEATURE_COLS = [
    "home_win_rate", "home_draw_rate", "home_goals_scored", "home_goals_conceded",
    "home_gd", "home_matches",
    "away_win_rate", "away_draw_rate", "away_goals_scored", "away_goals_conceded",
    "away_gd", "away_matches",
    "win_rate_diff", "gd_diff", "goals_scored_diff",
    "stage_knockout", "year",
]


def train(feat_df):
    X = feat_df[FEATURE_COLS].values
    y = feat_df["label"].values

    models = {
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                               max_depth=4, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                           random_state=42, n_jobs=-1))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42))
        ]),
    }

    results = {}
    best_score = -1
    best_model = None
    best_name = ""

    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        results[name] = {"mean": scores.mean(), "std": scores.std()}
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = pipe
            best_name = name

    best_model.fit(X, y)
    return best_model, best_name, results


def get_team_stats_from_history(team_history, team, window=10):
    hist = team_history.get(team, [])
    recent = hist[-window:]
    n = len(recent)
    if n == 0:
        return dict(win_rate=0.33, draw_rate=0.25,
                    goals_scored=1.2, goals_conceded=1.2,
                    gd=0.0, matches=0)
    wins   = sum(h["win"]  for h in recent)
    draws  = sum(h["draw"] for h in recent)
    gs     = sum(h["goals_scored"]    for h in recent)
    gc     = sum(h["goals_conceded"]  for h in recent)
    return dict(
        win_rate  = wins / n,
        draw_rate = draws / n,
        goals_scored   = gs / n,
        goals_conceded = gc / n,
        gd = (gs - gc) / n,
        matches = n,
    )


def predict_match(model, team_history, home_team, away_team,
                  stage_knockout=1, year=2026):
    hs  = get_team_stats_from_history(team_history, home_team)
    as_ = get_team_stats_from_history(team_history, away_team)

    features = np.array([[
        hs["win_rate"], hs["draw_rate"], hs["goals_scored"], hs["goals_conceded"],
        hs["gd"], hs["matches"],
        as_["win_rate"], as_["draw_rate"], as_["goals_scored"], as_["goals_conceded"],
        as_["gd"], as_["matches"],
        hs["win_rate"] - as_["win_rate"],
        hs["gd"] - as_["gd"],
        hs["goals_scored"] - as_["goals_scored"],
        int(stage_knockout),
        year,
    ]])

    proba = model.predict_proba(features)[0]   # shape (3,)  classes: 0,1,2
    label = int(model.predict(features)[0])
    outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    return {
        "prediction": outcome_map[label],
        "home_win_prob":  round(float(proba[0]) * 100, 1),
        "draw_prob":      round(float(proba[1]) * 100, 1),
        "away_win_prob":  round(float(proba[2]) * 100, 1),
        "home_stats": hs,
        "away_stats": as_,
    }


if __name__ == "__main__":
    print("Loading & engineering features…")
    feat_df, team_history = load_and_engineer()
    print(f"  {len(feat_df)} matches ready")

    print("Training models…")
    model, best_name, cv_results = train(feat_df)
    print(f"\nModel comparison (5-fold CV accuracy):")
    for name, r in cv_results.items():
        mark = " ◄ BEST" if name == best_name else ""
        print(f"  {name}: {r['mean']:.3f} ± {r['std']:.3f}{mark}")

    print("\nSample prediction: Brazil vs Germany (knockout):")
    result = predict_match(model, team_history, "Brazil", "Germany",
                           stage_knockout=1, year=2026)
    for k, v in result.items():
        if k not in ("home_stats", "away_stats"):
            print(f"  {k}: {v}")

    # Save artifacts
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("team_history.pkl", "wb") as f:
        pickle.dump(team_history, f)
    with open("cv_results.pkl", "wb") as f:
        pickle.dump({"results": cv_results, "best": best_name}, f)
    print("\nSaved model.pkl, team_history.pkl, cv_results.pkl")
