"""
World Cup match outcome predictor.
Features: rolling WC stats + Elo ratings from full international history.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings("ignore")

WC_DATA    = "matches.csv"
INTL_DATA  = "intl_results.csv"
ELO_PKL    = "elo.pkl"
WINDOW     = 10   # rolling WC match window


# ─── Elo helpers ──────────────────────────────────────────────────────────────

def load_elo():
    """Load pre-built Elo dict. Build it if missing."""
    try:
        with open(ELO_PKL, "rb") as f:
            data = pickle.load(f)
        return data["elo"], data["history"]
    except FileNotFoundError:
        import elo as elo_mod
        e, h, _ = elo_mod.build_elo()
        with open(ELO_PKL, "wb") as f:
            pickle.dump({"elo": e, "history": h}, f)
        return e, h


def get_elo_at(history, team, date):
    """Elo just before a given date (no leakage)."""
    from elo import START_ELO
    entries = history.get(team, [])
    if not entries:
        return START_ELO
    lo, hi, result = 0, len(entries) - 1, START_ELO
    while lo <= hi:
        mid = (lo + hi) // 2
        if entries[mid][0] < date:
            result = entries[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def get_current_elo(elo_dict, team):
    from elo import START_ELO
    return elo_dict.get(team, START_ELO)


# ─── Feature engineering ──────────────────────────────────────────────────────

FEATURE_COLS = [
    # Rolling WC form
    "home_win_rate", "home_draw_rate", "home_goals_scored", "home_goals_conceded",
    "home_gd", "home_wc_matches",
    "away_win_rate", "away_draw_rate", "away_goals_scored", "away_goals_conceded",
    "away_gd", "away_wc_matches",
    "win_rate_diff", "gd_diff", "goals_scored_diff",
    # Elo features  ← NEW
    "home_elo", "away_elo", "elo_diff",
    "home_elo_zscore", "away_elo_zscore",
    # Match context
    "stage_knockout", "year",
]


def load_and_engineer(wc_path=WC_DATA):
    elo_dict, elo_history = load_elo()

    df = pd.read_csv(wc_path)
    df = df[df["replayed"] == 0].copy()
    df["year"] = df["match_date"].str[:4].astype(int)
    df = df[df["year"] >= 1950].copy()
    df["match_date_dt"] = pd.to_datetime(df["match_date"])

    label_map = {"home team win": 0, "draw": 1, "away team win": 2}
    df["label"] = df["result"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values("match_date").reset_index(drop=True)

    # Rolling WC form tracker
    wc_history = {}

    def get_wc_stats(team):
        hist = wc_history.get(team, [])[-WINDOW:]
        n = len(hist)
        if n == 0:
            return dict(win_rate=0.33, draw_rate=0.25,
                        goals_scored=1.2, goals_conceded=1.2, gd=0.0, matches=0)
        return dict(
            win_rate       = sum(h["win"]  for h in hist) / n,
            draw_rate      = sum(h["draw"] for h in hist) / n,
            goals_scored   = sum(h["gs"]   for h in hist) / n,
            goals_conceded = sum(h["gc"]   for h in hist) / n,
            gd             = sum(h["gs"] - h["gc"] for h in hist) / n,
            matches        = n,
        )

    rows = []
    for _, row in df.iterrows():
        ht = row["home_team_name"]
        at = row["away_team_name"]
        match_date = row["match_date_dt"]

        hs  = get_wc_stats(ht)
        as_ = get_wc_stats(at)

        # Elo at match time (no leakage)
        h_elo = get_elo_at(elo_history, ht, match_date)
        a_elo = get_elo_at(elo_history, at, match_date)

        rows.append({
            "home_win_rate":       hs["win_rate"],
            "home_draw_rate":      hs["draw_rate"],
            "home_goals_scored":   hs["goals_scored"],
            "home_goals_conceded": hs["goals_conceded"],
            "home_gd":             hs["gd"],
            "home_wc_matches":     hs["matches"],
            "away_win_rate":       as_["win_rate"],
            "away_draw_rate":      as_["draw_rate"],
            "away_goals_scored":   as_["goals_scored"],
            "away_goals_conceded": as_["goals_conceded"],
            "away_gd":             as_["gd"],
            "away_wc_matches":     as_["matches"],
            "win_rate_diff":       hs["win_rate"]     - as_["win_rate"],
            "gd_diff":             hs["gd"]           - as_["gd"],
            "goals_scored_diff":   hs["goals_scored"] - as_["goals_scored"],
            "home_elo":            h_elo,
            "away_elo":            a_elo,
            "elo_diff":            h_elo - a_elo,
            "home_elo_zscore":     0.0,   # filled below
            "away_elo_zscore":     0.0,
            "stage_knockout":      int(row["knockout_stage"]),
            "year":                row["year"],
            "label":               row["label"],
            "home_team":           ht,
            "away_team":           at,
            "match_date":          row["match_date"],
        })

        # Update WC history AFTER extracting features
        for team, gs, gc, win, draw in [
            (ht, row["home_team_score"], row["away_team_score"],
             int(row["home_team_win"]), int(row["draw"])),
            (at, row["away_team_score"], row["home_team_score"],
             int(row["away_team_win"]), int(row["draw"])),
        ]:
            wc_history.setdefault(team, []).append(
                {"gs": gs, "gc": gc, "win": win, "draw": draw})

    feat_df = pd.DataFrame(rows)

    # Compute per-row Elo z-scores relative to all teams at that time
    elo_mean = feat_df[["home_elo", "away_elo"]].values.mean()
    elo_std  = feat_df[["home_elo", "away_elo"]].values.std() + 1e-9
    feat_df["home_elo_zscore"] = (feat_df["home_elo"] - elo_mean) / elo_std
    feat_df["away_elo_zscore"] = (feat_df["away_elo"] - elo_mean) / elo_std

    return feat_df, wc_history, elo_dict, elo_history


# ─── Training ─────────────────────────────────────────────────────────────────

def train(feat_df):
    X = feat_df[FEATURE_COLS].values
    y = feat_df["label"].values

    models = {
        "Gradient Boosting": Pipeline([
            ("sc", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                               max_depth=4, subsample=0.8,
                                               random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("sc", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=400, max_depth=7,
                                           min_samples_leaf=3,
                                           random_state=42, n_jobs=-1))
        ]),
        "Logistic Regression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=42))
        ]),
    }

    results = {}
    best_score, best_model, best_name = -1, None, ""
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        results[name] = {"mean": scores.mean(), "std": scores.std()}
        if scores.mean() > best_score:
            best_score, best_model, best_name = scores.mean(), pipe, name

    best_model.fit(X, y)
    return best_model, best_name, results


# ─── Prediction ───────────────────────────────────────────────────────────────

ELO_MEAN_APPROX = 1600   # rough global mean for z-score at inference
ELO_STD_APPROX  = 200


def get_wc_stats_from_history(wc_history, team):
    hist = wc_history.get(team, [])[-WINDOW:]
    n = len(hist)
    if n == 0:
        return dict(win_rate=0.33, draw_rate=0.25,
                    goals_scored=1.2, goals_conceded=1.2, gd=0.0, matches=0)
    return dict(
        win_rate       = sum(h["win"]  for h in hist) / n,
        draw_rate      = sum(h["draw"] for h in hist) / n,
        goals_scored   = sum(h["gs"]   for h in hist) / n,
        goals_conceded = sum(h["gc"]   for h in hist) / n,
        gd             = sum(h["gs"] - h["gc"] for h in hist) / n,
        matches        = n,
    )


def predict_match(model, wc_history, elo_dict, home_team, away_team,
                  stage_knockout=1, year=2026):
    hs  = get_wc_stats_from_history(wc_history, home_team)
    as_ = get_wc_stats_from_history(wc_history, away_team)

    from elo import START_ELO
    h_elo = elo_dict.get(home_team, START_ELO)
    a_elo = elo_dict.get(away_team, START_ELO)

    h_ez = (h_elo - ELO_MEAN_APPROX) / ELO_STD_APPROX
    a_ez = (a_elo - ELO_MEAN_APPROX) / ELO_STD_APPROX

    features = np.array([[
        hs["win_rate"], hs["draw_rate"], hs["goals_scored"], hs["goals_conceded"],
        hs["gd"], hs["matches"],
        as_["win_rate"], as_["draw_rate"], as_["goals_scored"], as_["goals_conceded"],
        as_["gd"], as_["matches"],
        hs["win_rate"] - as_["win_rate"],
        hs["gd"] - as_["gd"],
        hs["goals_scored"] - as_["goals_scored"],
        h_elo, a_elo, h_elo - a_elo,
        h_ez, a_ez,
        int(stage_knockout), year,
    ]])

    proba = model.predict_proba(features)[0]
    label = int(model.predict(features)[0])
    outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}

    return {
        "prediction":     outcome_map[label],
        "home_win_prob":  round(float(proba[0]) * 100, 1),
        "draw_prob":      round(float(proba[1]) * 100, 1),
        "away_win_prob":  round(float(proba[2]) * 100, 1),
        "home_elo":       round(h_elo),
        "away_elo":       round(a_elo),
        "elo_diff":       round(h_elo - a_elo),
        "home_stats":     hs,
        "away_stats":     as_,
    }


# ─── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Engineering features…")
    feat_df, wc_history, elo_dict, elo_history = load_and_engineer()
    print(f"  {len(feat_df)} matches, {len(FEATURE_COLS)} features")

    print("Training…")
    model, best_name, cv_results = train(feat_df)
    print(f"\n5-fold CV accuracy:")
    for name, r in cv_results.items():
        mark = " ◄ BEST" if name == best_name else ""
        print(f"  {name}: {r['mean']:.3f} ± {r['std']:.3f}{mark}")

    print("\nSample: Brazil vs Germany (knockout)")
    r = predict_match(model, wc_history, elo_dict, "Brazil", "Germany",
                      stage_knockout=1, year=2026)
    for k, v in r.items():
        if k not in ("home_stats", "away_stats"):
            print(f"  {k}: {v}")

    with open("model.pkl",       "wb") as f: pickle.dump(model, f)
    with open("wc_history.pkl",  "wb") as f: pickle.dump(wc_history, f)
    with open("elo_dict.pkl",    "wb") as f: pickle.dump(elo_dict, f)
    with open("cv_results.pkl",  "wb") as f:
        pickle.dump({"results": cv_results, "best": best_name}, f)
    print("\nSaved model.pkl, wc_history.pkl, elo_dict.pkl, cv_results.pkl")
