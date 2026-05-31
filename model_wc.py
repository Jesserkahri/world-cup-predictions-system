"""
WC-only prediction model — no Elo, no external data.
Uses rolling World Cup match stats only.
"""
import pandas as pd, numpy as np, pickle, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

WC_DATA = "matches.csv"
WINDOW  = 10

FEATURE_COLS = [
    "home_win_rate","home_draw_rate","home_goals_scored","home_goals_conceded",
    "home_gd","home_wc_matches",
    "away_win_rate","away_draw_rate","away_goals_scored","away_goals_conceded",
    "away_gd","away_wc_matches",
    "win_rate_diff","gd_diff","goals_scored_diff",
    "stage_knockout","year",
]

def _get_stats(hist, team):
    recent = hist.get(team, [])[-WINDOW:]
    n = len(recent)
    if n == 0:
        return dict(win_rate=0.33,draw_rate=0.25,goals_scored=1.2,
                    goals_conceded=1.2,gd=0.0,matches=0)
    return dict(
        win_rate       = sum(h["win"]  for h in recent)/n,
        draw_rate      = sum(h["draw"] for h in recent)/n,
        goals_scored   = sum(h["gs"]   for h in recent)/n,
        goals_conceded = sum(h["gc"]   for h in recent)/n,
        gd             = sum(h["gs"]-h["gc"] for h in recent)/n,
        matches        = n,
    )

def load_and_engineer(path=WC_DATA):
    df = pd.read_csv(path)
    df = df[df["replayed"]==0].copy()
    df["year"] = df["match_date"].str[:4].astype(int)
    df = df[df["year"]>=1950].copy()
    label_map = {"home team win":0,"draw":1,"away team win":2}
    df["label"] = df["result"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values("match_date").reset_index(drop=True)

    hist = {}
    rows = []
    for _, row in df.iterrows():
        ht, at = row["home_team_name"], row["away_team_name"]
        hs = _get_stats(hist, ht); as_ = _get_stats(hist, at)
        rows.append({
            "home_win_rate":hs["win_rate"],"home_draw_rate":hs["draw_rate"],
            "home_goals_scored":hs["goals_scored"],"home_goals_conceded":hs["goals_conceded"],
            "home_gd":hs["gd"],"home_wc_matches":hs["matches"],
            "away_win_rate":as_["win_rate"],"away_draw_rate":as_["draw_rate"],
            "away_goals_scored":as_["goals_scored"],"away_goals_conceded":as_["goals_conceded"],
            "away_gd":as_["gd"],"away_wc_matches":as_["matches"],
            "win_rate_diff":hs["win_rate"]-as_["win_rate"],
            "gd_diff":hs["gd"]-as_["gd"],
            "goals_scored_diff":hs["goals_scored"]-as_["goals_scored"],
            "stage_knockout":int(row["knockout_stage"]),"year":row["year"],
            "label":row["label"],
        })
        for team,gs,gc,win,draw in [
            (ht,row["home_team_score"],row["away_team_score"],int(row["home_team_win"]),int(row["draw"])),
            (at,row["away_team_score"],row["home_team_score"],int(row["away_team_win"]),int(row["draw"])),
        ]:
            hist.setdefault(team,[]).append({"gs":gs,"gc":gc,"win":win,"draw":draw})

    return pd.DataFrame(rows), hist, FEATURE_COLS

def train(feat_df, feature_cols=FEATURE_COLS):
    X = feat_df[feature_cols].values
    y = feat_df["label"].values
    pipes = {
        "Random Forest": Pipeline([("sc",StandardScaler()),
            ("clf",RandomForestClassifier(n_estimators=400,max_depth=7,random_state=42,n_jobs=-1))]),
        "Logistic Regression": Pipeline([("sc",StandardScaler()),
            ("clf",LogisticRegression(C=0.5,max_iter=1000,random_state=42))]),
    }
    results={}; best_score=-1; best_model=None; best_name=""
    for name,pipe in pipes.items():
        scores = cross_val_score(pipe,X,y,cv=5,scoring="accuracy")
        results[name]={"mean":scores.mean(),"std":scores.std()}
        if scores.mean()>best_score: best_score,best_model,best_name=scores.mean(),pipe,name
    best_model.fit(X,y)
    return best_model, best_name, results

def predict_match(model, wc_history, home_team, away_team,
                  stage_knockout=1, year=2026):
    hs  = _get_stats(wc_history, home_team)
    as_ = _get_stats(wc_history, away_team)
    feats = np.array([[
        hs["win_rate"],hs["draw_rate"],hs["goals_scored"],hs["goals_conceded"],
        hs["gd"],hs["matches"],
        as_["win_rate"],as_["draw_rate"],as_["goals_scored"],as_["goals_conceded"],
        as_["gd"],as_["matches"],
        hs["win_rate"]-as_["win_rate"], hs["gd"]-as_["gd"],
        hs["goals_scored"]-as_["goals_scored"],
        int(stage_knockout), year,
    ]])
    proba = model.predict_proba(feats)[0]
    label = int(model.predict(feats)[0])
    return {
        "prediction":    {0:"Home Win",1:"Draw",2:"Away Win"}[label],
        "home_win_prob": round(float(proba[0])*100,1),
        "draw_prob":     round(float(proba[1])*100,1),
        "away_win_prob": round(float(proba[2])*100,1),
        "home_stats":    hs,
        "away_stats":    as_,
    }
