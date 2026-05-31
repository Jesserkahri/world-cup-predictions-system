"""
Elo rating system for international football.
Uses the entire martj42 international results dataset (1872–present).

Key design choices:
- K-factor varies by tournament importance (WC=60, Continental=50, Qual=40, Friendly=20)
- Home advantage: +100 Elo points for non-neutral venues
- Starting Elo: 1500 for all teams
- Elo is computed chronologically with NO leakage
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

DATA_PATH = "intl_results.csv"

TOURNAMENT_K = {
    "FIFA World Cup": 60,
    "Copa América": 50,
    "UEFA Euro": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "CONCACAF Gold Cup": 50,
    "FIFA Confederations Cup": 45,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 40,
    "African Cup of Nations qualification": 35,
    "AFC Asian Cup qualification": 35,
    "UEFA Nations League": 40,
    "Friendly": 20,
}
DEFAULT_K = 30
HOME_ADV  = 100
START_ELO = 1500


def get_k(tournament):
    for key, k in TOURNAMENT_K.items():
        if key.lower() in tournament.lower():
            return k
    return DEFAULT_K


def expected(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def build_elo(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)

    elo = {}          # team → current Elo
    history = {}      # team → list of (date, elo_before_match)

    def get_elo(team):
        return elo.get(team, START_ELO)

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        neutral = bool(row["neutral"])
        tournament = str(row["tournament"])
        hs, as_ = row["home_score"], row["away_score"]

        ra = get_elo(home)
        rb = get_elo(away)

        # Record elo BEFORE this match (for feature extraction without leakage)
        history.setdefault(home, []).append((row["date"], ra))
        history.setdefault(away, []).append((row["date"], rb))

        # Home advantage (not applied on neutral ground)
        ra_adj = ra + (0 if neutral else HOME_ADV)

        ea = expected(ra_adj, rb)
        eb = 1 - ea

        if hs > as_:
            sa, sb = 1.0, 0.0
        elif hs < as_:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        k = get_k(tournament)
        elo[home] = ra + k * (sa - ea)
        elo[away] = rb + k * (sb - eb)

    return elo, history, df


def get_elo_at_date(history, team, date):
    """Return the Elo rating for a team just before a given date."""
    entries = history.get(team, [])
    if not entries:
        return START_ELO
    # Binary search for last entry before date
    lo, hi = 0, len(entries) - 1
    result = START_ELO
    while lo <= hi:
        mid = (lo + hi) // 2
        if entries[mid][0] < date:
            result = entries[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def get_current_elo(elo, team):
    return elo.get(team, START_ELO)


def top_rankings(elo, n=30):
    return sorted(elo.items(), key=lambda x: -x[1])[:n]


if __name__ == "__main__":
    print("Building Elo ratings from international results…")
    elo, history, df = build_elo()
    print(f"  {len(elo)} teams rated")
    print("\nTop 20 current Elo ratings:")
    for i, (team, rating) in enumerate(top_rankings(elo, 20), 1):
        print(f"  {i:2d}. {team:<30} {rating:.0f}")

    with open("elo.pkl", "wb") as f:
        pickle.dump({"elo": elo, "history": history}, f)
    print("\nSaved elo.pkl")
