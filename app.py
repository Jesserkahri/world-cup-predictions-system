import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import itertools
import random

# ─── Page config (must be first) ──────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Auto-train if artifacts missing ──────────────────────────────────────────
artifacts = ["model.pkl", "team_history.pkl", "cv_results.pkl"]
if not all(os.path.exists(a) for a in artifacts):
    with st.spinner("First run – training models on historical data… (~15s)"):
        import model as m
        feat_df, team_history = m.load_and_engineer()
        mdl, best_name, cv_results = m.train(feat_df)
        with open("model.pkl", "wb") as f:  pickle.dump(mdl, f)
        with open("team_history.pkl", "wb") as f:  pickle.dump(team_history, f)
        with open("cv_results.pkl", "wb") as f:
            pickle.dump({"results": cv_results, "best": best_name}, f)

@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb")        as f: mdl          = pickle.load(f)
    with open("team_history.pkl", "rb") as f: team_history = pickle.load(f)
    with open("cv_results.pkl", "rb")   as f: cv_info      = pickle.load(f)
    return mdl, team_history, cv_info

mdl, team_history, cv_info = load_artifacts()
from model import predict_match

df_raw = pd.read_csv("matches.csv")

# ─── 2026 World Cup Groups ────────────────────────────────────────────────────
# Name aliases: model name → display name
ALIASES = {
    "Turkey": "Türkiye",
}
REV_ALIASES = {v: k for k, v in ALIASES.items()}  # display → model

GROUPS = {
    "A": ["Mexico", "Czech Republic", "South Africa", "South Korea"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "D": ["United States", "Australia", "Paraguay", "Türkiye"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

def model_name(team):
    """Convert display name → model lookup name."""
    return REV_ALIASES.get(team, team)

def predict(home, away, knockout=0):
    return predict_match(mdl, team_history, model_name(home), model_name(away),
                         stage_knockout=knockout, year=2026)

# ─── Tournament simulation helpers ────────────────────────────────────────────
def sim_match(home, away, knockout=False):
    """
    Returns (winner_or_None, home_pts, away_pts, home_gd_delta, away_gd_delta).
    In knockout mode, draws resolved by penalty shootout (50/50).
    """
    r = predict(home, away, knockout=int(knockout))
    hw, dp, aw = r["home_win_prob"], r["draw_prob"], r["away_win_prob"]
    total = hw + dp + aw
    rnd = random.random() * total
    if rnd < hw:
        outcome = "home"
    elif rnd < hw + dp:
        outcome = "draw"
    else:
        outcome = "away"

    if knockout and outcome == "draw":
        outcome = "home" if random.random() < 0.5 else "away"

    # Estimate scoreline from stats
    hgs = r["home_stats"]["goals_scored"]
    ags = r["away_stats"]["goals_scored"]
    if outcome == "home":
        hg = max(1, round(hgs)); ag = max(0, round(ags) - 1)
        if hg <= ag: hg = ag + 1
        return "home", 3, 0, hg - ag, ag - hg
    elif outcome == "away":
        ag = max(1, round(ags)); hg = max(0, round(hgs) - 1)
        if ag <= hg: ag = hg + 1
        return "away", 0, 3, hg - ag, ag - hg
    else:
        g = max(0, round((hgs + ags) / 2))
        return "draw", 1, 1, 0, 0

def simulate_group(teams):
    """Returns standings: list of (team, pts, gd, gf) sorted."""
    pts = {t: 0 for t in teams}
    gd  = {t: 0 for t in teams}
    gf  = {t: 0 for t in teams}
    results = {}
    for home, away in itertools.combinations(teams, 2):
        outcome, hp, ap, hgd, agd = sim_match(home, away)
        pts[home] += hp; pts[away] += ap
        gd[home]  += hgd; gd[away] += agd
        # rough goal tallies
        hg_scored = max(0, hgd) if outcome == "home" else (1 if outcome == "draw" else max(0, -agd))
        ag_scored = max(0, agd) if outcome == "away" else (1 if outcome == "draw" else max(0, -hgd))
        gf[home] += hg_scored; gf[away] += ag_scored
        results[(home, away)] = outcome
    standings = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    return standings, pts, gd, results

def simulate_tournament(n=200):
    """Run n Monte Carlo simulations, tally champion counts."""
    from collections import Counter
    champ_counts = Counter()

    for _ in range(n):
        # ── Group stage ──
        group_winners = {}   # group letter → [1st, 2nd]
        third_place = []     # all 3rd-place teams with their pts/gd

        for grp, teams in GROUPS.items():
            standings, pts, gd, _ = simulate_group(teams)
            group_winners[grp] = standings[:2]
            third = standings[2]
            third_place.append((pts[third], gd[third], third))

        # Best 8 third-place teams advance (32 teams × 2 per group = 24, plus 8 thirds = 32 in R32)
        third_place.sort(reverse=True)
        advancing_thirds = [t[2] for t in third_place[:8]]

        # Build R32 pool: 24 group qualifiers + 8 best thirds
        r32_pool = []
        for grp in sorted(GROUPS.keys()):
            r32_pool.extend(group_winners[grp])
        r32_pool.extend(advancing_thirds)
        random.shuffle(r32_pool)

        # ── Knockout rounds ──
        bracket = r32_pool[:]
        while len(bracket) > 1:
            next_round = []
            for i in range(0, len(bracket), 2):
                if i + 1 < len(bracket):
                    outcome, *_ = sim_match(bracket[i], bracket[i+1], knockout=True)
                    winner = bracket[i] if outcome == "home" else bracket[i+1]
                else:
                    winner = bracket[i]
                next_round.append(winner)
            bracket = next_round

        champ_counts[bracket[0]] += 1

    total = sum(champ_counts.values())
    return {t: round(c / total * 100, 1) for t, c in champ_counts.most_common()}

def run_group_predictions():
    """Run one deterministic-ish pass (use probabilities, not random) for display."""
    all_group_results = {}
    for grp, teams in GROUPS.items():
        pts = {t: 0 for t in teams}
        gd  = {t: 0 for t in teams}
        match_results = {}
        for home, away in itertools.combinations(teams, 2):
            r = predict(home, away, knockout=0)
            hw, dp, aw = r["home_win_prob"], r["draw_prob"], r["away_win_prob"]
            if hw >= aw and hw >= dp:
                outcome = "home"; hp, ap = 3, 0; hgd = 1; agd = -1
            elif aw > hw and aw >= dp:
                outcome = "away"; hp, ap = 0, 3; hgd = -1; agd = 1
            else:
                outcome = "draw"; hp, ap = 1, 1; hgd = 0; agd = 0
            pts[home] += hp; pts[away] += ap
            gd[home]  += hgd; gd[away] += agd
            match_results[(home, away)] = (outcome, hw, dp, aw)
        standings = sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True)
        all_group_results[grp] = {"standings": standings, "pts": pts, "gd": gd,
                                   "matches": match_results}
    return all_group_results

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { background-color: #080810; color: #e0e0f0; }
h1,h2,h3,h4 { font-family:'Bebas Neue',sans-serif; letter-spacing:0.06em; }

.big-title {
    font-family:'Bebas Neue',sans-serif;
    font-size:3rem; letter-spacing:0.1em;
    background:linear-gradient(135deg,#f5c518 0%,#ff6b35 55%,#ff3366 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    text-align:center; margin-bottom:0;
}
.subtitle { text-align:center; color:#666; font-size:0.78rem; letter-spacing:0.2em;
            text-transform:uppercase; margin-bottom:1.5rem; }

.group-card {
    background:#0f0f1e; border:1px solid #1e1e3a;
    border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.6rem;
}
.group-header {
    font-family:'Bebas Neue',sans-serif; font-size:1.3rem;
    color:#f5c518; letter-spacing:0.1em; margin-bottom:0.5rem;
}
.team-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:0.3rem 0.5rem; border-radius:6px; margin:2px 0;
    font-size:0.85rem;
}
.team-row.qualified { background:#0d2b0d; border-left:3px solid #4ade80; }
.team-row.third      { background:#1a1a00; border-left:3px solid #f5c518; }
.team-row.eliminated { background:#1a0d0d; border-left:3px solid #444; color:#666; }
.team-name { font-weight:600; }
.team-pts  { color:#f5c518; font-weight:700; font-size:0.9rem; }
.team-gd   { color:#888; font-size:0.78rem; }

.match-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:0.4rem 0.6rem; background:#0f0f1e;
    border-radius:6px; margin:3px 0; font-size:0.82rem;
}
.match-home { flex:1; text-align:right; padding-right:0.6rem; }
.match-away { flex:1; text-align:left;  padding-left:0.6rem; }
.match-pred { font-family:'Bebas Neue',sans-serif; font-size:0.9rem;
              padding:0.1rem 0.5rem; border-radius:4px; text-align:center;
              min-width:60px; }
.pred-home { background:#0d2b0d; color:#4ade80; }
.pred-draw { background:#1a1a00; color:#f5c518; }
.pred-away { background:#2b0d0d; color:#f87171; }
.winner-bold { font-weight:700; color:#fff; }

.champ-card {
    background:linear-gradient(135deg,#1a1400 0%,#0f0f1e 100%);
    border:1px solid #f5c51844; border-radius:12px;
    padding:1rem 1.2rem; margin:0.4rem 0; display:flex;
    align-items:center; gap:0.8rem;
}
.champ-rank { font-family:'Bebas Neue',sans-serif; font-size:1.8rem; color:#333; min-width:32px; }
.champ-name { font-weight:700; font-size:1rem; flex:1; }
.champ-pct  { font-family:'Bebas Neue',sans-serif; font-size:1.6rem; color:#f5c518; }

.bracket-match {
    background:#0f0f1e; border:1px solid #1e1e3a;
    border-radius:8px; padding:0.5rem 0.8rem; margin:4px 0;
    font-size:0.82rem;
}
.bracket-winner { color:#4ade80; font-weight:700; }
.bracket-loser  { color:#555; }

[data-testid="stSidebar"] { background-color:#0a0a14; border-right:1px solid #1a1a2e; }
.stTabs [data-baseweb="tab"] { font-family:'Bebas Neue',sans-serif; letter-spacing:0.1em;
                                font-size:1rem; }
.stButton>button {
    background:linear-gradient(135deg,#f5c518,#ff6b35);
    color:#080810; font-family:'Bebas Neue',sans-serif;
    letter-spacing:0.12em; font-size:1rem; border:none;
    border-radius:8px; width:100%;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="big-title">🏆 FIFA WORLD CUP 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Tournament Predictor · Historical FIFA Data · Machine Learning</div>',
            unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Model")
    best = cv_info["best"]
    st.markdown(f"**Best model:** {best}")
    for name, res in cv_info["results"].items():
        st.caption(f"{name}: {res['mean']*100:.1f}%")
        st.progress(float(res["mean"]))
    st.markdown("---")
    st.markdown("### ⚙️ Simulation")
    n_sims = st.slider("Monte Carlo runs (champion %)", 100, 1000, 300, step=100)
    st.caption("More runs = more accurate championship probabilities.")
    st.markdown("---")
    st.caption("Data: jfjelstul/worldcup • scikit-learn")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_groups, tab_bracket, tab_champ = st.tabs([
    "⚽  GROUP STAGE", "🥊  KNOCKOUT BRACKET", "🏆  CHAMPION PREDICTION"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – GROUP STAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_groups:
    st.markdown("### Predicted Group Stage Standings")
    st.caption("Standings based on most-likely outcome per match. Green = qualified (top 2), Yellow = possible best-third.")

    if st.button("🔄 Recalculate Group Stage", key="grp_btn"):
        st.cache_data.clear()

    @st.cache_data
    def cached_group_predictions():
        return run_group_predictions()

    grp_data = cached_group_predictions()

    # Display groups in a 3-column grid
    group_keys = sorted(GROUPS.keys())
    for row_start in range(0, len(group_keys), 3):
        cols = st.columns(3)
        for ci, grp in enumerate(group_keys[row_start:row_start+3]):
            with cols[ci]:
                data = grp_data[grp]
                standings = data["standings"]
                pts_map   = data["pts"]
                gd_map    = data["gd"]

                st.markdown(f'<div class="group-header">GROUP {grp}</div>', unsafe_allow_html=True)

                for rank, team in enumerate(standings):
                    css = "qualified" if rank < 2 else ("third" if rank == 2 else "eliminated")
                    badge = "✅" if rank < 2 else ("⚡" if rank == 2 else "❌")
                    gd_str = f"{gd_map[team]:+d}"
                    st.markdown(f"""
                    <div class="team-row {css}">
                        <span class="team-name">{badge} {team}</span>
                        <span>
                            <span class="team-pts">{pts_map[team]}pts</span>
                            <span class="team-gd"> GD{gd_str}</span>
                        </span>
                    </div>""", unsafe_allow_html=True)

                # Match results
                with st.expander("Match predictions"):
                    for (home, away), (outcome, hw, dp, aw) in data["matches"].items():
                        if outcome == "home":
                            pred_css = "pred-home"; pred_lbl = f"{home[:3].upper()} WIN"
                            h_cls = "winner-bold"; a_cls = ""
                        elif outcome == "away":
                            pred_css = "pred-away"; pred_lbl = f"{away[:3].upper()} WIN"
                            h_cls = ""; a_cls = "winner-bold"
                        else:
                            pred_css = "pred-draw"; pred_lbl = "DRAW"
                            h_cls = ""; a_cls = ""
                        st.markdown(f"""
                        <div class="match-row">
                            <span class="match-home {h_cls}">{home}</span>
                            <span class="match-pred {pred_css}">{pred_lbl}</span>
                            <span class="match-away {a_cls}">{away}</span>
                        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – KNOCKOUT BRACKET (one simulated run)
# ══════════════════════════════════════════════════════════════════════════════
with tab_bracket:
    st.markdown("### Simulated Knockout Bracket")
    st.caption("One full tournament simulation. Each click runs a fresh simulation.")

    if st.button("🎲 Simulate New Bracket", key="bracket_btn"):
        if "bracket_seed" not in st.session_state:
            st.session_state.bracket_seed = 0
        st.session_state.bracket_seed += 1

    seed = st.session_state.get("bracket_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    @st.cache_data
    def run_single_bracket(seed):
        random.seed(seed); np.random.seed(seed)
        group_adv = {}
        third_pool = []
        for grp, teams in GROUPS.items():
            standings, pts, gd, _ = simulate_group(teams)
            group_adv[grp] = standings[:2]
            third = standings[2]
            third_pool.append((pts[third], gd[third], grp, third))

        third_pool.sort(reverse=True)
        advancing_thirds = [t[3] for t in third_pool[:8]]

        # Build seeded R32
        r32 = []
        for grp in sorted(GROUPS.keys()):
            r32.extend(group_adv[grp])
        r32.extend(advancing_thirds)

        rounds = {"Round of 32": [], "Round of 16": [], "Quarter-finals": [],
                  "Semi-finals": [], "Final": []}
        bracket = r32[:]
        round_names = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
        ri = 0
        while len(bracket) > 1:
            rname = round_names[ri] if ri < len(round_names) else f"Round {ri+1}"
            next_round = []
            for i in range(0, len(bracket), 2):
                if i + 1 < len(bracket):
                    outcome, *_ = sim_match(bracket[i], bracket[i+1], knockout=True)
                    winner = bracket[i] if outcome == "home" else bracket[i+1]
                    loser  = bracket[i+1] if outcome == "home" else bracket[i]
                    rounds[rname].append((bracket[i], bracket[i+1], winner, loser))
                    next_round.append(winner)
                else:
                    next_round.append(bracket[i])
            bracket = next_round
            ri += 1
        return rounds, bracket[0], group_adv, advancing_thirds

    rounds, champion, group_adv, thirds = run_single_bracket(seed)

    st.success(f"🏆 **Predicted Champion: {champion}**")

    # Show group qualifiers summary
    with st.expander("Group qualifiers (this simulation)"):
        col_list = st.columns(4)
        for gi, (grp, qualifiers) in enumerate(sorted(group_adv.items())):
            with col_list[gi % 4]:
                st.caption(f"Group {grp}")
                for q in qualifiers:
                    st.markdown(f"✅ {q}")
        st.caption(f"Best-third advancing: {', '.join(thirds)}")

    # Bracket rounds
    round_order = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
    for rname in round_order:
        matches = rounds.get(rname, [])
        if not matches:
            continue
        st.markdown(f"#### {rname}")
        ncols = min(len(matches), 4)
        cols = st.columns(ncols) if ncols > 1 else [st]
        for mi, (home, away, winner, loser) in enumerate(matches):
            with cols[mi % ncols]:
                h_cls = "bracket-winner" if winner == home else "bracket-loser"
                a_cls = "bracket-winner" if winner == away else "bracket-loser"
                st.markdown(f"""
                <div class="bracket-match">
                    <div class="{h_cls}">{home}</div>
                    <div style="color:#333;font-size:0.7rem;padding:1px 0">vs</div>
                    <div class="{a_cls}">{away}</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – CHAMPION PROBABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_champ:
    st.markdown(f"### Championship Probability ({n_sims} simulations)")
    st.caption("Monte Carlo: runs the full tournament hundreds of times, tallies who wins.")

    if st.button(f"🚀 Run {n_sims} Simulations", key="champ_btn"):
        st.cache_data.clear()

    @st.cache_data
    def cached_champion_probs(n):
        return simulate_tournament(n)

    with st.spinner(f"Simulating {n_sims} tournaments…"):
        champ_probs = cached_champion_probs(n_sims)

    # Top 16 podium display
    top_n = list(champ_probs.items())[:16]
    medals = ["🥇", "🥈", "🥉"] + [""] * 20

    # Bar chart
    chart_df = pd.DataFrame(top_n, columns=["Team", "Win %"])
    st.bar_chart(chart_df.set_index("Team"), use_container_width=True, height=320)

    # Cards
    st.markdown("#### Top Contenders")
    col1, col2 = st.columns(2)
    for i, (team, pct) in enumerate(top_n):
        medal = medals[i]
        bar_w = int(pct / max(v for _, v in top_n) * 100)
        card_html = f"""
        <div class="champ-card">
            <span class="champ-rank">{i+1}</span>
            <div style="flex:1">
                <div class="champ-name">{medal} {team}</div>
                <div style="background:#1e1e3a;border-radius:4px;height:6px;margin-top:4px;overflow:hidden;">
                    <div style="width:{bar_w}%;height:100%;background:linear-gradient(90deg,#f5c518,#ff6b35);border-radius:4px;"></div>
                </div>
            </div>
            <span class="champ-pct">{pct}%</span>
        </div>"""
        if i % 2 == 0:
            col1.markdown(card_html, unsafe_allow_html=True)
        else:
            col2.markdown(card_html, unsafe_allow_html=True)

    if len(champ_probs) > 16:
        with st.expander("All teams with championship wins"):
            rest = list(champ_probs.items())[16:]
            for team, pct in rest:
                st.caption(f"{team}: {pct}%")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ For entertainment only. Predictions based on historical World Cup data (1950–2022). "
           "Teams with no WC history use league-average stats. Model accuracy ~55% per match.")