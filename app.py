import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, itertools, random

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Auto-build artifacts ─────────────────────────────────────────────────────
needed_wc  = ["model_wc.pkl","wc_only_history.pkl","cv_results_wc.pkl"]
needed_elo = ["model.pkl","wc_history.pkl","elo_dict.pkl","cv_results.pkl","elo.pkl"]

if not all(os.path.exists(a) for a in needed_wc):
    with st.spinner("Building WC-only model…"):
        import model_wc as mwc
        feat_df, wc_hist, FCOLS = mwc.load_and_engineer()
        mdl, best, cvr = mwc.train(feat_df, FCOLS)
        with open("model_wc.pkl","wb")       as f: pickle.dump(mdl,f)
        with open("wc_only_history.pkl","wb") as f: pickle.dump(wc_hist,f)
        with open("cv_results_wc.pkl","wb")   as f: pickle.dump({"results":cvr,"best":best,"feature_cols":FCOLS},f)

if not all(os.path.exists(a) for a in needed_elo):
    with st.spinner("Building Elo ratings & Elo model (~30s)…"):
        import elo as elo_mod
        e,h,_ = elo_mod.build_elo()
        with open("elo.pkl","wb") as f: pickle.dump({"elo":e,"history":h},f)
        import model as m
        feat_df, wc_history, elo_dict, _ = m.load_and_engineer()
        mdl, best, cvr = m.train(feat_df)
        with open("model.pkl","wb")      as f: pickle.dump(mdl,f)
        with open("wc_history.pkl","wb") as f: pickle.dump(wc_history,f)
        with open("elo_dict.pkl","wb")   as f: pickle.dump(elo_dict,f)
        with open("cv_results.pkl","wb") as f: pickle.dump({"results":cvr,"best":best},f)

# ─── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    with open("model_wc.pkl","rb")        as f: mdl_wc    = pickle.load(f)
    with open("wc_only_history.pkl","rb") as f: hist_wc   = pickle.load(f)
    with open("cv_results_wc.pkl","rb")   as f: cv_wc     = pickle.load(f)
    with open("model.pkl","rb")           as f: mdl_elo   = pickle.load(f)
    with open("wc_history.pkl","rb")      as f: hist_elo  = pickle.load(f)
    with open("elo_dict.pkl","rb")        as f: elo_dict  = pickle.load(f)
    with open("cv_results.pkl","rb")      as f: cv_elo    = pickle.load(f)
    return mdl_wc, hist_wc, cv_wc, mdl_elo, hist_elo, elo_dict, cv_elo

mdl_wc, hist_wc, cv_wc, mdl_elo, hist_elo, elo_dict, cv_elo = load_all()

from model    import predict_match  as predict_elo
from model_wc import predict_match  as predict_wc_only
from elo      import START_ELO, top_rankings

df_raw = pd.read_csv("matches.csv")
df_modern = df_raw[df_raw["match_date"].str[:4].astype(int) >= 1990]
ALL_TEAMS = sorted(set(df_modern["home_team_name"].tolist() + df_modern["away_team_name"].tolist()))

# ─── 2026 groups & aliases ────────────────────────────────────────────────────
ALIASES     = {"Turkey": "Türkiye"}
REV_ALIASES = {v: k for k, v in ALIASES.items()}
GROUPS = {
    "A": ["Mexico","Czech Republic","South Africa","South Korea"],
    "B": ["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
    "C": ["Brazil","Haiti","Morocco","Scotland"],
    "D": ["United States","Australia","Paraguay","Türkiye"],
    "E": ["Germany","Ecuador","Ivory Coast","Curaçao"],
    "F": ["Netherlands","Japan","Sweden","Tunisia"],
    "G": ["Belgium","Egypt","Iran","New Zealand"],
    "H": ["Spain","Uruguay","Saudi Arabia","Cape Verde"],
    "I": ["France","Senegal","Iraq","Norway"],
    "J": ["Argentina","Algeria","Austria","Jordan"],
    "K": ["Portugal","Colombia","Uzbekistan","DR Congo"],
    "L": ["England","Croatia","Ghana","Panama"],
}
def mname(t): return REV_ALIASES.get(t, t)
def get_elo(t): return round(elo_dict.get(mname(t), START_ELO))

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{background:#080810;color:#e0e0f0;}
h1,h2,h3,h4{font-family:'Bebas Neue',sans-serif;letter-spacing:.06em;}

.big-title{font-family:'Bebas Neue',sans-serif;font-size:3rem;letter-spacing:.1em;
    background:linear-gradient(135deg,#f5c518 0%,#ff6b35 55%,#ff3366 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    text-align:center;margin-bottom:0;}
.subtitle{text-align:center;color:#555;font-size:.78rem;letter-spacing:.2em;
    text-transform:uppercase;margin-bottom:.5rem;}

/* Mode selector pills */
.mode-bar{display:flex;gap:.6rem;justify-content:center;margin:1.2rem 0 1.8rem;}
.mode-pill{padding:.55rem 1.4rem;border-radius:30px;font-family:'Bebas Neue',sans-serif;
    font-size:1rem;letter-spacing:.08em;cursor:pointer;border:1px solid #2a2a4a;
    background:#0f0f1e;color:#888;transition:all .2s;}
.mode-pill.active-wc  {background:#1a1a00;border-color:#f5c518;color:#f5c518;}
.mode-pill.active-elo {background:#001a1a;border-color:#4ade80;color:#4ade80;}
.mode-pill.active-soon{background:#1a001a;border-color:#a78bfa;color:#a78bfa;}

.mode-badge-wc  {display:inline-block;background:#f5c51822;border:1px solid #f5c51866;
    color:#f5c518;border-radius:6px;padding:.1rem .5rem;font-size:.72rem;
    letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;}
.mode-badge-elo {display:inline-block;background:#4ade8022;border:1px solid #4ade8066;
    color:#4ade80;border-radius:6px;padding:.1rem .5rem;font-size:.72rem;
    letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;}
.mode-badge-soon{display:inline-block;background:#a78bfa22;border:1px solid #a78bfa66;
    color:#a78bfa;border-radius:6px;padding:.1rem .5rem;font-size:.72rem;
    letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;}

.team-row{display:flex;justify-content:space-between;align-items:center;
    padding:.3rem .5rem;border-radius:6px;margin:2px 0;font-size:.85rem;}
.team-row.qualified{background:#0d2b0d;border-left:3px solid #4ade80;}
.team-row.third     {background:#1a1a00;border-left:3px solid #f5c518;}
.team-row.eliminated{background:#1a0d0d;border-left:3px solid #333;color:#555;}
.team-pts{color:#f5c518;font-weight:700;}
.team-gd {color:#666;font-size:.78rem;}
.elo-badge{background:#1e1e3a;border:1px solid #2a2a4a;border-radius:4px;
    padding:.05rem .3rem;font-size:.68rem;color:#7a8abf;font-family:monospace;margin-left:3px;}

.match-row{display:flex;justify-content:space-between;align-items:center;
    padding:.35rem .6rem;background:#0f0f1e;border-radius:6px;margin:3px 0;font-size:.82rem;}
.match-home{flex:1;text-align:right;padding-right:.6rem;}
.match-away{flex:1;text-align:left;padding-left:.6rem;}
.match-pred{font-family:'Bebas Neue',sans-serif;font-size:.85rem;padding:.1rem .45rem;
    border-radius:4px;text-align:center;min-width:56px;}
.pred-home{background:#0d2b0d;color:#4ade80;}
.pred-draw{background:#1a1a00;color:#f5c518;}
.pred-away{background:#2b0d0d;color:#f87171;}
.winner-bold{font-weight:700;color:#fff;}

.champ-card{background:linear-gradient(135deg,#1a1400,#0f0f1e);
    border:1px solid #f5c51833;border-radius:10px;
    padding:.8rem 1rem;margin:.3rem 0;display:flex;align-items:center;gap:.7rem;}
.champ-rank{font-family:'Bebas Neue',sans-serif;font-size:1.6rem;color:#333;min-width:28px;}
.champ-name{font-weight:700;font-size:.95rem;flex:1;}
.champ-pct{font-family:'Bebas Neue',sans-serif;font-size:1.5rem;color:#f5c518;}

.bracket-match{background:#0f0f1e;border:1px solid #1e1e3a;border-radius:8px;
    padding:.45rem .7rem;margin:4px 0;font-size:.82rem;}
.bracket-winner{color:#4ade80;font-weight:700;}
.bracket-loser {color:#444;}

.soon-box{background:#0f0f1a;border:2px dashed #a78bfa44;border-radius:16px;
    padding:3rem 2rem;text-align:center;margin:2rem 0;}

[data-testid="stSidebar"]{background:#0a0a14;}
.stTabs [data-baseweb="tab"]{font-family:'Bebas Neue',sans-serif;letter-spacing:.08em;font-size:.95rem;}
.stButton>button{background:linear-gradient(135deg,#f5c518,#ff6b35);color:#080810;
    font-family:'Bebas Neue',sans-serif;letter-spacing:.12em;font-size:1rem;
    border:none;border-radius:8px;width:100%;}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="big-title">🏆 FIFA WORLD CUP 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Match Predictor · Three Prediction Modes</div>',
            unsafe_allow_html=True)

# ─── Mode selector ────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "wc"

col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("⚽  WC DATA ONLY", use_container_width=True, key="btn_wc"):
        st.session_state.mode = "wc"
with col_b:
    if st.button("📊  WC + ELO RATINGS", use_container_width=True, key="btn_elo"):
        st.session_state.mode = "elo"
with col_c:
    if st.button("🔮  COMING SOON", use_container_width=True, key="btn_soon"):
        st.session_state.mode = "soon"

mode = st.session_state.mode

# Highlight active mode
badge_html = {
    "wc":   '<div style="text-align:center"><span class="mode-badge-wc">⚽ Mode: World Cup Data Only &nbsp;·&nbsp; ~55% accuracy</span></div>',
    "elo":  '<div style="text-align:center"><span class="mode-badge-elo">📊 Mode: WC Data + Elo Ratings &nbsp;·&nbsp; ~56% accuracy · 49k intl matches</span></div>',
    "soon": '<div style="text-align:center"><span class="mode-badge-soon">🔮 Mode: Coming Soon</span></div>',
}
st.markdown(badge_html[mode], unsafe_allow_html=True)
st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# COMING SOON placeholder
# ══════════════════════════════════════════════════════════════════════════════
if mode == "soon":
    st.markdown("""
    <div class="soon-box">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:3rem;color:#a78bfa;
                    letter-spacing:.1em;margin-bottom:.5rem">🔮 Coming Soon</div>
        <div style="color:#666;font-size:.95rem;max-width:400px;margin:0 auto;line-height:1.7">
            This mode is under construction.<br>
            Ideas in the pipeline:<br>
            <span style="color:#a78bfa">Player-level stats · Club form · Transfer market data · Neural network model</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SHARED: pick model & predict functions based on mode
# ══════════════════════════════════════════════════════════════════════════════
if mode == "wc":
    active_mdl  = mdl_wc
    active_hist = hist_wc
    active_cv   = cv_wc
    def do_predict(home, away, knockout=0):
        return predict_wc_only(active_mdl, active_hist,
                               mname(home), mname(away),
                               stage_knockout=knockout, year=2026)
else:  # elo
    active_mdl  = mdl_elo
    active_hist = hist_elo
    active_cv   = cv_elo
    def do_predict(home, away, knockout=0):
        return predict_elo(active_mdl, active_hist, elo_dict,
                           mname(home), mname(away),
                           stage_knockout=knockout, year=2026)

# ─── Shared simulation helpers ────────────────────────────────────────────────
def sim_match(home, away, knockout=False):
    r = do_predict(home, away, knockout=int(knockout))
    hw, dp, aw = r["home_win_prob"], r["draw_prob"], r["away_win_prob"]
    rnd = random.random() * (hw + dp + aw)
    outcome = "home" if rnd < hw else ("draw" if rnd < hw+dp else "away")
    if knockout and outcome == "draw":
        outcome = "home" if random.random() < 0.5 else "away"
    hgs = r["home_stats"]["goals_scored"]
    ags = r["away_stats"]["goals_scored"]
    if outcome == "home":
        hg = max(1, round(hgs)); ag = max(0, round(ags)-1)
        if hg <= ag: hg = ag+1
        return "home", 3, 0, hg-ag, ag-hg
    elif outcome == "away":
        ag = max(1, round(ags)); hg = max(0, round(hgs)-1)
        if ag <= hg: ag = hg+1
        return "away", 0, 3, hg-ag, ag-hg
    return "draw", 1, 1, 0, 0

def simulate_group(teams):
    pts={t:0 for t in teams}; gd={t:0 for t in teams}; gf={t:0 for t in teams}
    results={}
    for home, away in itertools.combinations(teams, 2):
        outcome,hp,ap,hgd,agd = sim_match(home, away)
        pts[home]+=hp; pts[away]+=ap; gd[home]+=hgd; gd[away]+=agd
        hg = max(0,hgd) if outcome=="home" else (1 if outcome=="draw" else max(0,-agd))
        ag = max(0,agd) if outcome=="away" else (1 if outcome=="draw" else max(0,-hgd))
        gf[home]+=hg; gf[away]+=ag
        results[(home,away)] = outcome
    standings = sorted(teams, key=lambda t:(pts[t],gd[t],gf[t]), reverse=True)
    return standings, pts, gd, results

def simulate_tournament(n=200):
    from collections import Counter
    counts = Counter()
    for _ in range(n):
        gw={}; thirds=[]
        for grp, teams in GROUPS.items():
            standings,pts,gd,_ = simulate_group(teams)
            gw[grp] = standings[:2]
            t = standings[2]
            thirds.append((pts[t],gd[t],t))
        thirds.sort(reverse=True)
        adv_thirds = [t[2] for t in thirds[:8]]
        pool = []
        for grp in sorted(GROUPS.keys()): pool.extend(gw[grp])
        pool.extend(adv_thirds); random.shuffle(pool)
        bracket = pool[:]
        while len(bracket) > 1:
            nxt=[]
            for i in range(0,len(bracket),2):
                if i+1<len(bracket):
                    out,*_ = sim_match(bracket[i],bracket[i+1],knockout=True)
                    nxt.append(bracket[i] if out=="home" else bracket[i+1])
                else: nxt.append(bracket[i])
            bracket=nxt
        counts[bracket[0]] += 1
    total = sum(counts.values())
    return {t: round(c/total*100,1) for t,c in counts.most_common()}

def run_group_deterministic():
    all_res={}
    for grp, teams in GROUPS.items():
        pts={t:0 for t in teams}; gd={t:0 for t in teams}; matches={}
        for home, away in itertools.combinations(teams, 2):
            r = do_predict(home, away, knockout=0)
            hw,dp,aw = r["home_win_prob"],r["draw_prob"],r["away_win_prob"]
            if hw>=aw and hw>=dp:   out="home"; hp,ap=3,0; hgd,agd=1,-1
            elif aw>hw and aw>=dp:  out="away"; hp,ap=0,3; hgd,agd=-1,1
            else:                   out="draw"; hp,ap=1,1;  hgd,agd=0,0
            pts[home]+=hp; pts[away]+=ap; gd[home]+=hgd; gd[away]+=agd
            matches[(home,away)] = (out,hw,dp,aw)
        standings = sorted(teams, key=lambda t:(pts[t],gd[t]),reverse=True)
        all_res[grp] = {"standings":standings,"pts":pts,"gd":gd,"matches":matches}
    return all_res

# ─── Shared UI helpers ────────────────────────────────────────────────────────
def prob_bars(home_lbl, hw, away_lbl, aw, dp):
    for lbl, prob, color in [
        (f"🏠 {home_lbl}", hw, "#4ade80"),
        ("🤝 Draw",         dp, "#f5c518"),
        (f"✈️  {away_lbl}", aw, "#f87171"),
    ]:
        st.markdown(f"""
        <div style="margin:.4rem 0">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px">
            <span style="font-size:.87rem">{lbl}</span>
            <span style="font-weight:700;color:{color}">{prob}%</span>
          </div>
          <div style="background:#1e1e3a;border-radius:5px;height:10px;overflow:hidden">
            <div style="width:{int(prob)}%;height:100%;background:{color};border-radius:5px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

def result_banner(label, color, emoji):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #2a2a4a;
                border-radius:12px;padding:1.3rem;margin:.8rem 0;text-align:center">
        <div style="color:#666;font-size:.7rem;letter-spacing:.18em;
                    text-transform:uppercase;margin-bottom:.3rem">Predicted Outcome</div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:2.4rem;
                    letter-spacing:.1em;color:{color}">{emoji} {label}</div>
    </div>""", unsafe_allow_html=True)

def team_stats_cols(res, home, away):
    hs, as_ = res["home_stats"], res["away_stats"]
    sc1, sc2 = st.columns(2)
    for col, stats, tname, elo_key in [
        (sc1, hs,  home, "home_elo"),
        (sc2, as_, away, "away_elo"),
    ]:
        with col:
            st.markdown(f"**{tname}**")
            rows_m = [
                ("Win Rate",       f"{stats['win_rate']*100:.0f}%"),
                ("Draw Rate",      f"{stats['draw_rate']*100:.0f}%"),
                ("Goals Scored",   f"{stats['goals_scored']:.2f}"),
                ("Goals Conceded", f"{stats['goals_conceded']:.2f}"),
                ("Goal Diff",      f"{stats['gd']:+.2f}"),
                ("WC Matches",     str(stats['matches']) if stats['matches'] else "—"),
            ]
            if mode == "elo" and elo_key in res:
                rows_m.insert(0, ("Elo Rating", str(res[elo_key])))
            r1 = rows_m[:len(rows_m)//2+len(rows_m)%2]
            r2 = rows_m[len(rows_m)//2+len(rows_m)%2:]
            ca, cb = st.columns(2)
            for k,v in r1: ca.metric(k,v)
            for k,v in r2: cb.metric(k,v)

def h2h_section(home, away):
    h2h = df_raw[
        ((df_raw["home_team_name"]==home)&(df_raw["away_team_name"]==away))|
        ((df_raw["home_team_name"]==away)&(df_raw["away_team_name"]==home))
    ]
    st.markdown("---")
    if len(h2h):
        st.markdown("#### Head-to-Head at World Cup")
        hw_c = (len(h2h[(h2h["home_team_name"]==home)&(h2h["home_team_win"]==1)])+
                len(h2h[(h2h["away_team_name"]==home)&(h2h["away_team_win"]==1)]))
        aw_c = (len(h2h[(h2h["home_team_name"]==away)&(h2h["home_team_win"]==1)])+
                len(h2h[(h2h["away_team_name"]==away)&(h2h["away_team_win"]==1)]))
        draws = int(h2h["draw"].sum())
        ca,cb,cc,cd = st.columns(4)
        ca.metric("Matches",len(h2h)); cb.metric(f"{home} Wins",hw_c)
        cc.metric("Draws",draws);      cd.metric(f"{away} Wins",aw_c)
        with st.expander("Full match history"):
            st.dataframe(
                h2h[["match_date","home_team_name","score","away_team_name","result","stage_name"]]
                .sort_values("match_date",ascending=False).reset_index(drop=True),
                use_container_width=True)
    else:
        st.info(f"No previous World Cup meetings between {home} and {away}.")

# ══════════════════════════════════════════════════════════════════════════════
# TABS (same for both WC and ELO mode)
# ══════════════════════════════════════════════════════════════════════════════
tab_grp, tab_ko, tab_champ, tab_custom = st.tabs([
    "⚽  GROUP STAGE", "🥊  KNOCKOUT BRACKET", "🏆  CHAMPION PREDICTION", "🎯  CUSTOM MATCH"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — GROUP STAGE
# ──────────────────────────────────────────────────────────────────────────────
with tab_grp:
    st.markdown("### Group Stage Predictions")
    st.caption("Based on most-likely outcome per match. ✅ Qualified · ⚡ Best-third · ❌ Eliminated")

    if st.button("🔄 Recalculate", key=f"grp_recalc_{mode}"):
        st.cache_data.clear()

    @st.cache_data(show_spinner=False)
    def cached_groups(m):
        return run_group_deterministic()

    grp_data = cached_groups(mode)

    for row_start in range(0, 12, 3):
        cols = st.columns(3)
        for ci, grp in enumerate(sorted(GROUPS.keys())[row_start:row_start+3]):
            with cols[ci]:
                data = grp_data[grp]
                st.markdown(f'<div style="font-family:\'Bebas Neue\',sans-serif;font-size:1.25rem;'
                            f'color:#f5c518;letter-spacing:.1em;margin-bottom:.4rem">'
                            f'GROUP {grp}</div>', unsafe_allow_html=True)
                for rank, team in enumerate(data["standings"]):
                    css   = "qualified" if rank<2 else ("third" if rank==2 else "eliminated")
                    badge = "✅" if rank<2 else ("⚡" if rank==2 else "❌")
                    elo_span = f'<span class="elo-badge">{get_elo(team)}</span>' if mode=="elo" else ""
                    st.markdown(f"""
                    <div class="team-row {css}">
                        <span>{badge} <b>{team}</b>{elo_span}</span>
                        <span>
                            <span class="team-pts">{data['pts'][team]}pts</span>
                            <span class="team-gd"> GD{data['gd'][team]:+d}</span>
                        </span>
                    </div>""", unsafe_allow_html=True)
                with st.expander("Matches"):
                    for (home,away),(out,hw,dp,aw) in data["matches"].items():
                        if out=="home":   pc,pl,hc,ac="pred-home",f"{home[:3].upper()} WIN","winner-bold",""
                        elif out=="away": pc,pl,hc,ac="pred-away",f"{away[:3].upper()} WIN","","winner-bold"
                        else:             pc,pl,hc,ac="pred-draw","DRAW","",""
                        st.markdown(f"""
                        <div class="match-row">
                            <span class="match-home {hc}">{home}</span>
                            <span class="match-pred {pc}">{pl}</span>
                            <span class="match-away {ac}">{away}</span>
                        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — KNOCKOUT BRACKET
# ──────────────────────────────────────────────────────────────────────────────
with tab_ko:
    st.markdown("### Knockout Bracket Simulation")
    st.caption("Probabilistic simulation. Click to re-roll.")

    seed_key = f"bracket_seed_{mode}"
    if st.button("🎲 New Simulation", key=f"bracket_btn_{mode}"):
        st.session_state[seed_key] = st.session_state.get(seed_key, 0) + 1

    seed = st.session_state.get(seed_key, 42)
    random.seed(seed); np.random.seed(seed)

    @st.cache_data(show_spinner=False)
    def run_bracket(m, seed):
        random.seed(seed); np.random.seed(seed)
        gw={}; thirds=[]
        for grp, teams in GROUPS.items():
            standings,pts,gd,_ = simulate_group(teams)
            gw[grp] = standings[:2]
            t = standings[2]; thirds.append((pts[t],gd[t],grp,t))
        thirds.sort(reverse=True)
        adv = [t[3] for t in thirds[:8]]
        r32 = []
        for grp in sorted(GROUPS.keys()): r32.extend(gw[grp])
        r32.extend(adv)
        round_names = ["Round of 32","Round of 16","Quarter-finals","Semi-finals","Final"]
        rounds = {rn:[] for rn in round_names}
        bracket=r32[:]; ri=0
        while len(bracket)>1:
            rn = round_names[ri] if ri<len(round_names) else f"Round {ri+1}"
            nxt=[]
            for i in range(0,len(bracket),2):
                if i+1<len(bracket):
                    out,*_ = sim_match(bracket[i],bracket[i+1],knockout=True)
                    w = bracket[i] if out=="home" else bracket[i+1]
                    l = bracket[i+1] if out=="home" else bracket[i]
                    rounds[rn].append((bracket[i],bracket[i+1],w,l)); nxt.append(w)
                else: nxt.append(bracket[i])
            bracket=nxt; ri+=1
        return rounds, bracket[0], gw, adv

    rounds, champion, gw, adv_thirds = run_bracket(mode, seed)

    elo_str = f"  (Elo: {get_elo(champion)})" if mode=="elo" else ""
    st.success(f"🏆 **Predicted Champion: {champion}**{elo_str}")

    with st.expander("Group qualifiers"):
        gcols = st.columns(4)
        for gi,(grp,qs) in enumerate(sorted(gw.items())):
            with gcols[gi%4]:
                st.caption(f"Group {grp}")
                for q in qs:
                    elo_t = f" `{get_elo(q)}`" if mode=="elo" else ""
                    st.markdown(f"✅ {q}{elo_t}")
        st.caption("Best-third: " + ", ".join(adv_thirds))

    for rn in ["Round of 32","Round of 16","Quarter-finals","Semi-finals","Final"]:
        matches = rounds.get(rn,[])
        if not matches: continue
        st.markdown(f"#### {rn}")
        ncols = min(len(matches),4)
        cols = st.columns(ncols)
        for mi,(home,away,winner,loser) in enumerate(matches):
            with cols[mi%ncols]:
                hc = "bracket-winner" if winner==home else "bracket-loser"
                ac = "bracket-winner" if winner==away else "bracket-loser"
                h_elo = f' <small style="color:#333">{get_elo(home)}</small>' if mode=="elo" else ""
                a_elo = f' <small style="color:#333">{get_elo(away)}</small>' if mode=="elo" else ""
                st.markdown(f"""
                <div class="bracket-match">
                    <div class="{hc}">{home}{h_elo}</div>
                    <div style="color:#2a2a2a;font-size:.7rem;padding:1px 0">vs</div>
                    <div class="{ac}">{away}{a_elo}</div>
                </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — CHAMPION PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
with tab_champ:
    n_sims = st.slider("Simulations", 100, 1000, 300, step=100, key=f"nsims_{mode}")
    st.markdown(f"### Championship Probability  ({n_sims} Monte Carlo runs)")

    if st.button(f"🚀 Run Simulations", key=f"champ_btn_{mode}"):
        st.cache_data.clear()

    @st.cache_data(show_spinner="Simulating tournaments…")
    def cached_champ(m, n):
        return simulate_tournament(n)

    probs = cached_champ(mode, n_sims)
    top_n = list(probs.items())[:16]

    chart_df = pd.DataFrame(top_n, columns=["Team","Win %"])
    st.bar_chart(chart_df.set_index("Team"), use_container_width=True, height=300)

    medals = ["🥇","🥈","🥉"]+[""]*20
    col1,col2 = st.columns(2)
    for i,(team,pct) in enumerate(top_n):
        bw = int(pct/max(v for _,v in top_n)*100)
        elo_span = f'<span class="elo-badge">Elo {get_elo(team)}</span>' if mode=="elo" else ""
        card = f"""
        <div class="champ-card">
            <span class="champ-rank">{i+1}</span>
            <div style="flex:1">
                <div class="champ-name">{medals[i]} {team} {elo_span}</div>
                <div style="background:#1e1e3a;border-radius:4px;height:5px;margin-top:4px;overflow:hidden">
                    <div style="width:{bw}%;height:100%;background:linear-gradient(90deg,#f5c518,#ff6b35);border-radius:4px"></div>
                </div>
            </div>
            <span class="champ-pct">{pct}%</span>
        </div>"""
        (col1 if i%2==0 else col2).markdown(card, unsafe_allow_html=True)

    if len(probs)>16:
        with st.expander("All teams"):
            for t,p in list(probs.items())[16:]: st.caption(f"{t}: {p}%")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — CUSTOM MATCH
# ──────────────────────────────────────────────────────────────────────────────
with tab_custom:
    st.markdown("### Custom Match Predictor")

    c1, cvs, c2 = st.columns([5,1,5])
    with c1:
        st.markdown("**🏠 Home Team**")
        home_pick = st.selectbox("Home", ALL_TEAMS,
            index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0,
            label_visibility="collapsed", key=f"home_{mode}")
    with cvs:
        st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif;font-size:1.7rem;'
                    'color:#f5c518;text-align:center;padding-top:.5rem">VS</div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown("**✈️ Away Team**")
        away_pick = st.selectbox("Away", ALL_TEAMS,
            index=ALL_TEAMS.index("Germany") if "Germany" in ALL_TEAMS else 1,
            label_visibility="collapsed", key=f"away_{mode}")

    # Elo comparison (elo mode only)
    if mode == "elo" and home_pick != away_pick:
        h_ev = elo_dict.get(mname(home_pick), START_ELO)
        a_ev = elo_dict.get(mname(away_pick), START_ELO)
        diff = h_ev - a_ev
        e1,e2,e3 = st.columns(3)
        e1.metric(f"🏠 {home_pick} Elo", f"{h_ev:.0f}")
        e2.metric("Elo Diff", f"{diff:+.0f}")
        e3.metric(f"✈️ {away_pick} Elo", f"{a_ev:.0f}")

    cs1,cs2 = st.columns(2)
    with cs1: stage_pick = st.radio("Stage",["Group Stage","Knockout"],horizontal=True,key=f"stage_{mode}")
    with cs2: year_pick  = st.slider("Year",2022,2030,2026,step=4,key=f"year_{mode}")

    predict_btn = st.button("⚡ Predict Outcome", key=f"predict_btn_{mode}")

    if home_pick == away_pick:
        st.warning("Select two different teams.")
    elif predict_btn:
        ko  = 1 if stage_pick=="Knockout" else 0
        res = do_predict(home_pick, away_pick, knockout=ko)
        pred = res["prediction"]
        hw,dp,aw = res["home_win_prob"],res["draw_prob"],res["away_win_prob"]

        if pred=="Home Win":   color,emoji,lbl="#4ade80","🟢",f"{home_pick} Win"
        elif pred=="Draw":     color,emoji,lbl="#f5c518","🟡","Draw"
        else:                  color,emoji,lbl="#f87171","🔴",f"{away_pick} Win"

        result_banner(lbl, color, emoji)
        st.markdown("#### Win Probabilities")
        prob_bars(home_pick, hw, away_pick, aw, dp)
        st.markdown("---")
        st.markdown("#### Team Stats")
        team_stats_cols(res, home_pick, away_pick)
        h2h_section(mname(home_pick), mname(away_pick))

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ For entertainment only. WC data: 1950–2022. "
           "Elo computed from 49k+ international matches. Accuracy ~55–56%.")
