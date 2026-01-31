#!/usr/bin/env python3
"""
Advanced Monte Carlo CBB season simulator + tournament selection + seeding.

Inputs (defaults):
- cbb_ratings.csv        (must include Team; ideally AdjO, AdjD, AdjT, AdjEM or Current_AdjEM; optionally RQ)
- cbb_conferences.csv    (must include Team and conference column: Conference / Conf / conference)
- schedule.csv           (date,team,opponent,location_value,neutral,game_url,game_id)

Outputs (in --outdir):
- mc_seed_odds_wide.csv
- mc_seed_odds_long.csv
- mc_field_odds.csv
Optional:
- mc_seasons.jsonl       (one JSON object per simulated season; enabled by default, disable with --no-log-jsonl)

Key features:
- Stochastic game simulation with tempo + margin + total noise
- Latent strength evolution (AdjEM) based on surprise (observed - expected)
- Resume evolution based on (result - expected WP) weighted by opponent quality + location
- AQ determination by simulated conference record (wins, wpct, selection score tiebreak)
- Field selection: AQs + at-larges
- ALSO tracks "Selection Top-68 ignoring AQ" probability (committee-style, regardless of AQ status)
"""

from __future__ import annotations

import os
import math
import argparse
import datetime as dt
import json
import time
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -----------------------------
# Name Mapping (plug yours here)
# -----------------------------
NAME_MAPPING = {
   "Iowa State": "Iowa St.", "Brigham Young": "BYU", "Michigan State": "Michigan St.",
   "Saint Mary's (CA)": "Saint Mary's", "St. John's (NY)": "St. John's",
   "Mississippi State": "Mississippi St.", "Ohio State": "Ohio St.",
   "Southern Methodist": "SMU", "Utah State": "Utah St.",
   "San Diego State": "San Diego St.", "Southern California": "USC",
   "NC State": "N.C. State", "Virginia Commonwealth": "VCU",
   "Oklahoma State": "Oklahoma St.", "Louisiana State": "LSU",
   "Penn State": "Penn St.", "Boise State": "Boise St.",
   "Miami (FL)": "Miami FL", "Colorado State": "Colorado St.",
   "Arizona State": "Arizona St.", "Kansas State": "Kansas St.",
   "Florida State": "Florida St.", "McNeese State": "McNeese",
   "Nevada-Las Vegas": "UNLV", "Loyola (IL)": "Loyola Chicago",
   "Kennesaw State": "Kennesaw St.", "Murray State": "Murray St.",
   "Kent State": "Kent St.", "Illinois State": "Illinois St.",
   "College of Charleston": "Charleston", "Cal State Northridge": "CSUN",
   "Wichita State": "Wichita St.", "Miami (OH)": "Miami OH",
   "East Tennessee State": "East Tennessee St.", "South Dakota State": "South Dakota St.",
   "New Mexico State": "New Mexico St.", "Oregon State": "Oregon St.",
   "Jacksonville State": "Jacksonville St.", "Arkansas State": "Arkansas St.",
   "Montana State": "Montana St.", "Omaha": "Nebraska Omaha",
   "Sam Houston": "Sam Houston St.", "California Baptist": "Cal Baptist",
   "Portland State": "Portland St.", "Nicholls State": "Nicholls",
   "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
   "Illinois-Chicago": "Illinois Chicago", "Youngstown State": "Youngstown St.",
   "North Dakota State": "North Dakota St.", "Queens (NC)": "Queens",
   "Southeast Missouri State": "Southeast Missouri", "Texas State": "Texas St.",
   "Jackson State": "Jackson St.", "Appalachian State": "Appalachian St.",
   "Wright State": "Wright St.", "Indiana State": "Indiana St.",
   "Missouri State": "Missouri St.", "San Jose State": "San Jose St.",
   "Bethune-Cookman": "Bethune Cookman", "Southern Illinois-Edwardsville": "SIUE",
   "Loyola (MD)": "Loyola MD", "Norfolk State": "Norfolk St.",
   "Idaho State": "Idaho St.", "Texas-Rio Grande Valley": "UT Rio Grande Valley",
   "South Carolina State": "South Carolina St.", "Georgia State": "Georgia St.",
   "Washington State": "Washington St.", "Cleveland State": "Cleveland St.",
   "Northwestern State": "Northwestern St.", "Albany (NY)": "Albany",
   "Virginia Military Institute": "VMI", "Maryland-Baltimore County": "UMBC",
   "Pennsylvania": "Penn", "Long Island University": "LIU",
   "Tennessee-Martin": "Tennessee Martin", "Tennessee State": "Tennessee St.",
   "Central Connecticut State": "Central Connecticut", "Weber State": "Weber St.",
   "Tarleton State": "Tarleton St." , "Morgan State": "Morgan St.",
   "Morehead State": "Morehead St.", "Fresno State": "Fresno St.",
   "Cal State Bakersfield": "Cal St. Bakersfield", "Ball State": "Ball St.",
   "Alabama State": "Alabama St.", "Sacramento State": "Sacramento St.",
   "Long Beach State": "Long Beach St.", "Massachusetts-Lowell": "UMass Lowell",
   "South Carolina Upstate": "USC Upstate", "Florida International": "FIU",
   "Southern Miss.": "Southern Miss.", "Gardner-Webb": "Gardner Webb",
   "Cal State Fullerton": "Cal St. Fullerton", "Coppin State": "Coppin St.",
   "Maryland-Eastern Shore": "Maryland Eastern Shore",
   "Saint Francis (PA)": "Saint Francis", "FDU": "Fairleigh Dickinson",
   "Grambling": "Grambling St.", "Alcorn State": "Alcorn St.",
   "Delaware State": "Delaware St.", "Chicago State": "Chicago St.",
   "Louisiana-Monroe": "Louisiana Monroe", "Prairie View": "Prairie View A&M",
   "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
   "Mississippi Valley State": "Mississippi Valley St."
}


# -----------------------------
# Global modeling constants
# -----------------------------
AVG_EFF   = 106.0
AVG_TEMPO = 68.5

HFA_VAL = 3.2          # home court points (your site value)

PYTH_EXP = 11.5        # your site

# Uncertainty model (tune these)
BASE_MARGIN_SD = 11.0  # points
BASE_TOTAL_SD  = 15.0  # points
TEMPO_SD       = 2.3   # possessions

# Strength evolution (tune these)
K_ADJEM  = 0.65        # how much AdjEM moves per 1-sigma surprise (z)
K_RSM    = 0.45        # resume update scale
ADJEM_CLAMP_PER_GAME = 1.5  # max absolute AdjEM change per game

# Selection model weights (tune these)
W_PRED  = 3.20   # predictive strength weight (AdjEM)
W_RES   = 7.50   # resume strength weight (resume_score)
W_STAB  = 0.10   # mild stability toward preseason (optional)

# Selection weight variance (simulates committee opinion differences)
W_PRED_VARIANCE = 0.30   # standard deviation for W_PRED sampling
W_RES_VARIANCE  = 0.70   # standard deviation for W_RES sampling


# -----------------------------
# Utilities
# -----------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def safe_num(x, fb=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else fb
    except Exception:
        return fb


def today_local_date() -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.now().date())


def normalize_names(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).replace(NAME_MAPPING)
    return out


def detect_conf_col(df_conf: pd.DataFrame) -> str:
    candidates = ["Conference", "conference", "Conf", "conf"]
    for c in candidates:
        if c in df_conf.columns:
            return c
    raise KeyError(
        f"Conference column not found in {list(df_conf.columns)}. "
        f"Expected one of: {candidates}"
    )


def infer_adj_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    cand_O = ["AdjO", "ADJO", "adj_o", "AdjOE", "AdjOff"]
    cand_D = ["AdjD", "ADJD", "adj_d", "AdjDE", "AdjDef"]
    cand_T = ["AdjT", "ADJT", "adj_t", "Tempo", "AdjTempo"]

    def find(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    o = find(cand_O)
    d = find(cand_D)
    t = find(cand_T)
    if not (o and d and t):
        raise KeyError(
            f"Could not find AdjO/AdjD/AdjT columns. Found: "
            f"AdjO={o}, AdjD={d}, AdjT={t}. Columns={list(df.columns)}"
        )
    return o, d, t


def get_base_adjem(df: pd.DataFrame) -> pd.Series:
    if "Current_AdjEM" in df.columns:
        return pd.to_numeric(df["Current_AdjEM"], errors="coerce")
    if "AdjEM" in df.columns:
        return pd.to_numeric(df["AdjEM"], errors="coerce")
    o, d, _t = infer_adj_cols(df)
    return pd.to_numeric(df[o], errors="coerce") - pd.to_numeric(df[d], errors="coerce")


def current_rank_map(states: dict[str, "TeamState"]) -> dict[str, int]:
    """
    Rank 1 = best AdjEM. Deterministic tie-break: team name.
    """
    items = [(t, st.adjem) for t, st in states.items()]
    items.sort(key=lambda x: (-x[1], x[0]))
    return {t: i + 1 for i, (t, _em) in enumerate(items)}


# -----------------------------
# Quadrant approximation (resume)
# -----------------------------
def quadrant_from_opp_rank(opp_rank: int, loc_val: int, neutral: bool) -> int:
    if neutral:
        t1, t2, t3 = 50, 100, 200
    else:
        if loc_val == 1:      # home
            t1, t2, t3 = 30, 75, 160
        elif loc_val == -1:   # away
            t1, t2, t3 = 75, 135, 240
        else:
            t1, t2, t3 = 50, 100, 200

    if opp_rank <= t1: return 1
    if opp_rank <= t2: return 2
    if opp_rank <= t3: return 3
    return 4


def loc_multiplier(loc_val: int, neutral: bool) -> float:
    if neutral:
        return 1.00
    if loc_val == -1:
        return 1.10
    if loc_val == 1:
        return 0.95
    return 1.00


def opp_quality_weight(opp_adjem: float, field_adjem_mean: float, field_adjem_sd: float) -> float:
    if field_adjem_sd <= 1e-6:
        return 1.0
    z = (opp_adjem - field_adjem_mean) / field_adjem_sd
    return float(np.clip(1.0 + 0.15 * z, 0.70, 1.30))


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class TeamState:
    team: str
    conf: str
    base_adjo: float
    base_adjd: float
    base_adjt: float
    base_adjem: float

    d_adjo: float = 0.0
    d_adjd: float = 0.0
    d_adjt: float = 0.0

    resume: float = 0.0
    q_w: dict = None
    q_l: dict = None

    conf_w: int = 0
    conf_l: int = 0

    def __post_init__(self):
        if self.q_w is None:
            self.q_w = {1: 0, 2: 0, 3: 0, 4: 0}
        if self.q_l is None:
            self.q_l = {1: 0, 2: 0, 3: 0, 4: 0}

    @property
    def adjo(self) -> float:
        return self.base_adjo + self.d_adjo

    @property
    def adjd(self) -> float:
        return self.base_adjd + self.d_adjd

    @property
    def adjt(self) -> float:
        return self.base_adjt + self.d_adjt

    @property
    def adjem(self) -> float:
        return self.adjo - self.adjd

    def apply_adjem_update(self, delta_em: float):
        delta_em = float(np.clip(delta_em, -ADJEM_CLAMP_PER_GAME, ADJEM_CLAMP_PER_GAME))
        self.d_adjo += 0.5 * delta_em
        self.d_adjd -= 0.5 * delta_em


# -----------------------------
# Core model: expected + simulate game
# -----------------------------
def expected_game(home: TeamState, away: TeamState, loc_val_home: int, neutral: bool):
    hfa = 0.0
    if not neutral:
        if loc_val_home == 1:
            hfa = HFA_VAL
        elif loc_val_home == -1:
            hfa = -HFA_VAL

    exp_tempo = home.adjt + away.adjt - AVG_TEMPO
    exp_eff_home = home.adjo + away.adjd - AVG_EFF + hfa
    exp_eff_away = away.adjo + home.adjd - AVG_EFF - hfa

    score_home = (exp_eff_home * exp_tempo) / 100.0
    score_away = (exp_eff_away * exp_tempo) / 100.0
    margin = score_home - score_away
    total = score_home + score_away

    sh = max(score_home, 1e-6)
    sa = max(score_away, 1e-6)
    wp_home = float((sh ** PYTH_EXP) / ((sh ** PYTH_EXP) + (sa ** PYTH_EXP)))

    return exp_tempo, score_home, score_away, margin, total, wp_home


def simulate_game(
    rng: np.random.Generator,
    home: TeamState,
    away: TeamState,
    loc_val_home: int,
    neutral: bool
):
    exp_tempo, sH_mean, sA_mean, margin_mean, total_mean, wp_home = expected_game(home, away, loc_val_home, neutral)

    tempo_obs = rng.normal(exp_tempo, TEMPO_SD)
    tempo_ratio = tempo_obs / max(exp_tempo, 1e-6)
    sH_mean_t = sH_mean * tempo_ratio
    sA_mean_t = sA_mean * tempo_ratio
    margin_mean_t = sH_mean_t - sA_mean_t
    total_mean_t  = sH_mean_t + sA_mean_t

    margin_sd = BASE_MARGIN_SD * (0.90 + 0.15 * abs(tempo_obs - AVG_TEMPO) / AVG_TEMPO)
    total_sd  = BASE_TOTAL_SD  * (0.95 + 0.20 * abs(tempo_obs - AVG_TEMPO) / AVG_TEMPO)

    margin_obs = rng.normal(margin_mean_t, margin_sd)
    total_obs  = rng.normal(total_mean_t, total_sd)

    score_home = 0.5 * (total_obs + margin_obs)
    score_away = 0.5 * (total_obs - margin_obs)

    score_home = int(round(max(0.0, score_home)))
    score_away = int(round(max(0.0, score_away)))

    margin_real = float(score_home - score_away)
    total_real  = float(score_home + score_away)

    return {
        "exp_margin": float(margin_mean_t),
        "exp_total": float(total_mean_t),
        "exp_wp_home": float(wp_home),
        "margin_sd": float(margin_sd),
        "total_sd": float(total_sd),
        "tempo_exp": float(exp_tempo),
        "tempo_obs": float(tempo_obs),
        "score_home": int(score_home),
        "score_away": int(score_away),
        "margin_obs": float(margin_real),
        "total_obs": float(total_real),
        "home_win": int(score_home > score_away),
        "away_win": int(score_away > score_home),
        "tie": int(score_home == score_away),
    }


# -----------------------------
# Build team states + schedule
# -----------------------------
def load_inputs(ratings_file: str, conferences_file: str, schedule_file: str):
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(ratings_file)
    if not os.path.exists(conferences_file):
        raise FileNotFoundError(conferences_file)
    if not os.path.exists(schedule_file):
        raise FileNotFoundError(schedule_file)

    df_raw = pd.read_csv(ratings_file)
    df_conf = pd.read_csv(conferences_file)

    if "Team" not in df_raw.columns:
        raise KeyError(f"ratings file missing 'Team' column: {list(df_raw.columns)}")
    df_raw["Team"] = df_raw["Team"].astype(str).replace(NAME_MAPPING)

    if "Team" not in df_conf.columns:
        raise KeyError(f"conferences file missing 'Team' column: {list(df_conf.columns)}")
    df_conf["Team"] = df_conf["Team"].astype(str).replace(NAME_MAPPING)

    conf_col = detect_conf_col(df_conf)
    df_conf = df_conf.rename(columns={conf_col: "Conference"})
    df_conf["Conference"] = df_conf["Conference"].astype(str)

    ocol, dcol, tcol = infer_adj_cols(df_raw)
    df_raw[ocol] = pd.to_numeric(df_raw[ocol], errors="coerce")
    df_raw[dcol] = pd.to_numeric(df_raw[dcol], errors="coerce")
    df_raw[tcol] = pd.to_numeric(df_raw[tcol], errors="coerce")

    base_adjem = get_base_adjem(df_raw)
    df_raw["AdjEM_BASE"] = pd.to_numeric(base_adjem, errors="coerce")

    df = df_raw.merge(df_conf[["Team", "Conference"]], on="Team", how="left")
    df["Conference"] = df["Conference"].fillna("Unknown")

    if "RQ" in df.columns:
        df["RQ_BASE"] = pd.to_numeric(df["RQ"], errors="coerce").fillna(0.0)
    else:
        df["RQ_BASE"] = 0.0

    sch = pd.read_csv(schedule_file)
    sch = normalize_names(sch, ["team", "opponent"])
    sch["date"] = pd.to_datetime(sch["date"], errors="coerce")
    sch = sch.dropna(subset=["date", "team", "opponent"])

    if "location_value" not in sch.columns:
        sch["location_value"] = 0
    sch["location_value"] = pd.to_numeric(sch["location_value"], errors="coerce").fillna(0).astype(int)

    if "neutral" not in sch.columns:
        sch["neutral"] = False
    sch["neutral"] = sch["neutral"].apply(
        lambda x: bool(int(x)) if str(x).isdigit() else str(x).strip().lower() in ["true", "t", "yes", "y"]
    )

    teams_set = set(df["Team"].tolist())
    sch = sch[sch["team"].isin(teams_set) & sch["opponent"].isin(teams_set)].copy()

    sch = sch[sch["date"] >= today_local_date()].copy()
    sch = sch.sort_values(["date", "team", "opponent"]).reset_index(drop=True)

    return df, sch, (ocol, dcol, tcol)


def init_team_states(df: pd.DataFrame, ocol: str, dcol: str, tcol: str) -> dict[str, TeamState]:
    states: dict[str, TeamState] = {}
    for _, r in df.iterrows():
        team = str(r["Team"])
        conf = str(r["Conference"]) if pd.notna(r["Conference"]) else "Unknown"
        base_adjo = safe_num(r[ocol], 0.0)
        base_adjd = safe_num(r[dcol], 0.0)
        base_adjt = safe_num(r[tcol], AVG_TEMPO)
        base_adjem = safe_num(r["AdjEM_BASE"], base_adjo - base_adjd)

        st = TeamState(
            team=team,
            conf=conf,
            base_adjo=base_adjo,
            base_adjd=base_adjd,
            base_adjt=base_adjt,
            base_adjem=base_adjem,
            resume=safe_num(r.get("RQ_BASE", 0.0), 0.0),
        )
        states[team] = st
    return states


# -----------------------------
# Tournament selection + seeding
# -----------------------------
def selection_score(st: TeamState, w_pred: float = W_PRED, w_res: float = W_RES) -> float:
    return (w_pred * st.adjem) + (w_res * st.resume)


def seed_tournament(states: dict[str, TeamState], w_pred: float = W_PRED, w_res: float = W_RES):
    """
    Returns:
      field_df     : 68-team field with seeds + Bid_Type
      aq_list      : list of AQ teams
      selection68  : "committee-style top 68 ignoring AQ status"
    """
    conf_to_teams = defaultdict(list)
    for _t, st in states.items():
        conf_to_teams[st.conf].append(st)

    aq_list: list[str] = []
    for conf, lst in conf_to_teams.items():
        if conf == "Unknown":
            continue

        def keyfun(s: TeamState):
            g = s.conf_w + s.conf_l
            wp = (s.conf_w / g) if g > 0 else 0.0
            return (s.conf_w, wp, selection_score(s, w_pred, w_res), s.adjem)

        top = sorted(lst, key=keyfun, reverse=True)[0]
        aq_list.append(top.team)

    all_states = list(states.values())
    all_states.sort(key=lambda s: selection_score(s, w_pred, w_res), reverse=True)

    # committee-style selection ignoring AQ
    selection68 = [s.team for s in all_states[:68]]

    aq_set = set(aq_list)
    auto_bids = [states[t] for t in aq_list if t in states]

    at_large_pool = [s for s in all_states if s.team not in aq_set]
    num_at_large = max(0, 68 - len(auto_bids))
    at_large_bids = at_large_pool[:num_at_large]

    field = auto_bids + at_large_bids
    field.sort(key=lambda s: selection_score(s, w_pred, w_res), reverse=True)

    seeds = []
    for s in range(1, 11):
        seeds.extend([s] * 4)
    seeds.extend([11] * 6)
    for s in range(12, 16):
        seeds.extend([s] * 4)
    seeds.extend([16] * 6)
    if len(field) > len(seeds):
        seeds.extend([16] * (len(field) - len(seeds)))

    rows = []
    for i, st in enumerate(field):
        rows.append({
            "Team": st.team,
            "Conference": st.conf,
            "AdjEM_SIM": st.adjem,
            "Resume_SIM": st.resume,
            "Selection_Score": selection_score(st, w_pred, w_res),
            "Conf_W": st.conf_w,
            "Conf_L": st.conf_l,
            "Seed": int(seeds[i]),
            "Bid_Type": "Auto" if st.team in aq_set else "At-Large",
        })

    return pd.DataFrame(rows), aq_list, selection68


# -----------------------------
# Simulation loop: season evolution
# -----------------------------
def run_one_sim(
    rng: np.random.Generator,
    base_states: dict[str, TeamState],
    schedule: pd.DataFrame,
    preseason_anchor: dict[str, float] | None = None,
    w_pred: float = W_PRED,
    w_res: float = W_RES,
):
    """
    Returns:
      - final_states (team->TeamState)
      - field_df (68-team field, with seeds)
      - aq_list
      - selection68 (top 68 ignoring AQ)
    """
    states: dict[str, TeamState] = {}
    for t, s in base_states.items():
        states[t] = TeamState(
            team=s.team, conf=s.conf,
            base_adjo=s.base_adjo, base_adjd=s.base_adjd, base_adjt=s.base_adjt, base_adjem=s.base_adjem,
            d_adjo=s.d_adjo, d_adjd=s.d_adjd, d_adjt=s.d_adjt,
            resume=s.resume,
            conf_w=s.conf_w, conf_l=s.conf_l,
            q_w=dict(s.q_w), q_l=dict(s.q_l),
        )

    adjem0 = np.array([st.adjem for st in states.values()], dtype=float)
    mu0 = float(np.nanmean(adjem0))
    sd0 = float(np.nanstd(adjem0) if np.nanstd(adjem0) > 1e-6 else 10.0)

    # For quadrant approximation ranks: compute occasionally (not every game).
    # You can tune this cadence; 1 recompute per game is accurate but costly.
    # Here we recompute each game for correctness/simplicity.
    for _, g in schedule.iterrows():
        team = str(g["team"])
        opp  = str(g["opponent"])
        loc_val = int(g["location_value"])
        neutral = bool(g["neutral"])

        if team not in states or opp not in states:
            continue

        # Determine home/away roles based on location_value from 'team' perspective
        if neutral:
            home_name, away_name = team, opp
            loc_home = 0
        else:
            if loc_val == 1:
                home_name, away_name = team, opp
                loc_home = 1
            elif loc_val == -1:
                home_name, away_name = opp, team
                loc_home = 1
            else:
                home_name, away_name = team, opp
                loc_home = 0

        home = states[home_name]
        away = states[away_name]

        sim = simulate_game(rng, home, away, loc_home, neutral)

        # Translate result into 'team' perspective quantities for updates
        if team == home_name:
            team_win = sim["home_win"]
            opp_win  = sim["away_win"]
            exp_wp_team = sim["exp_wp_home"]
            margin_obs_team = sim["margin_obs"]
            exp_margin_team = sim["exp_margin"]
            margin_sd = sim["margin_sd"]
        else:
            team_win = sim["away_win"]
            opp_win  = sim["home_win"]
            exp_wp_team = 1.0 - sim["exp_wp_home"]
            margin_obs_team = -sim["margin_obs"]
            exp_margin_team = -sim["exp_margin"]
            margin_sd = sim["margin_sd"]

        conf_team = states[team].conf
        conf_opp  = states[opp].conf
        is_conf_game = (conf_team == conf_opp) and (conf_team != "Unknown")

        if is_conf_game:
            if team_win == 1:
                states[team].conf_w += 1
                states[opp].conf_l += 1
            elif opp_win == 1:
                states[team].conf_l += 1
                states[opp].conf_w += 1

        # Strength evolution (AdjEM) from surprise
        surprise = margin_obs_team - exp_margin_team
        z = float(surprise / max(margin_sd, 1e-6))
        delta_em = K_ADJEM * z

        if preseason_anchor is not None and team in preseason_anchor:
            pull = preseason_anchor[team] - states[team].adjem
            delta_em += W_STAB * (pull / 20.0)

        states[team].apply_adjem_update(delta_em)
        states[opp].apply_adjem_update(-delta_em * 0.40)

        # Resume evolution (result - expected WP), opponent-quality + location weighting
        oqw = opp_quality_weight(states[opp].adjem, mu0, sd0)
        lmw = loc_multiplier(loc_val, neutral)

        result = 1.0 if team_win == 1 else (0.0 if opp_win == 1 else 0.5)
        delta_res = K_RSM * (result - exp_wp_team) * oqw * lmw

        states[team].resume += delta_res
        states[opp].resume -= delta_res * 0.75

        # Quadrant bookkeeping (approx) using current ranks
        rank_map = current_rank_map(states)
        opp_rank = rank_map.get(opp, 999)
        q = quadrant_from_opp_rank(opp_rank, loc_val, neutral)

        if team_win == 1:
            states[team].q_w[q] += 1
            states[opp].q_l[q] += 1
        elif opp_win == 1:
            states[team].q_l[q] += 1
            states[opp].q_w[q] += 1

    field_df, aq_list, selection68 = seed_tournament(states, w_pred, w_res)
    return states, field_df, aq_list, selection68


# -----------------------------
# Aggregation helpers
# -----------------------------
def seed_odds_from_counts(seed_counts: dict[str, Counter], total_sims: int):
    rows = []
    for team, ctr in seed_counts.items():
        for seed, c in ctr.items():
            rows.append({"Team": team, "Seed": int(seed), "Prob": float(c) / float(total_sims)})
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame()

    wide = (
        long.pivot_table(index="Team", columns="Seed", values="Prob", aggfunc="sum", fill_value=0.0)
            .sort_index(axis=1)
            .reset_index()
    )

    for s in range(1, 17):
        if s not in wide.columns:
            wide[s] = 0.0
    wide = wide[["Team"] + list(range(1, 17))]
    return long.sort_values(["Team", "Seed"]), wide


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_jsonl_line(fp, obj: dict):
    fp.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="cbb_ratings.csv")
    ap.add_argument("--confs", default="cbb_conferences.csv")
    ap.add_argument("--schedule", default="schedule.csv")
    ap.add_argument("--outdir", default="mc_out")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)

    # Selection weight variance
    ap.add_argument("--weight-variance", dest="weight_variance", action="store_true", default=True,
                    help="Enable selection weight variance to simulate committee differences (default: on)")
    ap.add_argument("--no-weight-variance", dest="weight_variance", action="store_false",
                    help="Disable selection weight variance (use fixed weights)")
    ap.add_argument("--w-pred-var", type=float, default=W_PRED_VARIANCE,
                    help=f"Standard deviation for predictive weight sampling (default: {W_PRED_VARIANCE})")
    ap.add_argument("--w-res-var", type=float, default=W_RES_VARIANCE,
                    help=f"Standard deviation for resume weight sampling (default: {W_RES_VARIANCE})")

    # JSONL logging
    ap.add_argument("--log-jsonl", dest="log_jsonl", action="store_true", default=True,
                    help="Write one JSON object per sim to mc_seasons.jsonl (default: on)")
    ap.add_argument("--no-log-jsonl", dest="log_jsonl", action="store_false",
                    help="Disable JSONL season logging")
    ap.add_argument("--jsonl-name", default="mc_seasons.jsonl",
                    help="Filename for JSONL log inside outdir")

    args = ap.parse_args()

    ensure_dir(args.outdir)

    df, sch, (ocol, dcol, tcol) = load_inputs(args.ratings, args.confs, args.schedule)
    base_states = init_team_states(df, ocol, dcol, tcol)

    preseason_anchor = {t: st.adjem for t, st in base_states.items()}
    rng = np.random.default_rng(args.seed)

    seed_counts = defaultdict(Counter)
    field_counts = Counter()
    atlarge_counts = Counter()
    aq_counts = Counter()
    noaq_counts = Counter()
    atlarge_given_noaq_counts = Counter()


    # NEW: "committee style" selection ignoring AQ
    selection_in_counts = Counter()          # in top 68 by selection score, regardless of AQ
    selection_noaq_counts = Counter()        # in top 68 by selection score AND did NOT win AQ

    jsonl_path = os.path.join(args.outdir, args.jsonl_name)
    jsonl_fp = open(jsonl_path, "w", encoding="utf-8") if args.log_jsonl else None

    if sch.empty:
        print("WARNING: schedule is empty after filtering to future games and known teams.")
        print("You will still get deterministic seeds from current ratings/resume baseline.")
    else:
        print(f"Loaded {len(sch)} future games. Simulating {args.sims} seasons...")
    
    if args.weight_variance:
        print(f"Selection weight variance enabled: W_PRED ~ N({W_PRED:.2f}, {args.w_pred_var:.2f}), W_RES ~ N({W_RES:.2f}, {args.w_res_var:.2f})")
    else:
        print(f"Selection weight variance disabled: using fixed W_PRED={W_PRED:.2f}, W_RES={W_RES:.2f}")
    
    # Time estimation variables
    start_time = time.time()
    checkpoint_interval = max(1, args.sims // 100)  # Update every 1% of progress
    
    sims = 0
    for sim in range(args.sims):
        # Sample selection weights for this simulation to simulate committee variability
        # Weights are sampled from normal distribution centered at base weights
        if args.weight_variance:
            sim_w_pred = max(0.1, rng.normal(W_PRED, args.w_pred_var))  # ensure positive
            sim_w_res = max(0.1, rng.normal(W_RES, args.w_res_var))    # ensure positive
        else:
            sim_w_pred = W_PRED
            sim_w_res = W_RES
        
        # Progress and time estimation
        if sim > 0 and sim % checkpoint_interval == 0:
            elapsed = time.time() - start_time
            rate = sim / elapsed  # sims per second
            remaining_sims = args.sims - sim
            eta_seconds = remaining_sims / rate if rate > 0 else 0
            
            # Format ETA
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"
            
            pct = 100.0 * sim / args.sims
            print(f" Progress: {sim}/{args.sims} ({pct:.1f}%) | ETA: {eta_str}        ", end="\r", flush=True)
        elif sim == 0:
            print(f" Simulating season {sim + 1} / {args.sims}...", end="\r", flush=True)
        
        _states, field_df, aq_list, selection68 = run_one_sim(
            rng, base_states, sch, preseason_anchor=preseason_anchor,
            w_pred=sim_w_pred, w_res=sim_w_res
        )

        aq_set = set(aq_list)
        sel_set = set(selection68)

        # record AQs
        for t in aq_list:
            aq_counts[t] += 1

        # record selection top 68 (ignoring AQ)
        for t in sel_set:
            selection_in_counts[t] += 1
            if t not in aq_set:
                selection_noaq_counts[t] += 1

        if field_df is None or field_df.empty:
            # still log selection + AQ if you want
            if jsonl_fp is not None:
                write_jsonl_line(jsonl_fp, {
                    "type": "season",
                    "sim": int(sim),
                    "timestamp_utc": utc_now_iso(),
                    "aq": list(aq_list),
                    "selection68": list(selection68),
                    "field": [],
                })
            continue
        
        aq_set = set(aq_list)

# everyone not in aq_set contributes to the conditional denominator
        for team in base_states.keys():
            if team not in aq_set:
                noaq_counts[team] += 1

        # count conditional numerator: at-large teams that are not AQ (should always be true if Bid_Type correct)
        for _, r in field_df.iterrows():
            team = r["Team"]
            if team not in aq_set and r["Bid_Type"] == "At-Large":
                atlarge_given_noaq_counts[team] += 1


        # record field / at-large / seeds
        field_records = []
        for _, r in field_df.iterrows():
            team = str(r["Team"])
            seed = int(r["Seed"])
            bid = str(r["Bid_Type"])

            seed_counts[team][seed] += 1
            field_counts[team] += 1
            if bid == "At-Large":
                atlarge_counts[team] += 1

            field_records.append({
                "Team": team,
                "Seed": seed,
                "Bid_Type": bid,
                "Conference": str(r.get("Conference", "")),
                "Selection_Score": float(r.get("Selection_Score", 0.0)),
                "AdjEM_SIM": float(r.get("AdjEM_SIM", 0.0)),
                "Resume_SIM": float(r.get("Resume_SIM", 0.0)),
                "Conf_W": int(r.get("Conf_W", 0)),
                "Conf_L": int(r.get("Conf_L", 0)),
            })

        # JSONL logging for this simulated season
        if jsonl_fp is not None:
            write_jsonl_line(jsonl_fp, {
                "type": "season",
                "sim": int(sim),
                "timestamp_utc": utc_now_iso(),
                "aq": list(aq_list),
                "selection68": list(selection68),
                "field": field_records,
            })

    # Print final timing information
    print()  # New line after progress
    total_time = time.time() - start_time
    if total_time < 60:
        time_str = f"{total_time:.1f} seconds"
    elif total_time < 3600:
        time_str = f"{total_time/60:.1f} minutes"
    else:
        time_str = f"{total_time/3600:.2f} hours"
    
    avg_time_per_sim = total_time / args.sims if args.sims > 0 else 0
    print(f"Completed {args.sims} simulations in {time_str} ({avg_time_per_sim:.3f}s per simulation)")

    if jsonl_fp is not None:
        jsonl_fp.close()

    total_sims = args.sims

    # outputs: seed odds
    seed_long, seed_wide = seed_odds_from_counts(seed_counts, total_sims)
    seed_long.to_csv(os.path.join(args.outdir, "mc_seed_odds_long.csv"), index=False)
    seed_wide.to_csv(os.path.join(args.outdir, "mc_seed_odds_wide.csv"), index=False)

    # field odds table (expanded)
    teams = sorted(set(
        list(field_counts.keys())
        + list(aq_counts.keys())
        + list(atlarge_counts.keys())
        + list(selection_in_counts.keys())
        + list(selection_noaq_counts.keys())
    ))
    rows = []
    den = noaq_counts[t]
    for t in teams:
        den = noaq_counts[t]
        rows.append({
            "Team": t,
            "MakeField_Prob": field_counts[t] / total_sims,
            "AtLarge_Prob": atlarge_counts[t] / total_sims,
            "AutoBid_Prob": aq_counts[t] / total_sims,
            "NoAQ_Prob": den / total_sims,
            "AtLargeIfNoAQ_Prob": (atlarge_given_noaq_counts[t] / den) if den > 0 else 0.0,
            "Selection68_Prob": selection_in_counts[t] / total_sims,
            "Selection68_NoAQ_Prob": selection_noaq_counts[t] / total_sims,
        })

            # NEW
            

    df_field = pd.DataFrame(rows).sort_values(
        ["MakeField_Prob", "Selection68_Prob"], ascending=[False, False]
    )
    df_field.to_csv(os.path.join(args.outdir, "mc_field_odds.csv"), index=False)

    print("Done.")
    print(f"Wrote: {os.path.join(args.outdir, 'mc_seed_odds_wide.csv')}")
    print(f"Wrote: {os.path.join(args.outdir, 'mc_seed_odds_long.csv')}")
    print(f"Wrote: {os.path.join(args.outdir, 'mc_field_odds.csv')}")
    if args.log_jsonl:
        print(f"Wrote: {jsonl_path}")


if __name__ == "__main__":
    main()