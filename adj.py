import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.sparse import lil_matrix
from scipy.stats import norm
from datetime import datetime

# ================= CONFIG =================
BASE_PRESEASON_GAMES = 4     # baseline preseason strength
SURPRISE_SCALE = 6           # AdjEM std dev
SURPRISE_CUTOFF = 1.75       # z-score where collapse accelerates
SURPRISE_STEEPNESS = 1.5     # how sharp the cutoff is

# Prediction Constants
AVG_EFF = 106.0
AVG_TEMPO = 68.5
HFA_VAL = 3.2
PYTH_EXP = 11.5
# ==========================================

# --- NAME MAPPING ---
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

# ============================================================
# NEW: MANUAL NON-D1 RESULTS (inject into df before resume metrics)
# ============================================================
# Fill in the REAL opponent name/date/score if you want to display it later.
# opponent does NOT need to exist in D1 ratings. We'll treat it as "slightly worse than worst D1"
# via opp_adjem = worst_D1_Blended_AdjEM - 2.0 by default (if opp_adjem is None).
MANUAL_NON_D1_GAMES = [
    {
        "team": "Boise St.",
        "opponent": "Hawaii Pacific",
        "date": "2025-11-03",
        "location_value": 1,     # Boise home
        "possessions": 67.0,
        "mp": 200,
        "pts": 78,
        "opp_pts": 79,
        "opp_adjem": None,       # None => will auto-set to (worst D1 - 2.0)
    }
]

def inject_manual_non_d1_games(df: pd.DataFrame,
                               ratings: pd.DataFrame,
                               pts_col: str,
                               manual_games: list[dict]) -> pd.DataFrame:
    """
    Append synthetic game rows (team row + mirrored opponent row) so:
      - opponent-point merges work in resume functions
      - the loss counts in RQ, quadrants, SOS, etc.
      - opponent strength is treated as slightly worse than the worst D1 team.

    IMPORTANT:
      - Should be called AFTER ratings['Blended_AdjEM'] exists
      - Should be called BEFORE calculate_rq_pythag/calculate_resume_stats/etc
    """
    if not manual_games:
        return df

    df2 = df.copy()

    # Default "slightly worse than worst D1"
    worst_d1_em = None
    if "Blended_AdjEM" in ratings.columns and len(ratings) > 0:
        worst_d1_em = float(pd.to_numeric(ratings["Blended_AdjEM"], errors="coerce").min())
    default_opp_adjem = (worst_d1_em - 2.0) if (worst_d1_em is not None and np.isfinite(worst_d1_em)) else -12.0

    rows = []
    for g in manual_games:
        team = str(g["team"])
        team = NAME_MAPPING.get(team, team)

        opp = str(g["opponent"])
        # (Optionally map if you ever put a known alias here)
        opp = NAME_MAPPING.get(opp, opp)

        date = pd.to_datetime(g["date"])
        loc  = int(g.get("location_value", 0))
        poss = float(g.get("possessions", np.nan))
        mp   = float(g.get("mp", 200))
        pts  = float(g.get("pts", np.nan))
        opp_pts = float(g.get("opp_pts", np.nan))

        if not np.isfinite(poss) or poss <= 0:
            raise ValueError(f"Manual game must include positive possessions. Got {poss} for {team} vs {opp}.")
        if not np.isfinite(pts) or not np.isfinite(opp_pts):
            raise ValueError(f"Manual game must include pts and opp_pts. Got pts={pts}, opp_pts={opp_pts} for {team} vs {opp}.")

        opp_adjem = g.get("opp_adjem", None)
        if opp_adjem is None:
            opp_adjem = default_opp_adjem
        opp_adjem = float(opp_adjem)

        # Your ridge model expects raw_off_eff to exist
        raw_off_eff_team = (pts / poss) * 100.0
        raw_off_eff_opp  = (opp_pts / poss) * 100.0

        # Team-facing row (the one that should affect Boise resume)
        rows.append({
            "date": date,
            "team": team,
            "opponent": opp,
            "location_value": loc,
            "possessions": poss,
            "mp": mp,
            pts_col: pts,
            "raw_off_eff": raw_off_eff_team,
            "_manual_non_d1": 1,
            "_manual_opp_adjem": opp_adjem,  # used by RQ fallback
        })

        # Mirrored opponent row so "merge opponent points" logic works
        rows.append({
            "date": date,
            "team": opp,
            "opponent": team,
            "location_value": (-loc if loc in (-1, 1) else 0),
            "possessions": poss,
            "mp": mp,
            pts_col: opp_pts,
            "raw_off_eff": raw_off_eff_opp,
            "_manual_non_d1": 1,
            "_manual_opp_adjem": np.nan,
        })

    add = pd.DataFrame(rows)

    # Align columns (avoid KeyErrors downstream)
    for c in df2.columns:
        if c not in add.columns:
            add[c] = np.nan
    for c in add.columns:
        if c not in df2.columns:
            df2[c] = np.nan

    add = add[df2.columns]
    df2 = pd.concat([df2, add], ignore_index=True)
    return df2


# --- BLOWOUT CONTROL ---
def cap_efficiency_margin(raw_diff, cap=25.0):
    signs = np.sign(raw_diff)
    abs_diff = np.abs(raw_diff)
    excess = np.maximum(0, abs_diff - cap)
    return signs * (np.minimum(abs_diff, cap) + 0.5 * excess)

# --- PRIOR COLLAPSE FUNCTION ---
def prior_collapse_factor(z):
    return 1 / (1 + np.exp(SURPRISE_STEEPNESS * (np.abs(z) - SURPRISE_CUTOFF)))

# ============================================================
# Venue +/- (AdjEM-based)
# ============================================================
def calculate_venue_adjem_split(
    ratings_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    pts_col: str,
    model_hfa: float,
    shrink_lambda: int = 8,
    non_d1_adjem: float = -10.0,
):
    """
    (unchanged from your version)
    """
    print("Calculating Venue +/- (AdjEM-based): Away/Neutral vs Home...")

    if 'Blended_AdjEM' not in ratings_df.columns:
        raise ValueError("Blended_AdjEM must be computed before calculate_venue_adjem_split().")

    games = scores_df.copy()
    games = games.dropna(subset=['date', 'team', 'opponent'])

    # Normalize date to day (kills timestamp duplicates)
    games['date_day'] = pd.to_datetime(games['date']).dt.normalize()

    # Canonical (order-invariant) game id
    def _pair_key(r):
        a, b = str(r['team']), str(r['opponent'])
        x, y = (a, b) if a <= b else (b, a)
        return f"{x}__{y}"

    games['pair_key'] = games.apply(_pair_key, axis=1)
    games['game_id'] = games['date_day'].astype(str) + "__" + games['pair_key']

    # Attach opponent points
    opp_pts_df = games[['game_id', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    opp_pts_df = opp_pts_df.groupby(['game_id', 'opponent'], as_index=False).first()
    games = games.merge(opp_pts_df, on=['game_id', 'opponent'], how='left')

    # Collapse to ONE row per (team, game_id) to avoid duplicates inflating venue stats
    games = games.sort_values(['team', 'game_id'])
    games = games.groupby(['team', 'game_id'], as_index=False).first()

    # EM lookup for expectation
    em_map = ratings_df.set_index('Team')['Blended_AdjEM'].to_dict()

    out_rows = []
    for team in ratings_df['Team']:
        tg = games[games['team'] == team].copy()
        if tg.empty:
            out_rows.append({
                'Team': team,
                'Venue_AdjEM_Home': 0.0,
                'Venue_AdjEM_AwayNeutral': 0.0,
                'Venue_AdjEM_AN_minus_Home': 0.0,
                'Venue_AdjEM_Home_N': 0,
                'Venue_AdjEM_AwayNeutral_N': 0,
            })
            continue

        home_resids = []
        an_resids = []

        team_em = em_map.get(team, 0.0)

        for _, row in tg.iterrows():
            opp = row['opponent']
            poss = row.get('possessions', np.nan)
            pts = row.get(pts_col, np.nan)
            opp_pts = row.get('opp_pts', np.nan)

            if pd.isna(poss) or poss == 0 or pd.isna(pts) or pd.isna(opp_pts):
                continue

            actual_margin100 = (pts - opp_pts) / poss * 100.0

            loc_val = row.get('location_value', 0)
            hfa_adj = model_hfa if loc_val == 1 else (-model_hfa if loc_val == -1 else 0.0)

            opp_em = em_map.get(opp, non_d1_adjem)
            expected_margin100 = team_em - opp_em + hfa_adj

            resid = float(actual_margin100 - expected_margin100)

            if loc_val == 1:
                home_resids.append(resid)
            else:
                an_resids.append(resid)

        home_n = len(home_resids)
        an_n = len(an_resids)

        home_mean = float(np.mean(home_resids)) if home_n > 0 else 0.0
        an_mean = float(np.mean(an_resids)) if an_n > 0 else 0.0

        w_home = home_n / (home_n + shrink_lambda) if home_n > 0 else 0.0
        w_an = an_n / (an_n + shrink_lambda) if an_n > 0 else 0.0
        home_mean_s = w_home * home_mean
        an_mean_s = w_an * an_mean

        n_eff = min(home_n, an_n)
        w_diff = n_eff / (n_eff + shrink_lambda) if n_eff > 0 else 0.0
        diff_s = w_diff * (an_mean - home_mean)

        out_rows.append({
            'Team': team,
            'Venue_AdjEM_Home': float(home_mean_s),
            'Venue_AdjEM_AwayNeutral': float(an_mean_s),
            'Venue_AdjEM_AN_minus_Home': float(diff_s),
            'Venue_AdjEM_Home_N': int(home_n),
            'Venue_AdjEM_AwayNeutral_N': int(an_n),
        })

    out_df = pd.DataFrame(out_rows)
    ratings_df = ratings_df.merge(out_df, on='Team', how='left')
    return ratings_df


# --- STRENGTH ADJUST ---
def calculate_strength_adjust(
    ratings_df,
    scores_df,
    pts_col,
    model_hfa,
    sigma0=11.0,
    opp_weight_power=2.0,
    shrink_lambda=8,
    non_d1_adjem=-10.0
):
    print("Calculating Strength Adjust (performance vs strong opponents vs weak opponents)...")

    if 'Blended_AdjEM' not in ratings_df.columns:
        raise ValueError("Blended_AdjEM must be computed before calculate_strength_adjust().")

    tmp = ratings_df[['Team', 'Blended_AdjEM']].copy()
    tmp = tmp.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    tmp['strength_pct'] = 1.0 - (tmp.index / max(1, (len(tmp) - 1)))

    strength_pct_map = tmp.set_index('Team')['strength_pct'].to_dict()
    team_adjem = ratings_df.set_index('Team')['Blended_AdjEM'].to_dict()

    opp_scores = scores_df[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    opp_scores = opp_scores.drop_duplicates(subset=['date', 'opponent'])
    games = scores_df.merge(opp_scores, on=['date', 'opponent'], how='left')
    games = games.sort_values(['team', 'date'])

    out = []
    for team in ratings_df['Team']:
        team_games = games[games['team'] == team].copy()
        if team_games.empty:
            out.append({'Team': team, 'StrengthAdjust': 0.0, 'TopPerf': 0.0, 'BadBeat': 0.0,
                        'StrengthAdjust_z': 0.0, 'StrengthAdjust_n': 0})
            continue

        top_num = top_den = bad_num = bad_den = 0.0
        n_used = 0

        for _, row in team_games.iterrows():
            opp = row['opponent']
            poss = row.get('possessions', np.nan)
            opp_pts = row.get('opp_pts', np.nan)
            pts = row.get(pts_col, np.nan)

            if pd.isna(poss) or poss == 0 or pd.isna(opp_pts) or pd.isna(pts):
                continue
            if team not in team_adjem:
                continue

            actual_margin100 = (pts - opp_pts) / poss * 100.0

            loc_val = row.get('location_value', 0)
            hfa_adj = model_hfa if loc_val == 1 else (-model_hfa if loc_val == -1 else 0.0)

            opp_em = team_adjem.get(opp, non_d1_adjem)
            expected_margin100 = team_adjem[team] - opp_em + hfa_adj
            resid = actual_margin100 - expected_margin100

            s = float(np.clip(strength_pct_map.get(opp, 0.0), 0.0, 1.0))
            w_top = (s ** opp_weight_power)
            w_bad = ((1.0 - s) ** opp_weight_power)

            top_num += w_top * resid
            top_den += w_top
            bad_num += w_bad * resid
            bad_den += w_bad
            n_used += 1

        top_perf = (top_num / top_den) if top_den > 0 else 0.0
        bad_beat = (bad_num / bad_den) if bad_den > 0 else 0.0

        strength_adjust = top_perf - bad_beat
        strength_adjust_z = strength_adjust / sigma0 if sigma0 else 0.0

        w_sh = n_used / (n_used + shrink_lambda) if n_used > 0 else 0.0
        strength_adjust *= w_sh
        top_perf *= w_sh
        bad_beat *= w_sh
        strength_adjust_z *= w_sh

        out.append({
            'Team': team,
            'StrengthAdjust': float(strength_adjust),
            'TopPerf': float(top_perf),
            'BadBeat': float(bad_beat),
            'StrengthAdjust_z': float(strength_adjust_z),
            'StrengthAdjust_n': int(n_used)
        })

    out_df = pd.DataFrame(out)
    ratings_df = ratings_df.merge(out_df, on='Team', how='left')
    return ratings_df


# --- RESUME CALCULATOR ---
def calculate_resume_stats(ratings_df, scores_df, pts_col):
    print("Calculating Resume Metrics (Quadrants, SOS, Road Record)...")

    ratings_df = ratings_df.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    ratings_df['Rank'] = ratings_df.index + 1

    rank_map = ratings_df.set_index('Team')['Rank'].to_dict()
    adjem_map = ratings_df.set_index('Team')['Blended_AdjEM'].to_dict()

    # default "slightly worse than worst D1"
    worst_d1_em = float(pd.to_numeric(ratings_df["Blended_AdjEM"], errors="coerce").min())
    non_d1_em_default = worst_d1_em - 2.0 if np.isfinite(worst_d1_em) else -12.0
    non_d1_rank_default = len(ratings_df) + 10

    opp_scores = scores_df[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    opp_scores = opp_scores.drop_duplicates(subset=['date', 'opponent'])
    games = scores_df.merge(opp_scores, on=['date', 'opponent'], how='left')

    def get_quadrant(opp_rank, location_val):
        if location_val == 1:   # Home
            if opp_rank <= 30: return 1
            if opp_rank <= 75: return 2
            if opp_rank <= 160: return 3
            return 4
        elif location_val == 0: # Neutral
            if opp_rank <= 50: return 1
            if opp_rank <= 100: return 2
            if opp_rank <= 200: return 3
            return 4
        else:                   # Away
            if opp_rank <= 75: return 1
            if opp_rank <= 135: return 2
            if opp_rank <= 240: return 3
            return 4

    resume_data = []
    for team in ratings_df['Team']:
        team_games = games[games['team'] == team].copy()

        q1_w=q1_l=q2_w=q2_l=q3_w=q3_l=q4_w=q4_l=0
        top100_w=top100_l=0
        sos_list=[]
        road_wins=road_games=0

        for _, row in team_games.iterrows():
            opp = row['opponent']

            # NEW: do NOT skip non-D1; treat as slightly worse than worst D1
            opp_rank = rank_map.get(opp, non_d1_rank_default)
            opp_adjem = adjem_map.get(opp, non_d1_em_default)
            sos_list.append(opp_adjem)

            opp_pts = row.get('opp_pts', np.nan)
            pts = row.get(pts_col, np.nan)
            if pd.isna(opp_pts) or pd.isna(pts):
                continue
            is_win = pts > opp_pts

            if row.get('location_value', 0) != 1:
                road_games += 1
                if is_win:
                    road_wins += 1

            quad = get_quadrant(opp_rank, row.get('location_value', 0))
            if quad == 1:
                if is_win: q1_w += 1
                else: q1_l += 1
            elif quad == 2:
                if is_win: q2_w += 1
                else: q2_l += 1
            elif quad == 3:
                if is_win: q3_w += 1
                else: q3_l += 1
            else:
                if is_win: q4_w += 1
                else: q4_l += 1

            if opp_rank <= 100:
                if is_win: top100_w += 1
                else: top100_l += 1

        avg_sos = sum(sos_list) / len(sos_list) if sos_list else non_d1_em_default
        road_pct = (road_wins / road_games) if road_games > 0 else 0.0

        resume_data.append({
            'Team': team,
            'SOS': round(avg_sos, 2),
            'Road_Pct': round(road_pct, 3),
            'Q1_W': q1_w, 'Q1_L': q1_l,
            'Q2_W': q2_w, 'Q2_L': q2_l,
            'Q3_W': q3_w, 'Q3_L': q3_l,
            'Q4_W': q4_w, 'Q4_L': q4_l,
            'T100_W': top100_w, 'T100_L': top100_l
        })

    resume_df = pd.DataFrame(resume_data)
    ratings_df = ratings_df.merge(resume_df, on='Team', how='left')
    return ratings_df


# --- CONSISTENCY / VOLATILITY METRIC ---
def calculate_consistency(ratings_df, scores_df, pts_col, model_hfa,
                          sigma0=11.0,
                          tau_vol=0.90,
                          tau_swing=1.20,
                          alpha=0.40,
                          shrink_lambda=8,
                          vol_prior_z=1.00,
                          swing_prior_z=1.20):
    print("Calculating Volatility-Based Consistency Scores...")

    stats_map = ratings_df.set_index('Team')[['Blended_AdjEM']].to_dict('index')

    opp_scores = scores_df[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    opp_scores = opp_scores.drop_duplicates(subset=['date', 'opponent'])
    games = scores_df.merge(opp_scores, on=['date', 'opponent'], how='left')
    games = games.sort_values(['team', 'date'])

    team_z = {}
    for _, row in games.iterrows():
        team = row['team']; opp = row['opponent']
        if team not in stats_map:
            continue

        poss = row.get('possessions', np.nan); opp_pts = row.get('opp_pts', np.nan)
        pts = row.get(pts_col, np.nan)
        if pd.isna(poss) or poss == 0 or pd.isna(opp_pts) or pd.isna(pts):
            continue

        actual_margin100 = (pts - opp_pts) / poss * 100.0

        loc_val = row.get('location_value', 0)
        hfa_adj = model_hfa if loc_val == 1 else (-model_hfa if loc_val == -1 else 0.0)

        # NEW: allow non-D1 opponent with fallback EM
        opp_em = stats_map.get(opp, {"Blended_AdjEM": -12.0})["Blended_AdjEM"]
        expected_margin100 = stats_map[team]['Blended_AdjEM'] - opp_em + hfa_adj

        resid = actual_margin100 - expected_margin100
        z = float(resid / sigma0)

        team_z.setdefault(team, []).append(z)

    cons_scores, vol_out, swing_out = [], [], []
    for team in ratings_df['Team']:
        zlist = team_z.get(team, [])
        n = len(zlist)

        if n < 2:
            vol = vol_prior_z
            swing = swing_prior_z
        else:
            zarr = np.array(zlist, dtype=float)
            vol = float(np.std(zarr - zarr.mean(), ddof=0))
            swing = float(np.mean(np.abs(np.diff(zarr))))

        w = n / (n + shrink_lambda) if n > 0 else 0.0
        vol_s = w * vol + (1.0 - w) * vol_prior_z
        swing_s = w * swing + (1.0 - w) * swing_prior_z

        vol_score = 100.0 * np.exp(-vol_s / tau_vol)
        swing_score = 100.0 * np.exp(-swing_s / tau_swing)
        score = (1.0 - alpha) * vol_score + alpha * swing_score

        cons_scores.append(float(np.clip(score, 0.0, 100.0)))
        vol_out.append(vol_s)
        swing_out.append(swing_s)

    ratings_df['Consistency'] = cons_scores
    ratings_df['Volatility_z'] = vol_out
    ratings_df['Swing_z'] = swing_out
    return ratings_df


def calculate_rq_pythag(ratings_df, scores_df, pts_col):
    print("Calculating Resume Quality (RQ) using Pythagenport (robust game-level)...")

    sorted_ratings = ratings_df.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    start_idx, end_idx = 44, 50
    if len(sorted_ratings) < 50:
        start_idx = max(0, len(sorted_ratings) - 6)
        end_idx = len(sorted_ratings)

    bubble_subset = sorted_ratings.iloc[start_idx:end_idx]
    bubble_stats = {
        'AdjO': float(bubble_subset['AdjO'].mean()),
        'AdjD': float(bubble_subset['AdjD'].mean()),
        'AdjT': float(bubble_subset['AdjT'].mean())
    }
    print(f"  Bubble Profile: AdjO {bubble_stats['AdjO']:.1f} | AdjD {bubble_stats['AdjD']:.1f} | Tempo {bubble_stats['AdjT']:.1f}")

    stats_map = ratings_df.set_index('Team')[['AdjO', 'AdjD', 'AdjT']].to_dict('index')

    games = scores_df.copy()
    games = games.dropna(subset=['date', 'team', 'opponent'])

    games['date_day'] = pd.to_datetime(games['date']).dt.normalize()

    def _pair_key(r):
        a, b = str(r['team']), str(r['opponent'])
        x, y = (a, b) if a <= b else (b, a)
        return f"{x}__{y}"

    games['pair_key'] = games.apply(_pair_key, axis=1)
    games['game_id'] = games['date_day'].astype(str) + "__" + games['pair_key']

    opp_pts_df = games[['game_id', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    opp_pts_df = opp_pts_df.groupby(['game_id', 'opponent'], as_index=False).first()

    games = games.merge(opp_pts_df, on=['game_id', 'opponent'], how='left')
    games['is_win'] = (games[pts_col] > games['opp_pts']).astype(int)

    games = games.sort_values(['team', 'game_id'])
    games = games.groupby(['team', 'game_id'], as_index=False).first()

    rq_total_map, rq_win_avg_map, rq_loss_avg_map = {}, {}, {}
    rq_wins_n_map, rq_losses_n_map = {}, {}

    for team in ratings_df['Team']:
        tg = games[games['team'] == team].copy()
        if tg.empty:
            rq_total_map[team] = 0.0
            rq_win_avg_map[team] = 0.0
            rq_loss_avg_map[team] = 0.0
            rq_wins_n_map[team] = 0
            rq_losses_n_map[team] = 0
            continue

        win_resids, loss_resids = [], []
        expected_wins_sum = 0.0
        actual_wins_sum = 0.0

        for _, row in tg.iterrows():
            opp_name = row['opponent']
            location = row.get('location_value', 0)

            # NEW: if opponent not in stats_map, allow manual non-D1 via _manual_opp_adjem
            opp = stats_map.get(opp_name, None)
            if opp is None:
                manual_opp_em = row.get("_manual_opp_adjem", np.nan)
                if pd.notna(manual_opp_em):
                    opp_em = float(manual_opp_em)
                    opp = {
                        "AdjO": AVG_EFF + 0.5 * opp_em,
                        "AdjD": AVG_EFF - 0.5 * opp_em,
                        "AdjT": AVG_TEMPO
                    }
                else:
                    opp = {'AdjO': 95.0, 'AdjD': 115.0, 'AdjT': AVG_TEMPO}

            hfa_adj = HFA_VAL if location == 1 else (-HFA_VAL if location == -1 else 0.0)

            exp_eff_bubble = bubble_stats['AdjO'] + opp['AdjD'] - AVG_EFF + hfa_adj
            exp_eff_opp    = opp['AdjO'] + bubble_stats['AdjD'] - AVG_EFF - hfa_adj
            exp_tempo = bubble_stats['AdjT'] + opp['AdjT'] - AVG_TEMPO

            score_bubble = (exp_eff_bubble * exp_tempo) / 100.0
            score_opp    = (exp_eff_opp * exp_tempo) / 100.0

            if score_bubble <= 0 or score_opp <= 0:
                p = 0.0 if score_opp > score_bubble else 1.0
            else:
                p = (score_bubble ** PYTH_EXP) / ((score_bubble ** PYTH_EXP) + (score_opp ** PYTH_EXP))

            expected_wins_sum += p
            is_win = int(row['is_win'])
            actual_wins_sum += is_win

            resid = is_win - p
            if is_win == 1:
                win_resids.append(resid)
            else:
                loss_resids.append(resid)

        # keep your Boise special-case if you still want it
        
        rq_total = actual_wins_sum - expected_wins_sum

        rq_total_map[team] = float(rq_total)
        rq_win_avg_map[team] = float(np.mean(win_resids)) if win_resids else 0.0
        rq_loss_avg_map[team] = float(np.mean(loss_resids)) if loss_resids else 0.0
        rq_wins_n_map[team] = int(len(win_resids))
        rq_losses_n_map[team] = int(len(loss_resids))

    ratings_df['RQ'] = ratings_df['Team'].map(rq_total_map).fillna(0.0)
    ratings_df['RQ_Win_Avg'] = ratings_df['Team'].map(rq_win_avg_map).fillna(0.0)
    ratings_df['RQ_Loss_Avg'] = ratings_df['Team'].map(rq_loss_avg_map).fillna(0.0)
    ratings_df['RQ_Wins_N'] = ratings_df['Team'].map(rq_wins_n_map).fillna(0).astype(int)
    ratings_df['RQ_Losses_N'] = ratings_df['Team'].map(rq_losses_n_map).fillna(0).astype(int)

    return ratings_df


# --- MAIN ENGINE ---
def calculate_ratings(input_file='cbb_scores.csv', preseason_file='pseason.csv'):
    print(f"Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Error: Scores file not found.")
        return

    pts_col = 'pts' if 'pts' in df.columns else 'points'

    # ---------- CLEAN ----------
    df['team'] = df['team'].replace(NAME_MAPPING)
    df['opponent'] = df['opponent'].replace(NAME_MAPPING)

    df['possessions'] = pd.to_numeric(df['possessions'], errors='coerce')
    df = df.dropna(subset=['possessions'])
    df = df[df['possessions'] > 0].copy()

    if 'mp' not in df.columns:
        df['mp'] = 200
    df['mp'] = pd.to_numeric(df['mp'], errors='coerce').fillna(200).replace(0, 200)

    df['date'] = pd.to_datetime(df['date'])
    most_recent = df['date'].max()
    df['days_ago'] = (most_recent - df['date']).dt.days
    decay_rate = 0.99
    weights = df['possessions'] * (decay_rate ** df['days_ago'])

    # ---------- BASELINES ----------
    global_ppp = (df[pts_col].sum() / df['possessions'].sum()) * 100.0
    df['pace_40'] = (df['possessions'] / df['mp']) * 200.0

    # raw_off_eff must exist in your file; keep your original assumption
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pace_40', 'raw_off_eff'])
    weights = df['possessions'] * (decay_rate ** df['days_ago'])
    global_pace = df['pace_40'].mean()

    if np.isnan(global_pace) or np.isnan(global_ppp):
        print("Error: global_pace or global_ppp is NaN.")
        return

    # ---------- MATRICES ----------
    teams = sorted(set(df['team']) | set(df['opponent']))
    n = len(teams)
    t2i = {t: i for i, t in enumerate(teams)}

    X_eff = lil_matrix((len(df), 2 * n + 1))
    X_pace = lil_matrix((len(df), n))

    for i, row in df.reset_index(drop=True).iterrows():
        t = t2i[row['team']]
        o = t2i[row['opponent']]
        X_eff[i, t] = 1
        X_eff[i, n + o] = -1
        X_eff[i, 2 * n] = row.get('location_value', 0)
        X_pace[i, t] = 1
        X_pace[i, o] = 1

    # ---------- TARGETS ----------
    raw_diff = df['raw_off_eff'].values - global_ppp
    y_eff = cap_efficiency_margin(raw_diff)
    y_pace = df['pace_40'].values - global_pace

    if np.isnan(y_eff).any() or np.isnan(y_pace).any():
        print("Error: NaNs detected in targets.")
        return

    # ---------- FIT ----------
    clf_eff = RidgeCV(alphas=[0.1, 1, 5, 10, 25], fit_intercept=False).fit(
        X_eff, y_eff, sample_weight=weights.values
    )
    clf_pace = RidgeCV(alphas=[0.1, 1, 5, 10], fit_intercept=False).fit(
        X_pace, y_pace, sample_weight=weights.values
    )

    coefs = clf_eff.coef_
    pace_coefs = clf_pace.coef_
    model_hfa = float(coefs[-1])

    # ---------- CURRENT RATINGS ----------
    current = []
    for team in teams:
        i = t2i[team]
        adj_o = global_ppp + coefs[i]
        adj_d = global_ppp - coefs[n + i]
        adj_em = adj_o - adj_d
        adj_t = global_pace + pace_coefs[i]
        current.append({
            "Team": team,
            "Current_AdjEM": float(adj_em),
            "AdjO": round(float(adj_o), 1),
            "AdjD": round(float(adj_d), 1),
            "AdjT": round(float(adj_t), 1),
        })

    ratings = pd.DataFrame(current)

    # ---------- GAMES PLAYED ----------
    games_played = (
        df.groupby('team').size().rename("Games").reset_index().rename(columns={"team": "Team"})
    )
    ratings = ratings.merge(games_played, on="Team", how="left")

    # ---------- PRESEASON ----------
    try:
        preseason = pd.read_csv(preseason_file)
        preseason['Team'] = preseason['Team'].replace(NAME_MAPPING)
        ratings = ratings.merge(preseason[['Team', 'Preseason_AdjEM']], on='Team', how='left')
    except FileNotFoundError:
        print("Warning: Preseason file not found. Using 0.0 defaults.")
        ratings['Preseason_AdjEM'] = 0.0

    ratings['Preseason_AdjEM'] = ratings['Preseason_AdjEM'].fillna(0.0)

    # ---------- PRIOR COLLAPSE ----------
    ratings['SurpriseZ'] = (ratings['Current_AdjEM'] - ratings['Preseason_AdjEM']) / SURPRISE_SCALE
    ratings['CollapseFactor'] = ratings['SurpriseZ'].apply(prior_collapse_factor)
    ratings['FinalPreseasonWeight'] = BASE_PRESEASON_GAMES * ratings['CollapseFactor']

    # ---------- BLENDED RATING ----------
    G = ratings['Games'].astype(float)
    w = ratings['FinalPreseasonWeight'].astype(float)
    ratings['Blended_AdjEM'] = (G / (G + w) * ratings['Current_AdjEM'] + w / (G + w) * ratings['Preseason_AdjEM'])

    # ============================================================
    # NEW: Inject manual non-D1 games AFTER Blended_AdjEM exists,
    #      BEFORE resume/consistency features are computed.
    # ============================================================
    df = inject_manual_non_d1_games(df, ratings, pts_col, MANUAL_NON_D1_GAMES)

    # --- ADVANCED CALCULATIONS ---
    ratings = calculate_rq_pythag(ratings, df, pts_col)
    ratings = calculate_consistency(ratings, df, pts_col, model_hfa)
    ratings = calculate_resume_stats(ratings, df, pts_col)
    ratings = calculate_strength_adjust(ratings, df, pts_col, model_hfa)

    # Venue +/-
    ratings = calculate_venue_adjem_split(ratings, df, pts_col, model_hfa)

    # ---------- RANK ----------
    ratings = ratings.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    ratings.index += 1
    ratings.index.name = 'Rank'

    ratings.to_csv('cbb_ratings.csv', index=True)

    print("\nTop 15 Analysis:")
    show_cols = [
        'Team', 'Blended_AdjEM', 'RQ', 'Venue_AdjEM_AN_minus_Home',
        'Venue_AdjEM_Home', 'Venue_AdjEM_AwayNeutral',
        'StrengthAdjust', 'TopPerf', 'BadBeat', 'SOS', 'Q1_W', 'Q1_L', 'T100_W'
    ]
    show_cols = [c for c in show_cols if c in ratings.columns]
    print(ratings[show_cols].head(15).round(2).to_string())

    print("\nSaved to cbb_ratings.csv")

if __name__ == "__main__":
    calculate_ratings()
