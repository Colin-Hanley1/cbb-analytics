import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.sparse import lil_matrix
from scipy.stats import norm
from datetime import datetime

# ================= CONFIG =================
BASE_PRESEASON_GAMES = 4     # baseline preseason strength
SURPRISE_SCALE = 8            # AdjEM std dev
SURPRISE_CUTOFF = 1.75            # z-score where collapse accelerates
SURPRISE_STEEPNESS = 1.5          # how sharp the cutoff is

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

# --- BLOWOUT CONTROL ---
def cap_efficiency_margin(raw_diff, cap=25.0):
    signs = np.sign(raw_diff)
    abs_diff = np.abs(raw_diff)
    excess = np.maximum(0, abs_diff - cap)
    return signs * (np.minimum(abs_diff, cap) + 0.5 * excess)

# --- PRIOR COLLAPSE FUNCTION ---
def prior_collapse_factor(z):
    return 1 / (1 + np.exp(SURPRISE_STEEPNESS * (np.abs(z) - SURPRISE_CUTOFF)))

# --- STRENGTH ADJUST (NEW) ---
def calculate_strength_adjust(
    ratings_df,
    scores_df,
    pts_col,
    model_hfa,
    sigma0=11.0,
    opp_weight_power=2.0,   # higher -> focuses more on truly elite/bottom opponents
    shrink_lambda=8,        # shrink toward 0 with small samples
    non_d1_adjem=-10.0      # fallback if opponent missing
):
    """
    StrengthAdjust:
      - Rewards overperformance vs strong opponents (TopPerf)
      - Penalizes "beating up on bad teams" (BadBeat)
      - StrengthAdjust = TopPerf - BadBeat

    residual margin/100 possessions:
      resid = actual_margin100 - expected_margin100

    Opponent strength weights are based on opponent Blended_AdjEM percentile (0..1):
      w_top = (strength_pct)^p
      w_bad = (1 - strength_pct)^p
    """

    print("Calculating Strength Adjust (performance vs strong opponents vs weak opponents)...")

    if 'Blended_AdjEM' not in ratings_df.columns:
        raise ValueError("Blended_AdjEM must be computed before calculate_strength_adjust().")

    # --- Build opponent lookup based on Blended_AdjEM ---
    tmp = ratings_df[['Team', 'Blended_AdjEM']].copy()
    tmp = tmp.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    tmp['strength_pct'] = 1.0 - (tmp.index / max(1, (len(tmp) - 1)))  # best ~1.0, worst ~0.0

    strength_pct_map = tmp.set_index('Team')['strength_pct'].to_dict()

    # Team strength for expected margin
    team_adjem = ratings_df.set_index('Team')['Blended_AdjEM'].to_dict()

    # --- Pair each row with opponent score on same date/opponent ---
    opp_scores = scores_df[['date', 'team', pts_col]].rename(
        columns={'team': 'opponent', pts_col: 'opp_pts'}
    )
    opp_scores = opp_scores.drop_duplicates(subset=['date', 'opponent'])
    games = scores_df.merge(opp_scores, on=['date', 'opponent'], how='left')

    # Ensure sorted for stability
    games = games.sort_values(['team', 'date'])

    out = []
    for team in ratings_df['Team']:
        team_games = games[games['team'] == team].copy()
        if team_games.empty:
            out.append({
                'Team': team,
                'StrengthAdjust': 0.0,
                'TopPerf': 0.0,
                'BadBeat': 0.0,
                'StrengthAdjust_z': 0.0,
                'StrengthAdjust_n': 0
            })
            continue

        top_num = 0.0
        top_den = 0.0
        bad_num = 0.0
        bad_den = 0.0

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

            # Actual margin per 100 possessions
            actual_margin100 = (pts - opp_pts) / poss * 100.0

            # Expected margin per 100 possessions from Blended AdjEM + HFA
            loc_val = row.get('location_value', 0)
            hfa_adj = model_hfa if loc_val == 1 else (-model_hfa if loc_val == -1 else 0.0)

            opp_em = team_adjem.get(opp, non_d1_adjem)
            expected_margin100 = team_adjem[team] - opp_em + hfa_adj

            resid = actual_margin100 - expected_margin100

            # Opponent strength percentile (0..1)
            s = strength_pct_map.get(opp, 0.0)
            s = float(np.clip(s, 0.0, 1.0))

            w_top = (s ** opp_weight_power)
            w_bad = ((1.0 - s) ** opp_weight_power)

            top_num += w_top * resid
            top_den += w_top

            bad_num += w_bad * resid
            bad_den += w_bad

            n_used += 1

        top_perf = (top_num / top_den) if top_den > 0 else 0.0
        bad_beat = (bad_num / bad_den) if bad_den > 0 else 0.0

        # Contrast metric: good vs top minus padding vs bad
        strength_adjust = top_perf - bad_beat

        # Optional: z-scale for comparability
        strength_adjust_z = strength_adjust / sigma0 if sigma0 else 0.0

        # Shrink toward 0 for small sample sizes
        w_sh = n_used / (n_used + shrink_lambda) if n_used > 0 else 0.0
        strength_adjust = w_sh * strength_adjust
        top_perf = w_sh * top_perf
        bad_beat = w_sh * bad_beat
        strength_adjust_z = w_sh * strength_adjust_z

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

# --- RESUME CALCULATOR (NEW) ---
def calculate_resume_stats(ratings_df, scores_df, pts_col):
    """
    Calculates detailed resume metrics: Quadrant Records, SOS, Top 100 Record, Road Performance.
    """
    print("Calculating Resume Metrics (Quadrants, SOS, Road Record)...")
    
    # 1. Setup Lookups
    # Rank map based on Blended AdjEM
    ratings_df = ratings_df.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    ratings_df['Rank'] = ratings_df.index + 1
    
    rank_map = ratings_df.set_index('Team')['Rank'].to_dict()
    adjem_map = ratings_df.set_index('Team')['Blended_AdjEM'].to_dict()
    
    # 2. Prepare Game Data
    opp_scores = scores_df[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    opp_scores = opp_scores.drop_duplicates(subset=['date', 'opponent'])
    games = scores_df.merge(opp_scores, on=['date', 'opponent'], how='left')
    
    # 3. Helper for Quadrant Logic (NCAA NET Definition)
    def get_quadrant(opp_rank, location_val):
        # 1=Home, 0=Neutral, -1=Away
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
        
        # Quadrant Counters
        q1_w, q1_l = 0, 0
        q2_w, q2_l = 0, 0
        q3_w, q3_l = 0, 0
        q4_w, q4_l = 0, 0
        
        top100_w, top100_l = 0, 0
        
        # New Variables Containers
        sos_list = []
        road_wins = 0
        road_games = 0
        
        for _, row in team_games.iterrows():
            opp = row['opponent']
            if opp not in rank_map: continue 
            
            opp_rank = rank_map[opp]
            opp_adjem = adjem_map[opp]
            
            # 1. Accumulate SOS (Average Opponent AdjEM)
            sos_list.append(opp_adjem)
            
            # Determine Win/Loss
            is_win = row[pts_col] > row['opp_pts']
            
            # 2. Accumulate Road/Neutral Record
            # location_value: 1 (Home), 0 (Neutral), -1 (Away)
            if row['location_value'] != 1:
                road_games += 1
                if is_win: road_wins += 1
            
            # Quadrants
            quad = get_quadrant(opp_rank, row['location_value'])
            
            if quad == 1:
                if is_win: q1_w += 1
                else: q1_l += 1
            elif quad == 2:
                if is_win: q2_w += 1
                else: q2_l += 1
            elif quad == 3:
                if is_win: q3_w += 1
                else: q3_l += 1
            else: # Quad 4
                if is_win: q4_w += 1
                else: q4_l += 1
            
            if opp_rank <= 100:
                if is_win: top100_w += 1
                else: top100_l += 1
        
        # Final Calculations
        avg_sos = sum(sos_list) / len(sos_list) if sos_list else -10.0
        
        # Road Win % (Protect against divide by zero)
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

# --- CONSISTENCY / VOLATILITY METRIC (NEW) ---
def calculate_consistency(ratings_df, scores_df, pts_col, model_hfa,
                          sigma0=11.0,
                          tau_vol=0.90,
                          tau_swing=1.20,
                          alpha=0.40,
                          shrink_lambda=8,
                          # ---- FIXED PRIORS (do not change year-to-year) ----
                          vol_prior_z=1.00,     # typical within-team volatility in z units
                          swing_prior_z=1.20):  # typical adjacent swing in z units
    print("Calculating Volatility-Based Consistency Scores...")

    stats_map = ratings_df.set_index('Team')[['Blended_AdjEM']].to_dict('index')

    opp_scores = scores_df[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    opp_scores = opp_scores.drop_duplicates(subset=['date', 'opponent'])
    games = scores_df.merge(opp_scores, on=['date', 'opponent'], how='left')
    games = games.sort_values(['team', 'date'])

    team_z = {}
    for _, row in games.iterrows():
        team = row['team']; opp = row['opponent']
        if team not in stats_map or opp not in stats_map:
            continue

        poss = row['possessions']; opp_pts = row['opp_pts']
        if pd.isna(poss) or poss == 0 or pd.isna(opp_pts):
            continue

        pts = row[pts_col]
        actual_margin100 = (pts - opp_pts) / poss * 100.0

        loc_val = row.get('location_value', 0)
        hfa_adj = model_hfa if loc_val == 1 else (-model_hfa if loc_val == -1 else 0.0)

        expected_margin100 = stats_map[team]['Blended_AdjEM'] - stats_map[opp]['Blended_AdjEM'] + hfa_adj
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

        # Shrink toward FIXED priors (not season averages)
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
    """
    Robust Resume Quality (RQ) vs bubble team using Pythagenport probabilities.

    Fixes duplicate-game inflation by:
      - creating canonical game_id (date-normalized + sorted team names)
      - collapsing to exactly one row per (team, game_id)

    Adds:
      - RQ_Win_Avg, RQ_Loss_Avg
      - RQ_Wins_N, RQ_Losses_N
    """

    print("Calculating Resume Quality (RQ) using Pythagenport (robust game-level)...")

    # --- Bubble profile ---
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

    # Opponent stats map (fallback handled below)
    stats_map = ratings_df.set_index('Team')[['AdjO', 'AdjD', 'AdjT']].to_dict('index')

    # --- Build game table with opponent points ---
    games = scores_df.copy()
    games = games.dropna(subset=['date', 'team', 'opponent'])

    # Normalize date to day (kills timestamp duplicates)
    games['date_day'] = pd.to_datetime(games['date']).dt.normalize()

    # Create canonical (order-invariant) game id
    def _pair_key(r):
        a, b = str(r['team']), str(r['opponent'])
        x, y = (a, b) if a <= b else (b, a)
        return f"{x}__{y}"

    games['pair_key'] = games.apply(_pair_key, axis=1)
    games['game_id'] = games['date_day'].astype(str) + "__" + games['pair_key']

    # Attach opponent points by matching opponent's row in same game_id
    opp_pts_df = games[['game_id', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    # if opponent has duplicates too, reduce before merge
    opp_pts_df = opp_pts_df.groupby(['game_id', 'opponent'], as_index=False).first()

    games = games.merge(opp_pts_df, on=['game_id', 'opponent'], how='left')

    # Determine win/loss for THIS team vs opponent
    games['is_win'] = (games[pts_col] > games['opp_pts']).astype(int)

    # Collapse to ONE row per (team, game_id)
    # Sort preference: keep row with non-null opp_pts, then most recent original date if you had timestamps
    games = games.sort_values(['team', 'game_id'])
    games = games.groupby(['team', 'game_id'], as_index=False).first()

    # --- Per-team accumulation ---
    rq_total_map = {}
    rq_win_avg_map = {}
    rq_loss_avg_map = {}
    rq_wins_n_map = {}
    rq_losses_n_map = {}

    for team in ratings_df['Team']:
        tg = games[games['team'] == team].copy()
        if tg.empty:
            rq_total_map[team] = 0.0
            rq_win_avg_map[team] = 0.0
            rq_loss_avg_map[team] = 0.0
            rq_wins_n_map[team] = 0
            rq_losses_n_map[team] = 0
            continue

        win_resids = []
        loss_resids = []
        expected_wins_sum = 0.0
        actual_wins_sum = 0.0

        for _, row in tg.iterrows():
            opp_name = row['opponent']
            location = row.get('location_value', 0)  # 1=Home, -1=Away, 0=Neutral

            # Opponent profile
            opp = stats_map.get(opp_name, {'AdjO': 95.0, 'AdjD': 115.0, 'AdjT': AVG_TEMPO})

            # Location adjustment from bubble perspective
            hfa_adj = HFA_VAL if location == 1 else (-HFA_VAL if location == -1 else 0.0)

            # Expected efficiencies
            exp_eff_bubble = bubble_stats['AdjO'] + opp['AdjD'] - AVG_EFF + hfa_adj
            exp_eff_opp    = opp['AdjO'] + bubble_stats['AdjD'] - AVG_EFF - hfa_adj

            # Expected tempo
            exp_tempo = bubble_stats['AdjT'] + opp['AdjT'] - AVG_TEMPO

            # Projected scores
            score_bubble = (exp_eff_bubble * exp_tempo) / 100.0
            score_opp    = (exp_eff_opp * exp_tempo) / 100.0

            # Win probability
            if score_bubble <= 0 or score_opp <= 0:
                p = 0.0 if score_opp > score_bubble else 1.0
            else:
                p = (score_bubble ** PYTH_EXP) / ((score_bubble ** PYTH_EXP) + (score_opp ** PYTH_EXP))

            expected_wins_sum += p

            is_win = int(row['is_win'])
            actual_wins_sum += is_win

            resid = is_win - p
            if is_win == 1:
                win_resids.append(resid)     # 1 - p
            else:
                loss_resids.append(resid)    # -p

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
def calculate_ratings(
    input_file='cbb_scores.csv',
    preseason_file='pseason.csv'
):
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
    if 'mp' not in df.columns: df['mp'] = 200
    df['mp'] = pd.to_numeric(df['mp'], errors='coerce').fillna(200).replace(0, 200)

    df['date'] = pd.to_datetime(df['date'])
    most_recent = df['date'].max()
    df['days_ago'] = (most_recent - df['date']).dt.days
    decay_rate = 0.99

    weights = df['possessions'] * (decay_rate ** df['days_ago'])

    # ---------- BASELINES ----------
    global_ppp = (df[pts_col].sum() / df['possessions'].sum()) * 100
    df['pace_40'] = (df['possessions'] / df['mp']) * 200
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

    X_eff = lil_matrix((len(df), 2*n + 1))
    X_pace = lil_matrix((len(df), n))

    for i, row in df.reset_index(drop=True).iterrows():
        t = t2i[row['team']]
        o = t2i[row['opponent']]
        X_eff[i, t] = 1
        X_eff[i, n + o] = -1
        X_eff[i, 2*n] = row['location_value']
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
    clf_eff = RidgeCV(
        alphas=[0.1, 1, 5, 10, 25],
        fit_intercept=False
    ).fit(X_eff, y_eff, sample_weight=weights.values)

    clf_pace = RidgeCV(
        alphas=[0.1, 1, 5, 10],
        fit_intercept=False
    ).fit(X_pace, y_pace, sample_weight=weights.values)

    coefs = clf_eff.coef_
    pace_coefs = clf_pace.coef_
    model_hfa = coefs[-1]

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
            "Current_AdjEM": adj_em,
            "AdjO": round(adj_o, 1),
            "AdjD": round(adj_d, 1),
            "AdjT": round(adj_t, 1)
        })

    ratings = pd.DataFrame(current)

    # ---------- GAMES PLAYED ----------
    games_played = df.groupby('team').size().rename("Games").reset_index().rename(columns={"team": "Team"})
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
    G = ratings['Games']
    w = ratings['FinalPreseasonWeight']
    ratings['Blended_AdjEM'] = (G/(G + w) * ratings['Current_AdjEM'] + w/(G + w) * ratings['Preseason_AdjEM'])

    # --- ADVANCED CALCULATIONS ---
    # 1. RQ
    ratings = calculate_rq_pythag(ratings, df, pts_col)
    
    # 2. Consistency
    ratings = calculate_consistency(ratings, df, pts_col, model_hfa)
    
    # 3. Resume Stats (Quadrants, SOS, Top 100)
    ratings = calculate_resume_stats(ratings, df, pts_col)

    # 4. Strength Adjust (NEW)
    ratings = calculate_strength_adjust(ratings, df, pts_col, model_hfa)

    # ---------- RANK ----------
    ratings = ratings.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    ratings.index += 1
    ratings.index.name = 'Rank'

    ratings.to_csv('cbb_ratings.csv')

    print("\nTop 15 Analysis:")
    print(
        ratings[
            ['Team', 'Blended_AdjEM', 'RQ', 'StrengthAdjust', 'TopPerf', 'BadBeat',
             'SOS', 'Q1_W', 'Q1_L', 'T100_W']
        ].head(15).round(2).to_string()
    )
    print("\nSaved to cbb_ratings.csv")

if __name__ == "__main__":
    calculate_ratings()
