import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.sparse import lil_matrix
from scipy.stats import norm
from datetime import datetime

# ================= CONFIG =================
BASE_PRESEASON_GAMES = 12         # baseline preseason strength
SURPRISE_SCALE = 8.0              # AdjEM std dev
SURPRISE_CUTOFF = 2.0             # z-score where collapse accelerates
SURPRISE_STEEPNESS = 1.2          # how sharp the cutoff is

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

# --- RESUME CALCULATOR (NEW) ---
def calculate_resume_stats(ratings_df, scores_df, pts_col):
    """
    Calculates detailed resume metrics: Quadrant Records, SOS, Top 100 Record.
    """
    print("Calculating Resume Metrics (Quadrants, SOS, Top 100)...")
    
    # 1. Setup Lookups
    # Rank map: Team -> Rank (1 to 362)
    ratings_df = ratings_df.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    ratings_df['Rank'] = ratings_df.index + 1
    
    rank_map = ratings_df.set_index('Team')['Rank'].to_dict()
    adjem_map = ratings_df.set_index('Team')['Blended_AdjEM'].to_dict()
    
    # 2. Prepare Game Data
    opp_scores = scores_df[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    opp_scores = opp_scores.drop_duplicates(subset=['date', 'opponent'])
    games = scores_df.merge(opp_scores, on=['date', 'opponent'], how='left')
    
    # 3. Helper for Quadrant Logic
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

    # 4. Aggregators
    resume_data = []

    for team in ratings_df['Team']:
        team_games = games[games['team'] == team].copy()
        
        # Counters
        q1_w, q1_l = 0, 0
        q2_w, q2_l = 0, 0
        q3_w, q3_l = 0, 0
        q4_w, q4_l = 0, 0
        
        top50_w, top50_l = 0, 0
        top100_w, top100_l = 0, 0
        
        sos_list = []
        
        for _, row in team_games.iterrows():
            opp = row['opponent']
            if opp not in rank_map: continue # Skip non-D1
            
            opp_rank = rank_map[opp]
            opp_adjem = adjem_map[opp]
            
            # Add to SOS
            sos_list.append(opp_adjem)
            
            # Result
            is_win = row[pts_col] > row['opp_pts']
            
            # Quadrant
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
            
            # Top 50 / 100
            if opp_rank <= 50:
                if is_win: top50_w += 1
                else: top50_l += 1
            if opp_rank <= 100:
                if is_win: top100_w += 1
                else: top100_l += 1
        
        # Calculate Averages
        sos = sum(sos_list) / len(sos_list) if sos_list else -10.0
        
        resume_data.append({
            'Team': team,
            'SOS': round(sos, 2),
            'Q1_W': q1_w, 'Q1_L': q1_l,
            'Q2_W': q2_w, 'Q2_L': q2_l,
            'Q3_W': q3_w, 'Q3_L': q3_l,
            'Q4_W': q4_w, 'Q4_L': q4_l,
            'T100_W': top100_w, 'T100_L': top100_l
        })

    # Merge Resume Data back into Ratings
    resume_df = pd.DataFrame(resume_data)
    ratings_df = ratings_df.merge(resume_df, on='Team', how='left')
    return ratings_df

# --- CONSISTENCY CALCULATOR ---
def calculate_consistency(ratings_df, scores_df, pts_col, model_hfa):
    print("Calculating Consistency Scores...")
    stats_map = ratings_df.set_index('Team')[['Blended_AdjEM']].to_dict('index')
    
    opp_scores = scores_df[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    opp_scores = opp_scores.drop_duplicates(subset=['date', 'opponent'])
    games = scores_df.merge(opp_scores, on=['date', 'opponent'], how='left')
    
    residuals = {}
    
    for _, row in games.iterrows():
        team = row['team']
        opp = row['opponent']
        
        if team not in stats_map or opp not in stats_map: continue
            
        pts = row[pts_col]
        opp_pts = row['opp_pts']
        poss = row['possessions']
        if poss == 0 or pd.isna(opp_pts): continue
        
        actual_margin = (pts - opp_pts) / poss * 100
        loc_val = row['location_value']
        hfa_adj = 0
        if loc_val == 1: hfa_adj = model_hfa
        elif loc_val == -1: hfa_adj = -model_hfa
            
        exp_margin = stats_map[team]['Blended_AdjEM'] - stats_map[opp]['Blended_AdjEM'] + hfa_adj
        resid = actual_margin - exp_margin
        
        if team not in residuals: residuals[team] = []
        residuals[team].append(resid)
        
    cons_scores = []
    for team in ratings_df['Team']:
        res_list = residuals.get(team, [])
        if len(res_list) < 3:
            score = 50.0 
        else:
            mad = np.mean(np.abs(res_list))
            score = 100 - (mad * 4.0)
            score = max(0, min(100, score))
        cons_scores.append(score)
        
    ratings_df['Consistency'] = cons_scores
    return ratings_df

# --- WAB CALCULATOR ---
def calculate_wab_pythag(ratings_df, scores_df, pts_col):
    print("Calculating Wins Above Bubble (WAB)...")
    sorted_ratings = ratings_df.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    
    start_idx = 44 if len(sorted_ratings) > 50 else max(0, len(sorted_ratings) - 6)
    end_idx = 50 if len(sorted_ratings) > 50 else len(sorted_ratings)
    bubble_subset = sorted_ratings.iloc[start_idx:end_idx]
    
    bubble_stats = {
        'AdjO': bubble_subset['AdjO'].mean(),
        'AdjD': bubble_subset['AdjD'].mean(),
        'AdjT': bubble_subset['AdjT'].mean()
    }
    
    stats_map = ratings_df.set_index('Team')[['AdjO', 'AdjD', 'AdjT']].to_dict('index')

    opp_scores = scores_df[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    opp_scores = opp_scores.drop_duplicates(subset=['date', 'opponent'])
    
    games = scores_df.merge(opp_scores, on=['date', 'opponent'], how='left')
    games['is_win'] = (games[pts_col] > games['opp_pts']).astype(float)
    
    expected_wins = {}
    actual_wins = {}
    
    for team in ratings_df['Team']:
        team_games = games[games['team'] == team].copy()
        if team_games.empty:
            expected_wins[team] = 0.0; actual_wins[team] = 0.0
            continue
            
        bubble_win_probs = []
        for _, row in team_games.iterrows():
            opp_name = row['opponent']
            location = row['location_value']
            
            if opp_name in stats_map:
                opp = stats_map[opp_name]
            else:
                opp = {'AdjO': 95.0, 'AdjD': 115.0, 'AdjT': 68.5}
            
            hfa_adj = 0
            if location == 1: hfa_adj = HFA_VAL
            elif location == -1: hfa_adj = -HFA_VAL
                
            exp_eff_bubble = bubble_stats['AdjO'] + opp['AdjD'] - AVG_EFF + hfa_adj
            exp_eff_opp = opp['AdjO'] + bubble_stats['AdjD'] - AVG_EFF - hfa_adj
            exp_tempo = bubble_stats['AdjT'] + opp['AdjT'] - AVG_TEMPO
            
            score_bubble = (exp_eff_bubble * exp_tempo) / 100
            score_opp = (exp_eff_opp * exp_tempo) / 100
            
            if score_bubble <= 0 or score_opp <= 0:
                prob = 0.0 if score_opp > score_bubble else 1.0
            else:
                prob = (score_bubble ** PYTH_EXP) / ((score_bubble ** PYTH_EXP) + (score_opp ** PYTH_EXP))
            
            bubble_win_probs.append(prob)
            
        expected_wins[team] = sum(bubble_win_probs)
        actual_wins[team] = team_games['is_win'].sum()
        
        # Manual override (preserved from user request)
        if team == "Boise St.": actual_wins[team] = actual_wins[team] - 1

    ratings_df['WAB'] = ratings_df['Team'].apply(lambda x: actual_wins.get(x, 0) - expected_wins.get(x, 0))
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

    # Clean Data
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

    # Baselines
    global_ppp = (df[pts_col].sum() / df['possessions'].sum()) * 100
    df['pace_40'] = (df['possessions'] / df['mp']) * 200
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pace_40', 'raw_off_eff'])
    weights = df['possessions'] * (decay_rate ** df['days_ago'])
    global_pace = df['pace_40'].mean()
    
    if np.isnan(global_pace) or np.isnan(global_ppp): return

    # Matrices
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

    # Regression
    raw_diff = df['raw_off_eff'].values - global_ppp
    y_eff = cap_efficiency_margin(raw_diff)
    y_pace = df['pace_40'].values - global_pace
    
    if np.isnan(y_eff).any() or np.isnan(y_pace).any(): return

    clf_eff = RidgeCV(alphas=[0.1, 1, 5, 10, 25], fit_intercept=False).fit(X_eff, y_eff, sample_weight=weights.values)
    clf_pace = RidgeCV(alphas=[0.1, 1, 5, 10], fit_intercept=False).fit(X_pace, y_pace, sample_weight=weights.values)

    coefs = clf_eff.coef_
    pace_coefs = clf_pace.coef_
    model_hfa = coefs[-1]

    # Build Ratings
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

    # Add Context
    games_played = df.groupby('team').size().rename("Games").reset_index().rename(columns={"team": "Team"})
    ratings = ratings.merge(games_played, on="Team", how="left")

    try:
        preseason = pd.read_csv(preseason_file)
        preseason['Team'] = preseason['Team'].replace(NAME_MAPPING)
        ratings = ratings.merge(preseason[['Team', 'Preseason_AdjEM']], on='Team', how='left')
    except FileNotFoundError:
        ratings['Preseason_AdjEM'] = 0.0

    ratings['Preseason_AdjEM'] = ratings['Preseason_AdjEM'].fillna(0.0)

    # Bayesian Blend
    ratings['SurpriseZ'] = (ratings['Current_AdjEM'] - ratings['Preseason_AdjEM']) / SURPRISE_SCALE
    ratings['CollapseFactor'] = ratings['SurpriseZ'].apply(prior_collapse_factor)
    ratings['FinalPreseasonWeight'] = BASE_PRESEASON_GAMES * ratings['CollapseFactor']

    ratings['Blended_AdjEM'] = (
        ratings['Games']/(ratings['Games'] + ratings['FinalPreseasonWeight']) * ratings['Current_AdjEM'] +
        ratings['FinalPreseasonWeight']/(ratings['Games'] + ratings['FinalPreseasonWeight']) * ratings['Preseason_AdjEM']
    )

    # --- ADVANCED CALCULATIONS ---
    # 1. WAB
    ratings = calculate_wab_pythag(ratings, df, pts_col)
    
    # 2. Consistency
    ratings = calculate_consistency(ratings, df, pts_col, model_hfa)

    # 3. Resume Stats (Quadrants, SOS, Top 100) -> NEW
    ratings = calculate_resume_stats(ratings, df, pts_col)

    # Final Rank & Save
    ratings = ratings.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    ratings.index += 1
    ratings.index.name = 'Rank'

    ratings.to_csv('cbb_ratings.csv')

    print("\nTop 15 Analysis:")
    print(ratings[['Team', 'Blended_AdjEM', 'WAB', 'SOS', 'Q1_W', 'Q1_L', 'T100_W']].head(15).round(2).to_string())
    print("\nSaved to cbb_ratings.csv")

if __name__ == "__main__":
    calculate_ratings()