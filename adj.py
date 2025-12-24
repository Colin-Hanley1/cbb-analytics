import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.sparse import lil_matrix
from datetime import datetime

# ================= CONFIG =================
BASE_PRESEASON_GAMES = 12         # baseline preseason strength
SURPRISE_SCALE = 8.0              # AdjEM std dev
SURPRISE_CUTOFF = 2.0             # z-score where collapse accelerates
SURPRISE_STEEPNESS = 1.2          # how sharp the cutoff is

# Prediction Constants (Matching predict.py)
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

# --- WAB CALCULATOR (UPDATED WITH PYTHAGENPORT) ---
def calculate_wab_pythag(ratings_df, scores_df, pts_col):
    """
    Calculates Wins Above Bubble (WAB) using the Pythagenport prediction logic.
    """
    print("Calculating Wins Above Bubble (WAB) using Pythagenport...")
    
    # 1. Determine Bubble Team Profile (Rank #45-#50)
    sorted_ratings = ratings_df.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    
    start_idx = 44
    end_idx = 50
    if len(sorted_ratings) < 50:
        start_idx = max(0, len(sorted_ratings) - 6)
        end_idx = len(sorted_ratings)
        
    bubble_subset = sorted_ratings.iloc[start_idx:end_idx]
    
    # We create a theoretical "Bubble Team" with these average stats
    bubble_stats = {
        'AdjO': bubble_subset['AdjO'].mean(),
        'AdjD': bubble_subset['AdjD'].mean(),
        'AdjT': bubble_subset['AdjT'].mean()
    }
    print(f"  Bubble Profile: AdjO {bubble_stats['AdjO']:.1f} | AdjD {bubble_stats['AdjD']:.1f} | Tempo {bubble_stats['AdjT']:.1f}")

    # 2. Create Lookup for Opponent Stats
    # We use the full stats map for lookups
    stats_map = ratings_df.set_index('Team')[['AdjO', 'AdjD', 'AdjT']].to_dict('index')

    # 3. Prepare Game Data
    # Merge scores to get opponent identity and game location
    opp_scores = scores_df[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    # Deduplicate to prevent merge explosions
    opp_scores = opp_scores.drop_duplicates(subset=['date', 'opponent'])
    
    games = scores_df.merge(opp_scores, on=['date', 'opponent'], how='left')
    
    # Determine Actual Wins
    games['is_win'] = (games[pts_col] > games['opp_pts']).astype(float)
    
    # 4. Iterate and Calculate Expected Wins for the Bubble Team
    expected_wins = {}
    actual_wins = {}
    
    for team in ratings_df['Team']:
        team_games = games[games['team'] == team].copy()
        
        if team_games.empty:
            expected_wins[team] = 0.0
            actual_wins[team] = 0.0
            continue
            
        bubble_win_probs = []
        
        for _, row in team_games.iterrows():
            opp_name = row['opponent']
            location = row['location_value'] # 1=Home, -1=Away, 0=Neutral
            
            # Get Opponent Stats (Default to a "Bad D1" profile if missing)
            if opp_name in stats_map:
                opp = stats_map[opp_name]
            else:
                # Default for Non-D1: Bad Offense, Bad Defense, Avg Tempo
                opp = {'AdjO': 95.0, 'AdjD': 115.0, 'AdjT': 68.5}
            
            # --- PYTHAGENPORT PREDICTION LOGIC ---
            # Bubble Team (A) vs Opponent (B)
            
            # 1. Adjust for Location (From Bubble Team's perspective)
            # If original team played Home (1), Bubble Team plays Home
            hfa_adj = 0
            if location == 1: hfa_adj = HFA_VAL
            elif location == -1: hfa_adj = -HFA_VAL
                
            # 2. Expected Efficiency
            # Bubble Offense vs Opp Defense
            exp_eff_bubble = bubble_stats['AdjO'] + opp['AdjD'] - AVG_EFF + hfa_adj
            # Opp Offense vs Bubble Defense
            exp_eff_opp = opp['AdjO'] + bubble_stats['AdjD'] - AVG_EFF - hfa_adj
            
            # 3. Expected Tempo
            exp_tempo = bubble_stats['AdjT'] + opp['AdjT'] - AVG_TEMPO
            
            # 4. Projected Score
            score_bubble = (exp_eff_bubble * exp_tempo) / 100
            score_opp = (exp_eff_opp * exp_tempo) / 100
            
            # 5. Win Probability
            # Avoid divide by zero if scores are weirdly 0
            if score_bubble <= 0 or score_opp <= 0:
                prob = 0.0 if score_opp > score_bubble else 1.0
            else:
                prob = (score_bubble ** PYTH_EXP) / ((score_bubble ** PYTH_EXP) + (score_opp ** PYTH_EXP))
            
            bubble_win_probs.append(prob)
            
        # Sum probabilities to get expected wins against this schedule
        expected_wins[team] = sum(bubble_win_probs)
        actual_wins[team] = team_games['is_win'].sum()

    # 5. Attach to DataFrame
    ratings_df['WAB'] = ratings_df['Team'].apply(lambda x: actual_wins.get(x, 0) - expected_wins.get(x, 0))
    
    return ratings_df

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

    df = df[df['possessions'] > 0].copy()
    df['mp'] = df.get('mp', 200).replace(0, 200)

    df['date'] = pd.to_datetime(df['date'])
    most_recent = df['date'].max()
    df['days_ago'] = (most_recent - df['date']).dt.days
    decay_rate = 0.99

    weights = df['possessions'] * (decay_rate ** df['days_ago'])

    # ---------- BASELINES ----------
    global_ppp = (df[pts_col].sum() / df['possessions'].sum()) * 100
    df['pace_40'] = (df['possessions'] / df['mp']) * 200
    global_pace = df['pace_40'].mean()

    # ---------- MATRICES ----------
    teams = sorted(set(df['team']) | set(df['opponent']))
    n = len(teams)
    t2i = {t: i for i, t in enumerate(teams)}

    X_eff = lil_matrix((len(df), 2*n + 1))
    X_pace = lil_matrix((len(df), n))

    for i, row in df.iterrows():
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

    # ---------- FIT ----------
    clf_eff = RidgeCV(alphas=[0.1, 1, 5, 10, 25], fit_intercept=False).fit(X_eff, y_eff, sample_weight=weights)
    clf_pace = RidgeCV(alphas=[0.1, 1, 5, 10], fit_intercept=False).fit(X_pace, y_pace, sample_weight=weights)

    coefs = clf_eff.coef_
    pace_coefs = clf_pace.coef_

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

    # ---------- WAB CALCULATION (PYTHAGENPORT) ----------
    ratings = calculate_wab_pythag(ratings, df, pts_col)

    # ---------- RANK ----------
    ratings = ratings.sort_values('Blended_AdjEM', ascending=False).reset_index(drop=True)
    ratings.index += 1
    ratings.index.name = 'Rank'

    ratings.to_csv('cbb_ratings.csv')

    print("\nTop 15 (Blended AdjEM):")
    print(ratings[['Team', 'Blended_AdjEM', 'WAB', 'Current_AdjEM', 'Games']].head(15).round(2).to_string())

    print("\nSaved to cbb_ratings.csv")

# ---------- RUN ----------
if __name__ == "__main__":
    calculate_ratings()