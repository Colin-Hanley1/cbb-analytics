import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.sparse import lil_matrix
from datetime import datetime

# ================= CONFIG =================
BASE_PRESEASON_GAMES = 12         # baseline preseason strength
SURPRISE_SCALE = 8.0              # AdjEM std dev (~7–9 typical)
SURPRISE_CUTOFF = 2.0             # z-score where collapse accelerates
SURPRISE_STEEPNESS = 1.2          # how sharp the cutoff is
# ==========================================

# --- NAME MAPPING ---
NAME_MAPPING = {
   "Iowa State": "Iowa St.",
   "Brigham Young": "BYU",
   "Michigan State": "Michigan St.",
   "Saint Mary's (CA)": "Saint Mary's",
   "St. John's (NY)": "St. John's",
   "Mississippi State": "Mississippi St.",
   "Ohio State": "Ohio St.",
   "Southern Methodist": "SMU",
   "Utah State": "Utah St.",
   "San Diego State": "San Diego St.",
   "Southern California": "USC",
   "NC State": "N.C. State",
   "Virginia Commonwealth": "VCU",
   "Oklahoma State": "Oklahoma St.",
   "Louisiana State": "LSU",
   "Penn State": "Penn St.",
   "Boise State": "Boise St.",
   "Miami (FL)": "Miami FL",
   "Colorado State": "Colorado St.",
   "Arizona State": "Arizona St.",
   "Kansas State": "Kansas St.",
   "Florida State": "Florida St.",
   "McNeese State": "McNeese",
   "Nevada-Las Vegas": "UNLV",
   "Loyola (IL)": "Loyola Chicago",
   "Kennesaw State": "Kennesaw St.",
   "Murray State": "Murray St.",
   "Kent State": "Kent St.",
   "Illinois State": "Illinois St.",
   "College of Charleston": "Charleston",
   "Cal State Northridge": "CSUN",
   "Wichita State": "Wichita St.",
   "Miami (OH)": "Miami OH",
   "East Tennessee State": "East Tennessee St.",
   "South Dakota State": "South Dakota St.",
   "New Mexico State": "New Mexico St.",
   "Oregon State": "Oregon St.",
   "Jacksonville State": "Jacksonville St.",
   "Arkansas State": "Arkansas St.",
   "Montana State": "Montana St.",
   "Omaha": "Nebraska Omaha",
   "Sam Houston": "Sam Houston St.",
   "California Baptist": "Cal Baptist",
   "Portland State": "Portland St.",
   "Nicholls State": "Nicholls",
   "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
   "Illinois-Chicago": "Illinois Chicago",
   "Youngstown State": "Youngstown St.",
   "North Dakota State": "North Dakota St.",
   "Queens (NC)": "Queens",
   "Southeast Missouri State": "Southeast Missouri",
   "Texas State": "Texas St.",
   "Jackson State": "Jackson St.",
   "Appalachian State": "Appalachian St.",
   "Wright State": "Wright St.",
   "Indiana State": "Indiana St.",
   "Missouri State": "Missouri St.",
   "San Jose State": "San Jose St.",
   "Bethune-Cookman": "Bethune Cookman",
   "Southern Illinois-Edwardsville": "SIUE",
   "Loyola (MD)": "Loyola MD",
   "Norfolk State": "Norfolk St.",
   "Idaho State": "Idaho St.",
   "Texas-Rio Grande Valley": "UT Rio Grande Valley",
   "South Carolina State": "South Carolina St.",
   "Georgia State": "Georgia St.",
   "Washington State": "Washington St.",
   "Cleveland State": "Cleveland St.",
   "Northwestern State": "Northwestern St.",
   "Albany (NY)": "Albany",
   "Virginia Military Institute": "VMI",
   "Maryland-Baltimore County": "UMBC",
   "Pennsylvania": "Penn",
   "Long Island University": "LIU",
   "Tennessee-Martin": "Tennessee Martin",
   "Tennessee State": "Tennessee St.",
   "Central Connecticut State": "Central Connecticut",
   "Weber State": "Weber St.",
   "Tarleton State": "Tarleton St." ,
   "Morgan State": "Morgan St.",
   "Morehead State": "Morehead St.",
   "Fresno State": "Fresno St.",
   "Cal State Bakersfield": "Cal St. Bakersfield",
   "Ball State": "Ball St.",
   "Alabama State": "Alabama St.",
   "Sacramento State": "Sacramento St.",
   "Long Beach State": "Long Beach St.",
   "Massachusetts-Lowell": "UMass Lowell",
   "South Carolina Upstate": "USC Upstate",
   "Florida International": "FIU",
   "Southern Miss.": "Southern Miss.",
   "Gardner-Webb": "Gardner Webb",
   "Cal State Fullerton": "Cal St. Fullerton",
   "Coppin State": "Coppin St.",
   "Maryland-Eastern Shore": "Maryland Eastern Shore",
   "Saint Francis (PA)": "Saint Francis",
   "FDU": "Fairleigh Dickinson",
   "Grambling": "Grambling St.",
   "Alcorn State": "Alcorn St.",
   "Delaware State": "Delaware St.",
   "Chicago State": "Chicago St.",
   "Louisiana-Monroe": "Louisiana Monroe",
   "Prairie View": "Prairie View A&M",
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
    """
    Logistic collapse of preseason belief when surprise is large
    """
    return 1 / (1 + np.exp(
        SURPRISE_STEEPNESS * (np.abs(z) - SURPRISE_CUTOFF)
    ))

def calculate_ratings(
    input_file='cbb_scores.csv',
    preseason_file='pseason.csv'
):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

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
    clf_eff = RidgeCV(
        alphas=[0.1, 1, 5, 10, 25],
        fit_intercept=False
    ).fit(X_eff, y_eff, sample_weight=weights)

    clf_pace = RidgeCV(
        alphas=[0.1, 1, 5, 10],
        fit_intercept=False
    ).fit(X_pace, y_pace, sample_weight=weights)

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
    games_played = (
        df.groupby('team')
          .size()
          .rename("Games")
          .reset_index()
          .rename(columns={"team": "Team"})
    )

    ratings = ratings.merge(games_played, on="Team", how="left")

    # ---------- PRESEASON ----------
    preseason = pd.read_csv(preseason_file)
    preseason['Team'] = preseason['Team'].replace(NAME_MAPPING)

    ratings = ratings.merge(
        preseason[['Team', 'Preseason_AdjEM']],
        on='Team',
        how='left'
    )

    ratings['Preseason_AdjEM'] = ratings['Preseason_AdjEM'].fillna(0.0)

    # ---------- PRIOR COLLAPSE ----------
    ratings['SurpriseZ'] = (
        ratings['Current_AdjEM'] - ratings['Preseason_AdjEM']
    ) / SURPRISE_SCALE

    ratings['CollapseFactor'] = ratings['SurpriseZ'].apply(prior_collapse_factor)

    ratings['FinalPreseasonWeight'] = (
        BASE_PRESEASON_GAMES * ratings['CollapseFactor']
    )

    # ---------- BLENDED RATING ----------
    G = ratings['Games']
    w = ratings['FinalPreseasonWeight']

    ratings['Blended_AdjEM'] = (
        G/(G + w) * ratings['Current_AdjEM'] +
        w/(G + w) * ratings['Preseason_AdjEM']
    )

    # ---------- RANK ----------
    ratings = ratings.sort_values(
        'Blended_AdjEM',
        ascending=False
    ).reset_index(drop=True)

    ratings.index += 1
    ratings.index.name = 'Rank'

    ratings.to_csv('cbb_ratings.csv')

    print("\nTop 15 (Blended AdjEM):")
    print(
        ratings[['Team', 'Blended_AdjEM',
                 'Current_AdjEM', 'Preseason_AdjEM',
                 'SurpriseZ', 'Games']]
        .head(15)
        .round(2)
        .to_string()
    )

    print("\nSaved to cbb_ratings.csv")

# ---------- RUN ----------
if __name__ == "__main__":
    calculate_ratings()
