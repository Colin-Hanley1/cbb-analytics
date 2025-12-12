import pandas as pd
import math

def predict_matchup(team_a, team_b, location='Neutral', ratings_file='cbb_ratings.csv'):
    # Load Ratings
    df = pd.read_csv(ratings_file)
    df.set_index('Team', inplace=True)
    
    try:
        stats_a = df.loc[team_a]
        stats_b = df.loc[team_b]
    except KeyError:
        print("One of these teams was not found in the ratings.")
        return

    # League Averages (Approximate, or calculate from CSV if you saved them)
    # You can also pass these in dynamically
    avg_eff = 106.0 
    avg_tempo = 68.5
    hfa = 3.2 # Standard Home Court Advantage points

    # 1. Adjust for Location
    # If Team A is home, they get the efficiency boost
    hfa_adj = 0
    if location == 'Home':
        hfa_adj = hfa
    elif location == 'Away':
        hfa_adj = -hfa

    # 2. Calculate Expected Efficiency for this specific game
    # Team A Offense vs Team B Defense
    exp_eff_a = stats_a['AdjO'] + stats_b['AdjD'] - avg_eff + hfa_adj
    # Team B Offense vs Team A Defense
    exp_eff_b = stats_b['AdjO'] + stats_a['AdjD'] - avg_eff - hfa_adj

    # 3. Calculate Expected Tempo
    exp_tempo = stats_a['AdjT'] + stats_b['AdjT'] - avg_tempo

    # 4. Calculate Final Score
    score_a = (exp_eff_a * exp_tempo) / 100
    score_b = (exp_eff_b * exp_tempo) / 100

    # 5. Calculate Spread & Total
    spread = score_b - score_a # Negative means Team A is favored
    total = score_a + score_b

    # 6. Win Probability (Pythagenport Formula)
    # Exponent 'x' usually around 11.0 for CBB
    x = 11.5
    win_prob_a = (score_a**x) / (score_a**x + score_b**x)

    print(f"\n--- {team_a} vs {team_b} ({location}) ---")
    print(f"Proj Score: {team_a} {score_a:.1f} - {team_b} {score_b:.1f}")
    print(f"Spread:     {team_a} {spread:+.1f}")
    print(f"Total:      {total:.1f}")
    print(f"Win Prob:   {team_a} {win_prob_a*100:.1f}%")

# Example Usage
home= ''
away=''
loc=''
while home != 'q':
    home = str(input('Home Team: '))
    away = str(input('Away Team: '))
    loc = str(input('Location (Home/Away/Neutral): '))
    predict_matchup(home, away, location=loc)

predict_matchup('Duke', 'North Carolina', location='Home')