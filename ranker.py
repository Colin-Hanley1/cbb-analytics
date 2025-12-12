import pandas as pd
import numpy as np

# ---------------- CONFIG ---------------- #

RATINGS_FILE = "cbb_ratings.csv"
SCORES_FILE  = "cbb_scores.csv"
OUTPUT_FILE  = "predictive_rankings.csv"

# ---------------- LOAD DATA ---------------- #

ratings = pd.read_csv(RATINGS_FILE)

if "AdjEM" not in ratings.columns:
    raise ValueError("cbb_ratings.csv must contain an AdjEM column.")

ratings = ratings[["Team", "AdjEM"]].dropna()

# ---------------- EXPECTED MARGIN LOGIC ---------------- #

# If AdjEM is already defined as expected margin vs average team:
#   E[m_i vs avg] = AdjEM_i
#
# But we compute it via round-robin anyway to:
#   • reduce noise
#   • future-proof if model changes

teams = ratings["Team"].tolist()
theta = dict(zip(ratings["Team"], ratings["AdjEM"]))

def expected_margin(team_a, team_b):
    return theta[team_a] - theta[team_b]

rows = []

for team in teams:
    margins = []
    for opp in teams:
        if opp == team:
            continue
        margins.append(expected_margin(team, opp))
    rows.append({
        "Team": team,
        "ExpectedMargin_vs_Avg": np.mean(margins)
    })

rankings = pd.DataFrame(rows)

# ---------------- RANK ---------------- #

rankings = rankings.sort_values(
    "ExpectedMargin_vs_Avg",
    ascending=False
).reset_index(drop=True)

rankings["Rank"] = rankings.index + 1
rankings["ExpectedMargin_vs_Avg"] = rankings["ExpectedMargin_vs_Avg"].round(2)

rankings = rankings[
    ["Rank", "Team", "ExpectedMargin_vs_Avg"]
]

# ---------------- SAVE ---------------- #

rankings.to_csv(OUTPUT_FILE, index=False)

print("\nTop 15 Predictive Rankings:")
print(rankings.head(15).to_string(index=False))
print(f"\nSaved → {OUTPUT_FILE}")
