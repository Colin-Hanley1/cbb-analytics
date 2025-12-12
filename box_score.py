import pandas as pd
import numpy as np

SOURCE_FILE = "cbb_scores.csv"
OUTPUT_FILE = "team_advanced_averages.csv"
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
   "Tarleton State": "Tarleton St.",
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

def apply_name_mapping(df):
    df["team"] = df["team"].apply(lambda x: NAME_MAPPING.get(x, x))
    df["opponent"] = df["opponent"].apply(lambda x: NAME_MAPPING.get(x, x))
    return df


def compute_team_stats(df):
    """
    Computes per-team averages (basic & advanced stats)
    including opponent-adjusted defensive metrics.
    """

    # Fill missing numeric columns with 0
    df = df.fillna(0)
    df = apply_name_mapping(df)
    # ---------------------------------------------------------
    # 1. PRE-COMPUTE OPPONENT MERGE FOR DEFENSIVE METRICS
    # ---------------------------------------------------------
    opp_df = df.copy()
    opp_df = opp_df.rename(columns={
        "team": "opponent",
        "opponent": "team"
    })
    merged = df.merge(
        opp_df,
        on=["team", "date"],
        suffixes=("", "_opp"),
        how="left"
    )

    # ---------------------------------------------------------
    # 2. ADVANCED METRICS (computed per game BEFORE AVERAGING)
    # ---------------------------------------------------------

    # Shooting splits
    merged["fg2"] = merged["fg"] - merged["fg3"]
    merged["fg2a"] = merged["fga"] - merged["fg3a"]

    # Avoid div-by-zero
    merged["fg2_pct"] = (merged["fg2"] / merged["fg2a"].replace(0, np.nan)).fillna(0)

    # eFG%
    merged["efg"] = (
        (merged["fg"] + 0.5 * merged["fg3"]) /
        merged["fga"].replace(0, np.nan)
    ).fillna(0)

    # 3PA rate, 2PA rate
    merged["rate_3pa"] = (merged["fg3a"] / merged["fga"].replace(0, np.nan)).fillna(0)
    merged["rate_2pa"] = (merged["fg2a"] / merged["fga"].replace(0, np.nan)).fillna(0)

    # FT rate
    merged["ft_rate"] = (merged["fta"] / merged["fga"].replace(0, np.nan)).fillna(0)
    merged["ftm_rate"] = (merged["ft"] / merged["fga"].replace(0, np.nan)).fillna(0)

    # Turnover Rate
    merged["tov_rate"] = (
        merged["tov"] / merged["possessions"].replace(0, np.nan)
    ).fillna(0)

    # Offensive Rebound Rate
    merged["orb_rate"] = (
        merged["orb"] /
        (merged["orb"] + merged["drb_opp"]).replace(0, np.nan)
    ).fillna(0)

    # Defensive Rebound Rate
    merged["drb_rate"] = (
        merged["drb"] /
        (merged["drb"] + merged["orb_opp"]).replace(0, np.nan)
    ).fillna(0)

    # Assist / Steal / Block rate (per possession)
    merged["ast_rate"] = (merged["ast"] / merged["possessions"].replace(0, np.nan)).fillna(0)
    merged["stl_rate"] = (merged["stl"] / merged["possessions"].replace(0, np.nan)).fillna(0)
    merged["blk_rate"] = (merged["blk"] / merged["possessions"].replace(0, np.nan)).fillna(0)

    # Pace (per 40 minutes)
    merged["pace"] = (
        merged["possessions"] * 40 / (merged["mp"].replace(0, np.nan))
    ).fillna(0)

    # Offensive Efficiency already computed as raw_off_eff
    # Defensive Efficiency
    merged["def_eff"] = (
        (merged["pts_opp"] / merged["possessions"].replace(0, np.nan)) * 100
    ).fillna(0)

    # Net Efficiency
    merged["net_eff"] = merged["raw_off_eff"] - merged["def_eff"]

    # ---------------------------------------------------------
    # 3. AGGREGATE MEAN BY TEAM
    # ---------------------------------------------------------
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    team_avgs = merged.groupby("team")[numeric_cols].mean()

    # Add number of games
    team_avgs["games_played"] = merged.groupby("team")["pts"].count()

    # Round everything to 4 decimal places
    team_avgs = team_avgs.round(4)

    return team_avgs.reset_index()


if __name__ == "__main__":
    print(f"Loading scraper file: {SOURCE_FILE}")
    df = pd.read_csv(SOURCE_FILE)

    print("Computing all team averages + advanced metrics...")
    results = compute_team_stats(df)

    print(f"Saving to: {OUTPUT_FILE}")
    results.to_csv(OUTPUT_FILE, index=False)

    print("\n✓ DONE!")
    print(f"Generated advanced statistical averages for {results.shape[0]} teams.")
