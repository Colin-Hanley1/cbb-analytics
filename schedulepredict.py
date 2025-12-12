import pandas as pd
import math
import os

# ---------------- CONFIG ---------------- #
RATINGS_FILE = "cbb_ratings.csv"
AVG_EFF = 106.0
AVG_TEMPO = 68.5
HFA = 3.2          # Home court advantage (points)
PYTH_EXPONENT = 11.5  # For win probability


def load_ratings(ratings_file: str = RATINGS_FILE) -> pd.DataFrame:
    """
    Load the ratings file and return a DataFrame indexed by Team.
    """
    df = pd.read_csv(ratings_file)
    df.set_index("Team", inplace=True)
    return df


def predict_matchup_from_df(
    team_a: str,
    team_b: str,
    location: str,
    df: pd.DataFrame,
    avg_eff: float = AVG_EFF,
    avg_tempo: float = AVG_TEMPO,
    hfa: float = HFA,
    x: float = PYTH_EXPONENT,
):
    """
    Predict a single matchup given a ratings DataFrame (df).

    location: 'Home', 'Away', or 'Neutral' from the perspective of team_a.
    Returns a dict with all relevant info.
    """
    try:
        stats_a = df.loc[team_a]
        stats_b = df.loc[team_b]
    except KeyError as e:
        raise KeyError(f"Team not found in ratings: {e}")

    # Location adjustment (for team_a)
    hfa_adj = 0.0
    loc_clean = location.strip().lower()

    if loc_clean == "home":
        hfa_adj = hfa
    elif loc_clean == "away":
        hfa_adj = -hfa
    else:
        hfa_adj = 0.0  # Neutral

    # 1. Expected efficiencies
    exp_eff_a = stats_a["AdjO"] + stats_b["AdjD"] - avg_eff + hfa_adj
    exp_eff_b = stats_b["AdjO"] + stats_a["AdjD"] - avg_eff - hfa_adj

    # 2. Expected tempo
    exp_tempo = stats_a["AdjT"] + stats_b["AdjT"] - avg_tempo

    # 3. Projected scores
    score_a = (exp_eff_a * exp_tempo) / 100.0
    score_b = (exp_eff_b * exp_tempo) / 100.0

    # 4. Spread & total (spread from perspective of Team A)
    spread = score_b - score_a  # negative => Team A favored
    total = score_a + score_b

    # 5. Win probability for Team A
    if score_a <= 0 and score_b <= 0:
        # Degenerate case (shouldn't really happen, but just in case)
        win_prob_a = 0.5
    else:
        win_prob_a = (score_a**x) / (score_a**x + score_b**x)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "location": location,
        "score_a": score_a,
        "score_b": score_b,
        "spread": spread,
        "total": total,
        "win_prob_a": win_prob_a,
    }


def map_location_code(loc_code: str) -> str:
    """
    Map schedule CSV location codes to 'Home'/'Away'/'Neutral'.

    Expected inputs: 'H', 'A', 'N' (case-insensitive), or already full words.
    """
    if not isinstance(loc_code, str):
        return "Neutral"

    code = loc_code.strip().upper()
    if code == "H":
        return "Home"
    elif code == "A":
        return "Away"
    elif code == "N":
        return "Neutral"

    # If user gives 'Home', 'Away', 'Neutral' directly
    code_lower = loc_code.strip().lower()
    if code_lower in ["home", "away", "neutral"]:
        return loc_code.capitalize()

    # Fallback
    return "Neutral"


def simulate_schedule(
    team_name: str,
    schedule_file: str,
    ratings_file: str = RATINGS_FILE,
    output_file: str | None = None,
):
    """
    Simulate a full schedule for a given team and write a text report.

    schedule_file format (CSV):
        Opponent,Location
        Team1,H
        Team2,A
        Team3,N
        ...

    Location is from the perspective of team_name:
        H = home, A = away, N = neutral
    """
    # Load ratings and schedule
    df_ratings = load_ratings(ratings_file)
    schedule = pd.read_csv(schedule_file)

    if "Opponent" not in schedule.columns or "Location" not in schedule.columns:
        raise ValueError("Schedule CSV must have columns: 'Opponent' and 'Location'.")

    # Default output filename
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(schedule_file))[0]
        safe_team = team_name.replace(" ", "_")
        output_file = f"{safe_team}_{base_name}_predictions.txt"

    games_results = []

    for _, row in schedule.iterrows():
        opp = str(row["Opponent"]).strip()
        loc_code = row["Location"]

        loc_str = map_location_code(loc_code)

        try:
            result = predict_matchup_from_df(team_name, opp, loc_str, df_ratings)
        except KeyError as e:
            # If opponent missing from ratings, record a note and skip
            games_results.append(
                {
                    "opponent": opp,
                    "location": loc_str,
                    "error": str(e),
                }
            )
            continue

        games_results.append(
            {
                "opponent": opp,
                "location": loc_str,
                "score_a": result["score_a"],
                "score_b": result["score_b"],
                "spread": result["spread"],
                "total": result["total"],
                "win_prob_a": result["win_prob_a"],
            }
        )

    # Compute record predictions
    predicted_wins = 0
    expected_wins = 0.0
    total_games = 0

    for g in games_results:
        if "error" in g:
            continue  # don't count missing-rating games in record
        total_games += 1
        wp = g["win_prob_a"]
        expected_wins += wp
        if wp >= 0.5:
            predicted_wins += 1

    predicted_losses = total_games - predicted_wins

    # Write output file
    with open(output_file, "w") as f:
        f.write(f"Team: {team_name}\n")
        f.write(f"Schedule file: {schedule_file}\n")
        f.write("-" * 50 + "\n\n")

        for i, g in enumerate(games_results, start=1):
            opp = g["opponent"]
            loc_str = g["location"]

            if "error" in g:
                f.write(f"Game {i}: vs {opp} ({loc_str})\n")
                f.write(f"  ERROR: {g['error']}\n\n")
                continue

            score_a = g["score_a"]
            score_b = g["score_b"]
            spread = g["spread"]
            total_pts = g["total"]
            win_prob = g["win_prob_a"]

            result_letter = "W" if win_prob >= 0.5 else "L"

            f.write(f"Game {i}: {team_name} vs {opp} ({loc_str})\n")
            f.write(
                f"  Projected Score: {team_name} {score_a:.1f} - {opp} {score_b:.1f}\n"
            )
            f.write(f"  Spread (from {team_name} POV): {spread:+.1f}\n")
            f.write(f"  Total: {total_pts:.1f}\n")
            f.write(f"  Win Probability ({team_name}): {win_prob*100:.1f}%\n")
            f.write(f"  Predicted Result: {result_letter}\n\n")

        f.write("-" * 50 + "\n")
        f.write(f"Total games with ratings: {total_games}\n")
        f.write(
            f"Expected Wins: {expected_wins:.1f} / {total_games} "
            f"({expected_wins/total_games*100:.1f}% win rate)\n"
            if total_games > 0
            else "Expected Wins: N/A (no rated games)\n"
        )
        f.write(
            f"Predicted Record (W/L using 50% threshold): {predicted_wins}-{predicted_losses}\n"
        )

    print(f"\nFinished! Predictions written to: {output_file}")
    if total_games > 0:
        print(
            f"Predicted record: {predicted_wins}-{predicted_losses} "
            f"(Expected wins: {expected_wins:.1f})"
        )


if __name__ == "__main__":
    team = input("Team name: ").strip()
    sched_file = "schedule.csv"

    simulate_schedule(team, sched_file)
