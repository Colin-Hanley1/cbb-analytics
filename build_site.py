import pandas as pd
import numpy as np
import subprocess
import datetime
import os
import json
import urllib.parse

# --- CRITICAL IMPORTS ---
import scrape_cbb
# ------------------------

# --- CONFIGURATION ---
SCRAPER_SCRIPT = "scrape_cbb.py"
BRACKET_SCRIPT = "bracket.py"
RATINGS_SCRIPT = "adj.py"

OUTPUT_INDEX = "index.html"
OUTPUT_MATCHUP = "matchup.html"
OUTPUT_SCHEDULE = "schedule.html"
OUTPUT_TEAM_DATA = "teams_data.js"

CONFERENCE_FILE = "cbb_conferences.csv"
WEIGHTS_FILE = "model_weights.json"
SCHEDULE_CSV = "schedule.csv"  # season schedule file

OUTPUT_CONFERENCES = "conferences.html"
OUTPUT_CONF_DATA = "conferences_data.js"
CONFERENCES_TEMPLATE = "conferences_template.html"


# --- RQ / WAB CONSTANTS (MATCH adj.py) ---
AVG_EFF = 106.0
AVG_TEMPO = 68.5
HFA_VAL = 3.2
PYTH_EXP = 11.5

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

# =========================
# Helpers: scripts + data
# =========================
def run_script(script_name: str) -> None:
    print(f"--- Running {script_name} ---")
    try:
        subprocess.run(["python3", script_name], check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

def get_data():
    try:
        df_rat = pd.read_csv("cbb_ratings.csv")
    except FileNotFoundError:
        print("Error: cbb_ratings.csv not found.")
        return None, None, None

    try:
        df_sco = pd.read_csv("cbb_scores.csv")
    except FileNotFoundError:
        print("Error: cbb_scores.csv not found.")
        return df_rat, None, None

    # Optional: Load Pure Ratings if available
    try:
        df_pure = pd.read_csv("kenpom_pure.csv")
        pure_map = df_pure.set_index('Team')['AdjEM'].to_dict()
        df_rat['Pure_AdjEM'] = df_rat['Team'].map(pure_map).fillna(0.0)
    except Exception:
        df_rat['Pure_AdjEM'] = 0.0

    try:
        df_conf = pd.read_csv(CONFERENCE_FILE)
        df_conf['Team'] = df_conf['Team'].replace(NAME_MAPPING)
        # Expect columns: Team, Conference
        if "Conference" not in df_conf.columns:
            print("Warning: cbb_conferences.csv missing 'Conference' column.")
            df_conf = None
    except FileNotFoundError:
        df_conf = None

    return df_rat, df_sco, df_conf

def get_nav_html(active_page: str):
    links = [
        ("index.html", "Rankings"),
        ("schedule.html", "Today's Games"),
        ("matchup.html", "Matchup Simulator"),
        ("bracketology.html", "Bracketology"),
        ("conferences.html", "Conferences")
    ]

    desktop_html = '<div class="hidden md:block"><div class="ml-10 flex items-baseline space-x-4">'
    mobile_html  = '<div class="md:hidden" id="mobile-menu"><div class="px-2 pt-2 pb-3 space-y-1 sm:px-3">'

    for url, label in links:
        is_active = (active_page == 'index' and url == 'index.html') or \
                    (active_page == 'matchup' and url == 'matchup.html') or \
                    (active_page == 'schedule' and url == 'schedule.html')

        d_cls = "bg-slate-900 text-white" if is_active else "text-gray-300 hover:bg-slate-700 hover:text-white"
        desktop_html += f'<a href="{url}" class="{d_cls} px-3 py-2 rounded-md text-sm font-medium transition-colors">{label}</a>'

        m_cls = "bg-slate-900 text-white" if is_active else "text-gray-300 hover:bg-slate-700 hover:text-white"
        mobile_html += f'<a href="{url}" class="{m_cls} block px-3 py-2 rounded-md text-base font-medium">{label}</a>'

    desktop_html += '</div></div>'
    mobile_html  += '</div></div>'

    return desktop_html, mobile_html

# ==========================================
# schedule.csv -> future predictions (+ conf)
# ==========================================
def load_schedule_csv(schedule_csv: str = SCHEDULE_CSV) -> pd.DataFrame:
    """
    Expected schedule.csv columns (minimum):
      - date (YYYY-MM-DD or parseable)
      - team
      - opponent
      - location_value   (1=home, -1=away, 0=neutral)

    Returns a cleaned DataFrame (or empty DF if missing).
    """
    if not os.path.exists(schedule_csv):
        print(f"Warning: {schedule_csv} not found. No future schedules will be attached.")
        return pd.DataFrame()

    try:
        sch = pd.read_csv(schedule_csv)
    except Exception as e:
        print(f"Warning: Could not read {schedule_csv} ({e}).")
        return pd.DataFrame()

    if sch.empty:
        return sch

    # Validate required columns
    req = {"date", "team", "opponent"}
    missing = req - set(sch.columns)
    if missing:
        print(f"Warning: {schedule_csv} missing columns: {sorted(list(missing))}.")
        return pd.DataFrame()

    # Normalize names to match site convention
    for c in ["team", "opponent"]:
        sch[c] = sch[c].replace(NAME_MAPPING)

    # Parse dates
    sch["date"] = pd.to_datetime(sch["date"], errors="coerce")
    sch = sch.dropna(subset=["date", "team", "opponent"])

    # Ensure location_value exists
    if "location_value" not in sch.columns:
        sch["location_value"] = 0

    sch["location_value"] = pd.to_numeric(sch["location_value"], errors="coerce").fillna(0).astype(int)
    sch["location_value"] = sch["location_value"].clip(-1, 1)

    return sch

def _predict_game_from_team_perspective(team: str, opp: str, loc_val: int, stats_map: dict):
    """
    stats_map: dict Team -> {AdjO, AdjD, AdjT}
    loc_val: 1 home, -1 away, 0 neutral
    """
    if team not in stats_map or opp not in stats_map:
        return None

    t = stats_map[team]
    o = stats_map[opp]

    hfa_adj = HFA_VAL if loc_val == 1 else (-HFA_VAL if loc_val == -1 else 0.0)

    exp_eff_team = t["AdjO"] + o["AdjD"] - AVG_EFF + hfa_adj
    exp_eff_opp  = o["AdjO"] + t["AdjD"] - AVG_EFF - hfa_adj
    exp_tempo = t["AdjT"] + o["AdjT"] - AVG_TEMPO

    team_pts = (exp_eff_team * exp_tempo) / 100.0
    opp_pts  = (exp_eff_opp  * exp_tempo) / 100.0

    if team_pts <= 0 or opp_pts <= 0:
        p = 1.0 if team_pts > opp_pts else 0.0
    else:
        p = (team_pts ** PYTH_EXP) / ((team_pts ** PYTH_EXP) + (opp_pts ** PYTH_EXP))

    loc_str = "H" if loc_val == 1 else ("A" if loc_val == -1 else "N")

    return {
        "opp": opp,
        "loc": loc_str,
        "p_win": round(float(p), 3),
        "proj_for": round(float(team_pts), 1),
        "proj_against": round(float(opp_pts), 1),
        "proj": f"{team_pts:.1f}-{opp_pts:.1f}",
        "spread": round(float(team_pts - opp_pts), 1),
    }

def build_future_schedule_map(
    df_ratings: pd.DataFrame,
    conf_map: dict,
    schedule_csv: str = SCHEDULE_CSV
) -> dict:
    """
    Returns: future_map: dict team -> list[dict]
    Each entry includes:
      date (MM/DD), opp, loc, p_win, proj, spread, proj_for, proj_against,
      opp_conf, is_conf
    """
    sch = load_schedule_csv(schedule_csv)
    if sch.empty:
        return {}

    # Future games from today's local date
    today = pd.Timestamp(datetime.datetime.now().date())
    sch = sch[sch["date"] >= today].copy()
    if sch.empty:
        return {}

    needed = {"AdjO", "AdjD", "AdjT"}
    if not needed.issubset(set(df_ratings.columns)):
        print("Warning: Ratings missing AdjO/AdjD/AdjT; cannot build future predictions.")
        return {}

    stats_map = df_ratings.set_index("Team")[["AdjO", "AdjD", "AdjT"]].to_dict("index")

    future_map = {}
    sch = sch.sort_values(["team", "date", "opponent"])

    for _, r in sch.iterrows():
        team = str(r["team"])
        opp = str(r["opponent"])
        loc_val = int(r["location_value"])

        pred = _predict_game_from_team_perspective(team, opp, loc_val, stats_map)
        if pred is None:
            continue

        team_conf = conf_map.get(team, "Unknown")
        opp_conf = conf_map.get(opp, "Unknown")
        is_conf = (team_conf != "Unknown") and (team_conf == opp_conf)

        pred["date"] = r["date"].strftime("%m/%d")
        pred["opp_conf"] = opp_conf
        pred["is_conf"] = bool(is_conf)

        future_map.setdefault(team, []).append(pred)

    return future_map

# ==========================================
# Site generation
# ==========================================
def generate_teams_js(df_ratings: pd.DataFrame, df_scores: pd.DataFrame, df_conf: pd.DataFrame):
    print("Generating Team Data JS (SPA)...")
    teams_dict = {}

    # Attach conference if provided
    if df_conf is not None and not df_conf.empty:
        # protect against accidental duplicate merges if caller passes an already-merged df
        if "Conference" not in df_ratings.columns:
            df_ratings = df_ratings.merge(df_conf[["Team", "Conference"]], on="Team", how="left")
        df_ratings["Conference"] = df_ratings["Conference"].fillna("Unknown")
    else:
        if "Conference" not in df_ratings.columns:
            df_ratings["Conference"] = "Unknown"

    conf_map = df_ratings.set_index("Team")["Conference"].to_dict()

    em_col = 'AdjEM' if 'AdjEM' in df_ratings.columns else 'Blended_AdjEM'

    # --- Bubble Team Profile (MATCH adj.py RQ LOGIC) ---
    sorted_df = df_ratings.sort_values(em_col, ascending=False).reset_index(drop=True)
    start_idx = 44 if len(sorted_df) > 50 else max(0, len(sorted_df) - 6)
    end_idx = 50 if len(sorted_df) > 50 else len(sorted_df)
    bubble_subset = sorted_df.iloc[start_idx:end_idx]

    bubble_profile = {
        "AdjO": float(bubble_subset["AdjO"].mean()),
        "AdjD": float(bubble_subset["AdjD"].mean()),
        "AdjT": float(bubble_subset["AdjT"].mean()),
        "AdjEM": float(bubble_subset[em_col].mean())
    }
    print(
        f"Bubble Profile: AdjO {bubble_profile['AdjO']:.1f} | "
        f"AdjD {bubble_profile['AdjD']:.1f} | Tempo {bubble_profile['AdjT']:.1f} | "
        f"AdjEM {bubble_profile['AdjEM']:.2f}"
    )

    # For expected margin display (val_add)
    ratings_map = df_ratings.set_index('Team')[em_col].to_dict()

    # For RQ-consistent per-game win prob
    opp_stats_map = df_ratings.set_index("Team")[["AdjO", "AdjD", "AdjT"]].to_dict("index")

    # --- NEW: future schedule predictions from schedule.csv (+ conference flags) ---
    future_map = build_future_schedule_map(df_ratings, conf_map=conf_map, schedule_csv=SCHEDULE_CSV)

    # --- Prepare full scores with opponent points merged ---
    if df_scores is not None and not df_scores.empty:
        pts_col = 'pts' if 'pts' in df_scores.columns else 'points'

        if 'date' in df_scores.columns:
            df_scores['dt_temp'] = pd.to_datetime(df_scores['date'], errors="coerce")
            df_scores = df_scores.dropna(subset=["dt_temp"]).sort_values('dt_temp', ascending=False)
            df_scores = df_scores.drop(columns=['dt_temp'])

        # De-dupe obvious duplicates
        df_scores = df_scores.drop_duplicates(subset=['date', 'team', 'opponent'], keep='first')

        opp_df = df_scores[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
        opp_df = opp_df.drop_duplicates(subset=['date', 'opponent'], keep='first')
        full_scores = df_scores.merge(opp_df, on=['date', 'opponent'], how='left')

        full_scores['date'] = pd.to_datetime(full_scores['date'], errors="coerce")
        full_scores = full_scores.dropna(subset=["date"])

        # Canonicalize to avoid double-counting (team-by-team view)
        full_scores['game_id'] = (
            full_scores['date'].dt.normalize().astype(str) + "_" +
            full_scores.apply(lambda r: "_".join(sorted([str(r['team']), str(r['opponent'])])), axis=1)
        )
        full_scores = (
            full_scores.sort_values(['team', 'game_id'])
            .groupby(['team', 'game_id'], as_index=False)
            .first()
        )
    else:
        pts_col = 'pts'
        full_scores = pd.DataFrame()

    # --- Build team objects ---
    for _, row in df_ratings.iterrows():
        team = row['Team']
        team_conf = conf_map.get(team, "Unknown")

        game_list = []
        if not full_scores.empty:
            games = full_scores[full_scores['team'] == team].copy()
            if not games.empty:
                games = games.sort_values('date', ascending=False)

                for _, g in games.iterrows():
                    opp = g['opponent']
                    opp_pt = g.get('opp_pts', None)
                    if pd.isna(opp_pt):
                        continue

                    loc_val = int(g.get('location_value', 0))
                    loc_str = "H" if loc_val == 1 else ("A" if loc_val == -1 else "N")

                    pts = float(g[pts_col])
                    margin = pts - float(opp_pt)
                    res = "W" if margin > 0 else "L"

                    # Existing "val" computation (margin/100 poss vs EM expectation)
                    opp_rating = float(ratings_map.get(opp, 0.0))
                    hfa_adj_em = HFA_VAL if loc_val == 1 else (-HFA_VAL if loc_val == -1 else 0.0)
                    exp_margin_em = (float(row[em_col]) - opp_rating) + hfa_adj_em

                    poss = g.get('possessions', 70)
                    try:
                        poss = float(poss)
                    except Exception:
                        poss = 70.0
                    if poss == 0:
                        poss = 70.0

                    act_eff = (margin / poss) * 100.0
                    val_add = act_eff - exp_margin_em

                    # RQ-consistent per-game residual (actual_win - p_bubble)
                    opp_stats = opp_stats_map.get(opp, {"AdjO": 95.0, "AdjD": 115.0, "AdjT": AVG_TEMPO})

                    if loc_val == 1:
                        hfa = HFA_VAL
                    elif loc_val == -1:
                        hfa = -HFA_VAL
                    else:
                        hfa = 0.0

                    exp_eff_bubble = bubble_profile["AdjO"] + opp_stats["AdjD"] - AVG_EFF + hfa
                    exp_eff_opp    = opp_stats["AdjO"] + bubble_profile["AdjD"] - AVG_EFF - hfa
                    exp_tempo = bubble_profile["AdjT"] + opp_stats["AdjT"] - AVG_TEMPO

                    score_bubble = (exp_eff_bubble * exp_tempo) / 100.0
                    score_opp    = (exp_eff_opp * exp_tempo) / 100.0

                    if score_bubble <= 0 or score_opp <= 0:
                        p_bubble = 0.0 if score_opp > score_bubble else 1.0
                    else:
                        p_bubble = (score_bubble ** PYTH_EXP) / ((score_bubble ** PYTH_EXP) + (score_opp ** PYTH_EXP))

                    actual_win = 1.0 if res == "W" else 0.0
                    wab_game = actual_win - p_bubble

                    # Past-game conference tagging
                    opp_conf = conf_map.get(opp, "Unknown")
                    is_conf = (team_conf != "Unknown") and (team_conf == opp_conf)

                    # date formatting guard
                    if isinstance(g.get('date', None), pd.Timestamp):
                        date_str = g['date'].strftime('%m/%d')
                    else:
                        try:
                            date_str = pd.to_datetime(g['date']).strftime('%m/%d')
                        except Exception:
                            date_str = str(g.get('date', ''))

                    game_list.append({
                        'date': date_str,
                        'opp': opp,
                        'res': res,
                        'loc': loc_str,
                        'score': f"{int(pts)}-{int(float(opp_pt))}",
                        'val': round(float(val_add), 1),
                        'rq': round(float(wab_game), 2),
                        'opp_conf': opp_conf,
                        'is_conf': bool(is_conf),
                    })

        teams_dict[team] = {
            'rank': int(row['Rank']) if 'Rank' in row else int(row.get('rank', 0)),
            'adjem': round(float(row[em_col]), 2),
            'adjo': round(float(row['AdjO']), 1),
            'adjd': round(float(row['AdjD']), 1),
            'adjt': round(float(row['AdjT']), 1),
            'wab_total': round(float(row.get('RQ', 0.0)), 2),
            'consistency': round(float(row.get('Consistency', 50.0)), 1),
            'sel_score': round(float(row.get('Selection_Score', 0.0)), 1),
            'sa': round(float(row.get('StrengthAdjust', 0.0)), 2),
            'sos': round(float(row.get('SOS', 0.0)), 2),
            'q1_w': int(row.get('Q1_W', 0)), 'q1_l': int(row.get('Q1_L', 0)),
            'q2_w': int(row.get('Q2_W', 0)), 'q2_l': int(row.get('Q2_L', 0)),
            'q3_w': int(row.get('Q3_W', 0)), 'q3_l': int(row.get('Q3_L', 0)),
            'q4_w': int(row.get('Q4_W', 0)), 'q4_l': int(row.get('Q4_L', 0)),
            't100_w': int(row.get('T100_W', 0)), 't100_l': int(row.get('T100_L', 0)),
            'conf': team_conf,
            'rkEM': int(row.get('Rank', 0)),
            'rkRQ': int(row.get('Rank_RQ', 0)),
            'games': game_list,
            'future': future_map.get(team, [])  # future schedule + predictions (+ is_conf)
        }

    js_content = f"const TEAMS_DATA = {json.dumps(teams_dict)};"
    with open(OUTPUT_TEAM_DATA, "w") as f:
        f.write(js_content)
    print(f"Generated {OUTPUT_TEAM_DATA}")

def generate_index(df: pd.DataFrame, df_conf: pd.DataFrame):
    print("Generating Index...")
    try:
        with open("template.html", "r") as f:
            template = f.read()
    except FileNotFoundError:
        print("Error: template.html not found.")
        return

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nav_d, nav_m = get_nav_html('index')

    df['Link'] = df['Team'].apply(lambda x: f"team.html?q={urllib.parse.quote(x)}")

    # Rename columns (so index page expects AdjEM/Pure_AdjEM)
    if 'Blended_AdjEM' in df.columns:
        df = df.rename(columns={'Blended_AdjEM': 'AdjEM'})
    if 'Current_AdjEM' in df.columns:
        df = df.rename(columns={'Current_AdjEM': 'Pure_AdjEM'})

    # Deduplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Selection Score
    try:
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, "r") as f:
                weights = json.load(f)
        else:
            weights = {'ADJEM': 1.0, 'RQ': 4.0}

        df['ADJEM'] = df['Pure_AdjEM'] if 'Pure_AdjEM' in df.columns else df.get('AdjEM', 0.0)

        df['Selection_Score'] = 0.0
        for feat, w in weights.items():
            if feat in df.columns:
                df['Selection_Score'] += df[feat] * w
    except Exception as e:
        print(f"Warning: Selection Score calc failed ({e}). Defaulting to 0.")
        df['Selection_Score'] = 0.0

    df['Rank_AdjO'] = df['AdjO'].rank(ascending=False, method='min').astype(int)
    df['Rank_AdjD'] = df['AdjD'].rank(ascending=True, method='min').astype(int)
    df['Rank_AdjT'] = df['AdjT'].rank(ascending=False, method='min').astype(int)

    if 'Consistency' not in df.columns:
        df['Consistency'] = 50.0

    df['Rank_Consistency'] = df['Consistency'].rank(ascending=False, method='min').astype(int)
    df['Rank_RQ'] = df['RQ'].rank(ascending=False, method='min').astype(int) if 'RQ' in df.columns else 0
    df['Rank_SLS'] = df['Selection_Score'].rank(ascending=False, method='min').astype(int)
    df['Rank_SOS'] = df['SOS'].rank(ascending=False, method='min').astype(int) if 'SOS' in df.columns else 0
    df['Rank_SA'] = df['StrengthAdjust'].rank(ascending=False, method='min').astype(int) if 'StrengthAdjust' in df.columns else 0
    df['Rank_awq'] = df['RQ_Win_Avg'].rank(ascending=False, method='min').astype(int) if 'RQ_Win_Avg' in df.columns else 0
    df['Rank_alq'] = df['RQ_Loss_Avg'].rank(ascending=False, method='min').astype(int) if 'RQ_Loss_Avg' in df.columns else 0
    df['Rank_rdadj'] = df['Venue_AdjEM_AN_minus_Home'].rank(ascending=False, method='min').astype(int) if 'Venue_AdjEM_AN_minus_Home' in df.columns else 0

    if df_conf is not None and not df_conf.empty:
        if "Conference" not in df.columns:
            df = df.merge(df_conf[["Team", "Conference"]], on='Team', how='left')
        df['Conference'] = df['Conference'].fillna('Unknown')
    else:
        if "Conference" not in df.columns:
            df['Conference'] = '-'

    conf_list = sorted([c for c in df['Conference'].unique() if c and c != 'Unknown' and c != '-'])
    conf_json = json.dumps(conf_list)

    df = df.loc[:, ~df.columns.duplicated()]

    cols = [
        'Rank', 'Team', 'Conference', 'Link', 'AdjEM', 'Pure_AdjEM', 'RQ', 'RQ_Win_Avg', 'RQ_Loss_Avg',
        'Rank_awq', 'Rank_alq', 'Consistency', 'Selection_Score', 'AdjO', 'AdjD', 'AdjT',
        'StrengthAdjust', 'Rank_SA', 'Rank_AdjO', 'Rank_AdjD', 'Rank_AdjT', 'Rank_RQ', 'Rank_SLS',
        'Rank_Consistency', 'SOS', 'Q1_W', 'T100_W', 'Rank_SOS',
        'Venue_AdjEM_AN_minus_Home', 'Rank_rdadj'
    ]
    cols = [c for c in cols if c in df.columns]
    rankings_json = df[cols].to_json(orient='records')

    html = template.replace("{{RANKINGS_JSON}}", rankings_json)
    html = html.replace("{{CONF_JSON}}", conf_json)
    html = html.replace("{{LAST_UPDATED}}", now)
    html = html.replace("{{NAV_DESKTOP}}", nav_d)
    html = html.replace("{{NAV_MOBILE}}", nav_m)

    with open(OUTPUT_INDEX, "w") as f:
        f.write(html)
    print(f"Generated {OUTPUT_INDEX}")

def build_conference_pages(df_ratings: pd.DataFrame, df_scores: pd.DataFrame, df_conf: pd.DataFrame):
    """
    Creates:
      1) conferences_data.js  (conference summary + projected standings)
      2) conferences.html     (rendered page from template)
    """
    if df_conf is None or df_conf.empty:
        print("Warning: cbb_conferences.csv missing; cannot generate conferences page.")
        return

    # Normalize team names
    df_conf = df_conf.copy()
    df_conf["Team"] = df_conf["Team"].replace(NAME_MAPPING)

    df = df_ratings.copy()
    df["Team"] = df["Team"].replace(NAME_MAPPING)

    # Attach conference
    df = df.merge(df_conf[["Team", "Conference"]], on="Team", how="left")
    df["Conference"] = df["Conference"].fillna("Unknown")

    em_col = "AdjEM" if "AdjEM" in df.columns else "Blended_AdjEM"
    if em_col not in df.columns:
        print("Warning: No AdjEM/Blended_AdjEM column found; cannot generate conferences page.")
        return

    # Build maps
    team_conf = df.set_index("Team")["Conference"].to_dict()
    stats_map = df.set_index("Team")[["AdjO", "AdjD", "AdjT"]].to_dict("index")
    adjem_map = df.set_index("Team")[em_col].to_dict()

    # ----------------------------
    # (A) Conference Summary Table
    # ----------------------------
    summary = []
    for conf, g in df[df["Conference"] != "Unknown"].groupby("Conference"):
        g2 = g.dropna(subset=[em_col])
        if g2.empty:
            continue
        avg_em = float(g2[em_col].mean())
        hi = g2.sort_values(em_col, ascending=False).iloc[0]
        lo = g2.sort_values(em_col, ascending=True).iloc[0]
        summary.append({
            "conference": conf,
            "team_count": int(len(g2)),
            "avg_adjem": round(avg_em, 2),
            "highest_team": str(hi["Team"]),
            "highest_adjem": round(float(hi[em_col]), 2),
            "lowest_team": str(lo["Team"]),
            "lowest_adjem": round(float(lo[em_col]), 2),
        })

    # sort by avg strength
    summary = sorted(summary, key=lambda x: x["avg_adjem"], reverse=True)

    # ----------------------------
    # (B) Projected Conference Standings
    # ----------------------------
    # helper: determine if a team/opponent is a conference game
    def is_conf_game(team, opp):
        return team_conf.get(team) == team_conf.get(opp) and team_conf.get(team) not in (None, "Unknown")

    # Completed conference record from df_scores (actual)
    # df_scores format: date, team, opponent, pts, opp_pts merged earlier in your teams build
    actual_rec = {}  # (team) -> {"w": int, "l": int}
    if df_scores is not None and not df_scores.empty:
        sco = df_scores.copy()
        # normalize
        sco["team"] = sco["team"].replace(NAME_MAPPING)
        sco["opponent"] = sco["opponent"].replace(NAME_MAPPING)

        # Need opponent points to know W/L; in your pipeline you sometimes merge opp_pts elsewhere.
        # Here we handle both:
        pts_col = "pts" if "pts" in sco.columns else ("points" if "points" in sco.columns else None)
        opp_pts_col = "opp_pts" if "opp_pts" in sco.columns else None

        # If opp_pts missing, do a quick merge similar to generate_teams_js
        if pts_col and opp_pts_col is None:
            tmp = sco[["date", "team", pts_col]].rename(columns={"team": "opponent", pts_col: "opp_pts"})
            tmp = tmp.drop_duplicates(subset=["date", "opponent"], keep="first")
            sco = sco.merge(tmp, on=["date", "opponent"], how="left")
            opp_pts_col = "opp_pts"

        if pts_col and opp_pts_col in sco.columns:
            # date parse
            if "date" in sco.columns:
                sco["date"] = pd.to_datetime(sco["date"], errors="coerce")

            # de-dupe team/opponent/date rows
            sco = sco.drop_duplicates(subset=["date", "team", "opponent"], keep="first")

            for _, r in sco.iterrows():
                team = str(r["team"])
                opp = str(r["opponent"])
                if not is_conf_game(team, opp):
                    continue

                pts = r.get(pts_col, None)
                opp_pts = r.get(opp_pts_col, None)
                if pd.isna(pts) or pd.isna(opp_pts):
                    continue

                w = 1 if float(pts) > float(opp_pts) else 0
                actual_rec.setdefault(team, {"w": 0, "l": 0})
                if w == 1:
                    actual_rec[team]["w"] += 1
                else:
                    actual_rec[team]["l"] += 1

    # Remaining conference expected wins from schedule.csv
    sch = load_schedule_csv(SCHEDULE_CSV)
    exp_remain = {}  # team -> expected remaining conf wins
    rem_games = {}   # team -> number of remaining conf games

    if not sch.empty:
        today = pd.Timestamp(datetime.datetime.now().date())
        sch = sch[sch["date"] >= today].copy()
        sch["team"] = sch["team"].replace(NAME_MAPPING)
        sch["opponent"] = sch["opponent"].replace(NAME_MAPPING)

        for _, r in sch.iterrows():
            team = str(r["team"])
            opp = str(r["opponent"])
            if not is_conf_game(team, opp):
                continue

            loc_val = int(r.get("location_value", 0))
            pred = _predict_game_from_team_perspective(team, opp, loc_val, stats_map)
            if pred is None:
                continue

            p = float(pred["p_win"])
            exp_remain[team] = exp_remain.get(team, 0.0) + p
            rem_games[team] = rem_games.get(team, 0) + 1

    # Build standings structure per conference
    standings_by_conf = {}
    teams = df[df["Conference"] != "Unknown"][["Team", "Conference"]].drop_duplicates()

    for _, tr in teams.iterrows():
        team = str(tr["Team"])
        conf = str(tr["Conference"])

        w_a = actual_rec.get(team, {}).get("w", 0)
        l_a = actual_rec.get(team, {}).get("l", 0)
        w_exp = float(exp_remain.get(team, 0.0))
        g_rem = int(rem_games.get(team, 0))
        w_proj = w_a + w_exp

        standings_by_conf.setdefault(conf, []).append({
            "team": team,
            "adjem": round(float(adjem_map.get(team, 0.0)), 2),
            "conf_w": int(w_a),
            "conf_l": int(l_a),
            "rem_conf_games": g_rem,
            "exp_conf_wins_remaining": round(w_exp, 2),
            "proj_conf_wins": round(w_proj, 2),
            "proj_conf_losses": round((l_a + g_rem) - w_exp, 2)  # expected losses remaining = games - expected wins
        })

    # Sort each conference standings:
    # 1) projected conf wins desc
    # 2) AdjEM desc
    # 3) team name asc
    for conf, arr in standings_by_conf.items():
        standings_by_conf[conf] = sorted(
            arr,
            key=lambda x: (-x["proj_conf_wins"], -x["adjem"], x["team"])
        )

    # ----------------------------
    # Export conferences_data.js
    # ----------------------------
    payload = {
        "summary": summary,
        "standings": standings_by_conf
    }

    with open(OUTPUT_CONF_DATA, "w") as f:
        f.write("const CONFERENCES_DATA = " + json.dumps(payload) + ";")
    print(f"Generated {OUTPUT_CONF_DATA}")

    # ----------------------------
    # Render conferences.html
    # ----------------------------
    try:
        with open(CONFERENCES_TEMPLATE, "r") as f:
            template = f.read()
    except FileNotFoundError:
        print(f"Warning: {CONFERENCES_TEMPLATE} not found; only generated {OUTPUT_CONF_DATA}.")
        return

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nav_d, nav_m = get_nav_html("conferences")

    html = template.replace("{{LAST_UPDATED}}", now)
    html = html.replace("{{NAV_DESKTOP}}", nav_d)
    html = html.replace("{{NAV_MOBILE}}", nav_m)

    with open(OUTPUT_CONFERENCES, "w") as f:
        f.write(html)
    print(f"Generated {OUTPUT_CONFERENCES}")


def generate_matchup(df: pd.DataFrame):
    print("Generating Matchup...")
    teams_data = {}
    em_col = 'Blended_AdjEM' if 'Blended_AdjEM' in df.columns else ('AdjEM' if 'AdjEM' in df.columns else None)
    if em_col is None:
        print("Warning: No AdjEM column found for matchup page.")
        return

    # Bubble Threshold for Matchup JS
    sorted_df = df.sort_values(em_col, ascending=False).reset_index(drop=True)
    start_idx = 44 if len(sorted_df) > 50 else max(0, len(sorted_df) - 6)
    end_idx = 50 if len(sorted_df) > 50 else len(sorted_df)
    bubble_adjem = float(sorted_df.iloc[start_idx:end_idx][em_col].mean())

    for _, row in df.sort_values('Team').iterrows():
        teams_data[row['Team']] = {
            'AdjO': float(row['AdjO']),
            'AdjD': float(row['AdjD']),
            'AdjT': float(row['AdjT']),
            'AdjEM': float(row[em_col])
        }
    teams_json = json.dumps(teams_data)

    try:
        with open("matchup_template.html", "r") as f:
            template = f.read()

        nav_d, nav_m = get_nav_html('matchup')
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        html = template.replace("{{TEAMS_JSON}}", teams_json)
        html = html.replace("{{BUBBLE_ADJEM}}", str(round(bubble_adjem, 2)))
        html = html.replace("{{LAST_UPDATED}}", now)
        html = html.replace("{{NAV_DESKTOP}}", nav_d)
        html = html.replace("{{NAV_MOBILE}}", nav_m)

        with open(OUTPUT_MATCHUP, "w") as f:
            f.write(html)
        print(f"Generated {OUTPUT_MATCHUP}")
    except FileNotFoundError:
        print("Warning: matchup_template.html not found.")

def generate_schedule():
    """
    schedule.html shows TODAY's games based on Sports-Reference.
    (Separate from schedule.csv, which is used for per-team future schedules on team.html via teams_data.js.)
    """
    print("Generating Schedule Page...")

    my_scraper = scrape_cbb.CBBScraper()
    games_list = my_scraper.get_todays_schedule()
    games_json = json.dumps(games_list)

    try:
        with open("schedule_template.html", "r") as f:
            template = f.read()

        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        nav_d, nav_m = get_nav_html('schedule')

        html = template.replace("{{GAMES_JSON}}", games_json)
        html = html.replace("{{LAST_UPDATED}}", now)
        html = html.replace("{{NAV_DESKTOP}}", nav_d)
        html = html.replace("{{NAV_MOBILE}}", nav_m)

        with open(OUTPUT_SCHEDULE, "w") as f:
            f.write(html)
        print(f"Generated {OUTPUT_SCHEDULE}")
    except Exception as e:
        print(f"Error generating schedule: {e}")

if __name__ == "__main__":
    run_script(SCRAPER_SCRIPT)
    run_script(RATINGS_SCRIPT)
    run_script(BRACKET_SCRIPT)

    df_ratings, df_scores, df_conf = get_data()

    if df_ratings is not None:
        generate_teams_js(df_ratings, df_scores, df_conf)
        generate_index(df_ratings, df_conf)
        generate_matchup(df_ratings)
        generate_schedule()
        build_conference_pages(df_ratings, df_scores, df_conf)  # <-- NEW
