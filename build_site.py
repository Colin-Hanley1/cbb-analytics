import pandas as pd
import subprocess
import datetime
import os
import json
import urllib.parse
from scipy.stats import norm

# --- CRITICAL IMPORTS ---
import scrape_cbb          
# ------------------------

# --- CONFIGURATION ---
SCRAPER_SCRIPT = "scrape_cbb.py" 
RATINGS_SCRIPT = "adj.py"
OUTPUT_INDEX = "index.html"
OUTPUT_MATCHUP = "matchup.html"
OUTPUT_SCHEDULE = "schedule.html"
OUTPUT_TEAM_DATA = "teams_data.js"
CONFERENCE_FILE = "cbb_conferences.csv"
WEIGHTS_FILE = "model_weights.json"

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

def run_script(script_name):
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
    except:
        df_rat['Pure_AdjEM'] = 0.0

    try:
        df_conf = pd.read_csv(CONFERENCE_FILE)
        df_conf['Team'] = df_conf['Team'].replace(NAME_MAPPING)
    except FileNotFoundError:
        df_conf = None

    return df_rat, df_sco, df_conf

def get_nav_html(active_page):
    links = [("index.html", "Rankings"), ("schedule.html", "Today's Games"), ("matchup.html", "Matchup Simulator"), ("bracketology.html", "Bracketology")]
    
    desktop_html = '<div class="hidden md:block"><div class="ml-10 flex items-baseline space-x-4">'
    mobile_html = '<div class="md:hidden" id="mobile-menu"><div class="px-2 pt-2 pb-3 space-y-1 sm:px-3">'
    
    for url, label in links:
        is_active = (active_page == 'index' and url == 'index.html') or \
                    (active_page == 'matchup' and url == 'matchup.html') or \
                    (active_page == 'schedule' and url == 'schedule.html')
        
        d_cls = "bg-slate-900 text-white" if is_active else "text-gray-300 hover:bg-slate-700 hover:text-white"
        desktop_html += f'<a href="{url}" class="{d_cls} px-3 py-2 rounded-md text-sm font-medium transition-colors">{label}</a>'
        
        m_cls = "bg-slate-900 text-white" if is_active else "text-gray-300 hover:bg-slate-700 hover:text-white"
        mobile_html += f'<a href="{url}" class="{m_cls} block px-3 py-2 rounded-md text-base font-medium">{label}</a>'
        
    desktop_html += '</div></div>'
    mobile_html += '</div></div>'
    
    return desktop_html, mobile_html

def generate_teams_js(df_ratings, df_scores):
    print("Generating Team Data JS (SPA)...")
    teams_dict = {}
    
    em_col = 'AdjEM' if 'AdjEM' in df_ratings.columns else 'Blended_AdjEM'
    ratings_map = df_ratings.set_index('Team')[em_col].to_dict()
    
    # Calculate Bubble Threshold
    sorted_df = df_ratings.sort_values(em_col, ascending=False).reset_index(drop=True)
    start_idx = 44 if len(sorted_df) > 50 else max(0, len(sorted_df) - 6)
    end_idx = 50 if len(sorted_df) > 50 else len(sorted_df)
    bubble_adjem = sorted_df.iloc[start_idx:end_idx][em_col].mean()
    print(f"Bubble AdjEM Threshold: {bubble_adjem:.2f}")
    
    if df_scores is not None and not df_scores.empty:
        pts_col = 'pts' if 'pts' in df_scores.columns else 'points'
        
        if 'date' in df_scores.columns:
             df_scores['dt_temp'] = pd.to_datetime(df_scores['date'])
             df_scores = df_scores.sort_values('dt_temp', ascending=False)
             df_scores = df_scores.drop(columns=['dt_temp'])
        
        df_scores = df_scores.drop_duplicates(subset=['date', 'team', 'opponent'], keep='first')
        opp_df = df_scores[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
        opp_df = opp_df.drop_duplicates(subset=['date', 'opponent'], keep='first')
        full_scores = df_scores.merge(opp_df, on=['date', 'opponent'], how='left')
        
        if 'date' in full_scores.columns:
            full_scores['date'] = pd.to_datetime(full_scores['date'])
    else:
        full_scores = pd.DataFrame()

    for _, row in df_ratings.iterrows():
        team = row['Team']
        
        game_list = []
        if not full_scores.empty:
            games = full_scores[full_scores['team'] == team].copy()
            if not games.empty:
                games = games.sort_values('date', ascending=False)
                
                for _, g in games.iterrows():
                    opp = g['opponent']
                    opp_pt = g['opp_pts']
                    if pd.isna(opp_pt): continue
                    
                    loc_val = g.get('location_value', 0)
                    loc_str = "H" if loc_val == 1 else ("A" if loc_val == -1 else "N")
                    
                    pts = g[pts_col]
                    margin = pts - opp_pt
                    res = "W" if margin > 0 else "L"
                    
                    opp_rating = ratings_map.get(opp, 0.0)
                    hfa_adj = 3.2 if loc_val == 1 else (-3.2 if loc_val == -1 else 0)
                    exp_margin = (row[em_col] - opp_rating) + hfa_adj
                    
                    poss = g.get('possessions', 70)
                    if poss == 0: poss = 70
                    act_eff = (margin / poss) * 100
                    val_add = act_eff - exp_margin
                    
                    bubble_margin = (bubble_adjem - opp_rating) + hfa_adj
                    try:
                        exp_win_bubble = norm.cdf(bubble_margin / 11.0)
                    except:
                        exp_win_bubble = 1 / (1 + 10 ** (-bubble_margin * 0.027))

                    actual_win = 1.0 if res == "W" else 0.0
                    wab_game = actual_win - exp_win_bubble
                    
                    game_list.append({
                        'date': g['date'].strftime('%m/%d'),
                        'opp': opp,
                        'res': res,
                        'loc': loc_str,
                        'score': f"{int(pts)}-{int(opp_pt)}",
                        'val': round(val_add, 1),
                        'wab': round(wab_game, 2)
                    })
        
        teams_dict[team] = {
            'rank': int(row['Rank']),
            'adjem': round(row[em_col], 2),
            'adjo': round(row['AdjO'], 1),
            'adjd': round(row['AdjD'], 1),
            'adjt': round(row['AdjT'], 1),
            'wab_total': round(row.get('WAB', 0.0), 2),
            'consistency': round(row.get('Consistency', 50.0), 1),
            'games': game_list
        }
    
    js_content = f"const TEAMS_DATA = {json.dumps(teams_dict)};"
    with open(OUTPUT_TEAM_DATA, "w") as f:
        f.write(js_content)
    print(f"Generated {OUTPUT_TEAM_DATA}")

def generate_index(df, df_conf):
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
    if 'Blended_AdjEM' in df.columns: df = df.rename(columns={'Blended_AdjEM': 'AdjEM'})
    if 'Current_AdjEM' in df.columns: df = df.rename(columns={'Current_AdjEM': 'Pure_AdjEM'})
    df = df.loc[:, ~df.columns.duplicated()]
    # --- STEP 1: CALCULATE SCORE (Before Renaming) ---
    try:
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, "r") as f: weights = json.load(f)
        else:
            weights = {'ADJEM': 1.0, 'WAB': 4.0}
            
        # FIX: Ensure we use the Current (Pure) AdjEM for Selection Score if desired
        # Or check 'ADJEM' in weights vs 'AdjEM' in DF
        
        # NOTE: Your weights file uses 'ADJEM' key.
        # We assume this maps to the 'Pure_AdjEM' (Current performance) 
        # or 'AdjEM' (Blended). Usually Blended is better for projection, but you asked for Current.
        
        # Using Pure/Current AdjEM as requested:
        df['ADJEM'] = df['Pure_AdjEM'] 
        
        df['Selection_Score'] = 0.0
        for feat, w in weights.items():
            if feat in df.columns:
                df['Selection_Score'] += df[feat] * w
    except Exception as e:
        print(f"Warning: Selection Score calc failed ({e}). Defaulting to 0.")
        df['Selection_Score'] = 0.0

    # --- STEP 2: RENAME COLUMNS ---
    # (Already renamed above for JSON compatibility)

    # --- STEP 3: DEDUPLICATE COLUMNS ---
    df = df.loc[:, ~df.columns.duplicated()]

    # Ranks
    df['Rank_AdjO'] = df['AdjO'].rank(ascending=False, method='min').astype(int)
    df['Rank_AdjD'] = df['AdjD'].rank(ascending=True, method='min').astype(int) 
    df['Rank_AdjT'] = df['AdjT'].rank(ascending=False, method='min').astype(int)
    df['Rank_Cons'] = df['Consistency'].rank(ascending=False, method='min').astype(int)
    df['Rank_WAB'] = df['WAB'].rank(ascending=False, method='min').astype(int)
    df['Rank_SLS'] = df['Selection_Score'].rank(ascending=False, method='min').astype(int)
    if df_conf is not None:
        df = df.merge(df_conf, on='Team', how='left')
        df['Conference'] = df['Conference'].fillna('Unknown')
    else:
        df['Conference'] = '-'
        
    conf_list = sorted([c for c in df['Conference'].unique() if c and c != 'Unknown' and c != '-'])
    conf_json = json.dumps(conf_list)
    df = df.loc[:, ~df.columns.duplicated()]

    cols = ['Rank', 'Team', 'Conference', 'Link', 'AdjEM', 'Pure_AdjEM', 'WAB', 'Consistency', 
            'Selection_Score', 'AdjO', 'AdjD', 'AdjT', 'Rank_AdjO', 'Rank_AdjD', 'Rank_AdjT', 'Rank_Cons', 'Rank_WAB', 'Rank_SLS']
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

def generate_matchup(df):
    print("Generating Matchup...")
    teams_data = {}
    em_col = 'Blended_AdjEM' if 'Blended_AdjEM' in df.columns else 'AdjEM'
    
    # --- NEW: CALCULATE BUBBLE FOR MATCHUP INJECTION ---
    sorted_df = df.sort_values(em_col, ascending=False).reset_index(drop=True)
    start_idx = 44 if len(sorted_df) > 50 else max(0, len(sorted_df) - 6)
    end_idx = 50 if len(sorted_df) > 50 else len(sorted_df)
    bubble_adjem = sorted_df.iloc[start_idx:end_idx][em_col].mean()
    # ---------------------------------------------------

    for _, row in df.sort_values('Team').iterrows():
        teams_data[row['Team']] = {
            'AdjO': row['AdjO'], 'AdjD': row['AdjD'], 'AdjT': row['AdjT'], 'AdjEM': row[em_col]
        }
    teams_json = json.dumps(teams_data)

    try:
        with open("matchup_template.html", "r") as f:
            template = f.read()
            
        nav_d, nav_m = get_nav_html('matchup')
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        html = template.replace("{{TEAMS_JSON}}", teams_json)
        # --- NEW: INJECT BUBBLE VAR ---
        html = html.replace("{{BUBBLE_ADJEM}}", f"{bubble_adjem:.2f}") 
        
        html = html.replace("{{LAST_UPDATED}}", now)
        html = html.replace("{{NAV_DESKTOP}}", nav_d)
        html = html.replace("{{NAV_MOBILE}}", nav_m)

        with open(OUTPUT_MATCHUP, "w") as f:
            f.write(html)
        print(f"Generated {OUTPUT_MATCHUP}")
    except FileNotFoundError:
        print("Warning: matchup_template.html not found.")

def generate_schedule():
    print("Generating Schedule Page...")
    
    # 1. Instantiate Scraper (Fixed)
    my_scraper = scrape_cbb.CBBScraper()
    games_list = my_scraper.get_todays_schedule()
    
    # 2. Serialize to JSON
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
    
    # Run bracket generation first so we can link to it if needed

    df_ratings, df_scores, df_conf = get_data()

    if df_ratings is not None:
        generate_teams_js(df_ratings, df_scores)
        generate_index(df_ratings, df_conf)
        generate_matchup(df_ratings)
        generate_schedule()