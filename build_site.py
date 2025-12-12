import pandas as pd
import subprocess
import datetime
import os
import json
import urllib.parse

# --- CONFIGURATION ---
SCRAPER_SCRIPT = "scrape_cbb.py" 
RATINGS_SCRIPT = "adj.py"
OUTPUT_INDEX = "index.html"
OUTPUT_MATCHUP = "matchup.html"
OUTPUT_TEAM_DATA = "teams_data.js"

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
        return None, None

    try:
        df_sco = pd.read_csv("cbb_scores.csv")
    except FileNotFoundError:
        print("Error: cbb_scores.csv not found.")
        return df_rat, None
    
    return df_rat, df_sco

# --- SHARED NAVIGATION GENERATOR ---
def get_nav_html(active_page):
    links = [("index.html", "Rankings"), ("matchup.html", "Matchup Simulator")]
    
    desktop_html = '<div class="hidden md:block"><div class="ml-10 flex items-baseline space-x-4">'
    mobile_html = '<div class="md:hidden" id="mobile-menu"><div class="px-2 pt-2 pb-3 space-y-1 sm:px-3">'
    
    for url, label in links:
        is_active = (active_page == 'index' and url == 'index.html') or \
                    (active_page == 'matchup' and url == 'matchup.html')
        
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
    
    # Use 'Blended_AdjEM'
    ratings_map = df_ratings.set_index('Team')['Blended_AdjEM'].to_dict()
    
    if df_scores is not None and not df_scores.empty:
        pts_col = 'pts' if 'pts' in df_scores.columns else 'points'
        
        # --- FIX: DATA DEDUPLICATION START ---
        
        # 1. Clean duplicates in the team's own rows
        # If dates are strings (YYYY-MM-DD), standard sort works fine.
        # We sort desc so if there are dupes, we keep the first one encountered (usually fine)
        # or we could keep 'last' if we assume file is appended chronologically.
        df_scores = df_scores.drop_duplicates(subset=['date', 'team', 'opponent'], keep='last')

        # 2. Prepare Opponent Lookup
        opp_df = df_scores[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
        
        # 3. Clean duplicates in opponent lookup (CRITICAL FIX)
        # This prevents the "Cartesian Explosion" where 1 game matches 2 opponent entries (e.g. correct score + bad score)
        opp_df = opp_df.drop_duplicates(subset=['date', 'opponent'], keep='last')
        
        # --- FIX END ---

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
                    exp_margin = (row['Blended_AdjEM'] - opp_rating) + hfa_adj
                    
                    poss = g.get('possessions', 70)
                    if poss == 0: poss = 70
                    act_eff = (margin / poss) * 100
                    val_add = act_eff - exp_margin
                    
                    game_list.append({
                        'date': g['date'].strftime('%m/%d'),
                        'opp': opp,
                        'res': res,
                        'loc': loc_str,
                        'score': f"{int(pts)}-{int(opp_pt)}",
                        'val': round(val_add, 1)
                    })
            
        teams_dict[team] = {
            'rank': int(row['Rank']),
            'adjem': round(row['Blended_AdjEM'], 2),
            'adjo': round(row['AdjO'], 1),
            'adjd': round(row['AdjD'], 1),
            'adjt': round(row['AdjT'], 1),
            'games': game_list
        }
        
    js_content = f"const TEAMS_DATA = {json.dumps(teams_dict)};"
    
    with open(OUTPUT_TEAM_DATA, "w") as f:
        f.write(js_content)
    print(f"Generated {OUTPUT_TEAM_DATA}")

def generate_index(df):
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
    
    # Rename columns for JS template compatibility
    json_df = df.rename(columns={
        'Blended_AdjEM': 'AdjEM',
        'Current_AdjEM': 'Pure_AdjEM'
    })
    
    # Drop duplicates if any exist after rename
    json_df = json_df.loc[:, ~json_df.columns.duplicated()]
    
    cols = ['Rank', 'Team', 'Link', 'AdjEM', 'Pure_AdjEM', 'AdjO', 'AdjD', 'AdjT']
    cols = [c for c in cols if c in json_df.columns]
    
    rankings_json = json_df[cols].to_json(orient='records')

    html = template.replace("{{RANKINGS_JSON}}", rankings_json)
    html = html.replace("{{LAST_UPDATED}}", now)
    html = html.replace("{{NAV_DESKTOP}}", nav_d)
    html = html.replace("{{NAV_MOBILE}}", nav_m)
    
    with open(OUTPUT_INDEX, "w") as f:
        f.write(html)
    print(f"Generated {OUTPUT_INDEX}")

def generate_matchup(df):
    print("Generating Matchup...")
    teams_data = {}
    for _, row in df.sort_values('Team').iterrows():
        # Pass Blended_AdjEM implicitly for logic via AdjO/AdjD if they are blended
        teams_data[row['Team']] = {
            'AdjO': row['AdjO'], 
            'AdjD': row['AdjD'], 
            'AdjT': row['AdjT'],
            'AdjEM': row['Blended_AdjEM']
        }
    teams_json = json.dumps(teams_data)

    try:
        with open("matchup_template.html", "r") as f:
            template = f.read()
            
        nav_d, nav_m = get_nav_html('matchup')
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        html = template.replace("{{TEAMS_JSON}}", teams_json)
        html = html.replace("{{LAST_UPDATED}}", now)
        html = html.replace("{{NAV_DESKTOP}}", nav_d)
        html = html.replace("{{NAV_MOBILE}}", nav_m)

        with open(OUTPUT_MATCHUP, "w") as f:
            f.write(html)
        print(f"Generated {OUTPUT_MATCHUP}")
    except FileNotFoundError:
        print("Warning: matchup_template.html not found.")

if __name__ == "__main__":
    # 1. Update Data
    run_script(SCRAPER_SCRIPT)
    run_script(RATINGS_SCRIPT)
    
    # 2. Get Data
    df_ratings, df_scores = get_data()

    if df_ratings is not None:
        # 3. Build SPA Assets
        generate_teams_js(df_ratings, df_scores)
        generate_index(df_ratings)
        generate_matchup(df_ratings)