import pandas as pd
import subprocess
import datetime
import os
import json
import shutil
import re

# --- CONFIGURATION ---
# Matching your specific filenames
SCRAPER_SCRIPT = "scraper.py" 
RATINGS_SCRIPT = "adj.py"
OUTPUT_INDEX = "index.html"
OUTPUT_MATCHUP = "matchup.html"
TEAMS_DIR = "teams"

def run_script(script_name):
    print(f"--- Running {script_name} ---")
    try:
        # Using "python3" for Mac/Linux (change to "python" if on Windows)
        subprocess.run(["python3", script_name], check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        pass

def get_data():
    try:
        df = pd.read_csv("cbb_ratings.csv")
    except FileNotFoundError:
        print("Error: cbb_ratings.csv not found. Did the ratings script fail?")
        return None, None

    try:
        df_scores = pd.read_csv("cbb_scores.csv")
    except FileNotFoundError:
        print("Error: cbb_scores.csv not found. Did the scraper fail?")
        return df, None

    return df, df_scores

def clean_filename(name):
    # Converts "St. John's (NY)" to "St_Johns_NY"
    clean = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return clean

def generate_team_pages(df_ratings, df_scores):
    print("Generating Team Detail Pages...")
    
    # Reset/Create 'teams' directory
    if os.path.exists(TEAMS_DIR):
        shutil.rmtree(TEAMS_DIR)
    os.makedirs(TEAMS_DIR)

    try:
        with open("team_template.html", "r") as f:
            template = f.read()
    except FileNotFoundError:
        print("Error: team_template.html not found. Skipping team pages.")
        return

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    
    # Create lookup for Opponent Ratings
    ratings_map = df_ratings.set_index('Team')['Blended_AdjEM'].to_dict()

    # Pre-process scores to get margins
    # We need to merge scores with itself to get opponent score
    pts_col = 'pts' if 'pts' in df_scores.columns else 'points'
    
    # Create a copy for opponents to merge
    opp_df = df_scores[['date', 'team', pts_col]].rename(columns={'team': 'opponent', pts_col: 'opp_pts'})
    
    # Merge to get game results
    full_scores = df_scores.merge(opp_df, on=['date', 'opponent'], how='left')
    
    # Loop through every ranked team
    for _, row in df_ratings.iterrows():
        team = row['Team']
        filename = f"{clean_filename(team)}.html"
        
        # Filter games for this team
        games = full_scores[full_scores['team'] == team].copy()
        
        # Sort by date descending if date column exists
        if 'date' in games.columns:
            games['date'] = pd.to_datetime(games['date'])
            games = games.sort_values('date', ascending=False)

        game_rows = ""
        for _, g in games.iterrows():
            opp = g['opponent']
            opp_rating = ratings_map.get(opp, 0.0)
            
            # 1. Determine Location
            loc_val = g.get('location_value', 0)
            if loc_val == 1: loc_str = "Home"
            elif loc_val == -1: loc_str = "Away"
            else: loc_str = "Neutral"

            # 2. Determine Result & Scores
            pts = g[pts_col]
            opp_pts = g['opp_pts']
            
            # Handle missing opponent score (rare data error)
            if pd.isna(opp_pts): continue
            
            margin = pts - opp_pts
            res_char = "W" if margin > 0 else "L"
            res_color = "text-green-700 bg-green-50" if margin > 0 else "text-red-700 bg-red-50"
            
            # 3. Calculate Value Add
            # Expected Margin = (TeamAdjEM - OppAdjEM) + HFA
            # HFA is approx 3.2 for home, -3.2 for away
            hfa_adj = 3.2 if loc_val == 1 else (-3.2 if loc_val == -1 else 0)
            exp_margin = (row['Blended_AdjEM'] - opp_rating) + hfa_adj
            
            # Actual Performance (Efficiency Margin approx)
            # Net Eff = (Margin / Possessions) * 100
            poss = g.get('possessions', 70)
            if poss == 0: poss = 70
            act_eff_margin = (margin / poss) * 100
            
            value_add = act_eff_margin - exp_margin
            val_color = "text-green-600" if value_add > 0 else "text-red-600"

            # 4. Build Row
            date_str = g['date'].strftime('%m/%d') if 'date' in g else "-"
            
            game_rows += f"""
            <tr class="hover:bg-slate-50 border-b border-gray-100 transition-colors">
                <td class="px-4 py-3 text-gray-500 text-xs">{date_str}</td>
                <td class="px-4 py-3 font-medium text-slate-900 truncate max-w-[140px]">{opp}</td>
                <td class="px-4 py-3 text-center"><span class="px-2 py-1 rounded-full text-xs font-bold {res_color}">{res_char}</span></td>
                <td class="px-4 py-3 text-right text-gray-700 font-mono text-sm">{int(pts)}-{int(opp_pts)}</td>
                <td class="px-4 py-3 text-right font-bold {val_color} text-sm">{value_add:+.1f}</td>
            </tr>
            """

        # Fill Template
        html = template.replace("{{TEAM_NAME}}", team)
        html = html.replace("{{RANK}}", str(int(row['Rank'])))
        html = html.replace("{{ADJEM}}", f"{row['Blended_AdjEM']:.2f}")
        html = html.replace("{{ADJO}}", f"{row['AdjO']:.1f}")
        html = html.replace("{{ADJD}}", f"{row['AdjD']:.1f}")
        html = html.replace("{{ADJT}}", f"{row['AdjT']:.1f}")
        html = html.replace("{{LAST_UPDATED}}", now)
        html = html.replace("{{GAME_ROWS}}", game_rows)

        with open(f"{TEAMS_DIR}/{filename}", "w") as f:
            f.write(html)

    print(f"Generated {len(df_ratings)} team pages.")

def generate_pages(df):
    print("Generating Website Pages...")

    now = datetime.datetime.now(datetime.timezone.utc)
    last_updated = now.strftime("%Y-%m-%d %H:%M UTC")

    # --- HELPER: Navigation Bar Generator ---
    def get_nav_html(active_page):
        links = [("index.html", "Rankings"), ("matchup.html", "Matchup Simulator")]
        
        # 1. Desktop Nav (Hidden on Mobile)
        desktop_html = '<div class="hidden md:block"><div class="ml-10 flex items-baseline space-x-4">'
        
        # 2. Mobile Nav (Hidden on Desktop, toggled via JS)
        mobile_html = '<div class="md:hidden" id="mobile-menu"><div class="px-2 pt-2 pb-3 space-y-1 sm:px-3">'
        
        for url, label in links:
            # Adjust link paths if we are in a subdirectory (like teams/)
            # But here we are generating root files, so paths are simple.
            # Team pages will need "../index.html" but that's handled in their template.
            
            is_active = (active_page == 'index' and url == 'index.html') or \
                        (active_page == 'matchup' and url == 'matchup.html')
            
            # Desktop Classes
            d_cls = "bg-slate-900 text-white" if is_active else "text-gray-300 hover:bg-slate-700 hover:text-white"
            desktop_html += f'<a href="{url}" class="{d_cls} px-3 py-2 rounded-md text-sm font-medium transition-colors">{label}</a>'
            
            # Mobile Classes (Needs 'block' and larger text)
            m_cls = "bg-slate-900 text-white" if is_active else "text-gray-300 hover:bg-slate-700 hover:text-white"
            mobile_html += f'<a href="{url}" class="{m_cls} block px-3 py-2 rounded-md text-base font-medium">{label}</a>'
            
        desktop_html += '</div></div>'
        mobile_html += '</div></div>'
        
        return desktop_html, mobile_html

    # --- 2. BUILD INDEX.HTML (Rankings) ---
    # CONVERT TO JSON FOR DYNAMIC DISPLAY
    # We map your specific column names to the generic names the JS template expects
    json_df = df.copy()
    
    # Add link column for team pages
    json_df['Link'] = json_df['Team'].apply(lambda x: f"teams/{clean_filename(x)}.html")

    json_df = json_df.rename(columns={
        'Blended_AdjEM': 'AdjEM',
        'Current_AdjEM': 'Pure_AdjEM'
    })
    
    # Select only necessary columns to keep file size down
    cols_to_keep = ['Rank', 'Team', 'Link', 'AdjEM', 'Pure_AdjEM', 'AdjO', 'AdjD', 'AdjT']
    # Filter to ensure columns exist (safety check)
    final_cols = [c for c in cols_to_keep if c in json_df.columns]
    
    rankings_json = json_df[final_cols].to_json(orient='records')

    try:
        with open("template.html", "r") as f:
            template_idx = f.read()
            
        # Get Nav Blocks
        nav_desk, nav_mob = get_nav_html('index')

        # Replace Placeholders
        # Note: We replaced {{TABLE_ROWS}} with {{RANKINGS_JSON}} for the dynamic template
        html_idx = template_idx.replace("{{RANKINGS_JSON}}", rankings_json)
        html_idx = html_idx.replace("{{LAST_UPDATED}}", last_updated)
        html_idx = html_idx.replace("{{NAV_DESKTOP}}", nav_desk)
        html_idx = html_idx.replace("{{NAV_MOBILE}}", nav_mob)
        
        with open(OUTPUT_INDEX, "w") as f:
            f.write(html_idx)
        print(f"Generated {OUTPUT_INDEX}")
        
    except FileNotFoundError:
        print("Error: template.html not found.")

    # --- 3. BUILD MATCHUP.HTML (Simulator) ---
    teams_data = {}
    for _, row in df.sort_values('Team').iterrows():
        teams_data[row['Team']] = {
            'AdjO': row['AdjO'], 
            'AdjD': row['AdjD'], 
            'AdjT': row['AdjT']
        }
    teams_json = json.dumps(teams_data)

    try:
        with open("matchup_template.html", "r") as f:
            template_sim = f.read()
            
        nav_desk_sim, nav_mob_sim = get_nav_html('matchup')
        
        html_sim = template_sim.replace("{{TEAMS_JSON}}", teams_json)
        html_sim = html_sim.replace("{{LAST_UPDATED}}", last_updated)
        html_sim = html_sim.replace("{{NAV_DESKTOP}}", nav_desk_sim)
        html_sim = html_sim.replace("{{NAV_MOBILE}}", nav_mob_sim)

        with open(OUTPUT_MATCHUP, "w") as f:
            f.write(html_sim)
        print(f"Generated {OUTPUT_MATCHUP}")
        
    except FileNotFoundError:
        print("Warning: matchup_template.html not found. Skipping simulator page.")

if __name__ == "__main__":
    # Step 1: Update Data
    run_script(SCRAPER_SCRIPT)
    
    # Step 2: Recalculate Ratings
    run_script(RATINGS_SCRIPT)
    
    # Step 3: Get Data
    df_ratings, df_scores = get_data()

    if df_ratings is not None:
        # Step 4: Build Website Pages
        generate_pages(df_ratings)
        
        # Step 5: Build Team Pages (if scores exist)
        if df_scores is not None:
            generate_team_pages(df_ratings, df_scores)