import pandas as pd
import subprocess
import datetime
import os
import json

# --- CONFIGURATION ---
# Matching your specific filenames
SCRAPER_SCRIPT = "scraper.py" 
RATINGS_SCRIPT = "adj.py"
OUTPUT_INDEX = "index.html"
OUTPUT_MATCHUP = "matchup.html"

def run_script(script_name):
    print(f"--- Running {script_name} ---")
    try:
        # Using "python3" for Mac/Linux (change to "python" if on Windows)
        subprocess.run(["python3", script_name], check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        pass

def generate_pages():
    print("Generating Website Pages...")
    
    # 1. Load Data
    try:
        df = pd.read_csv("cbb_ratings.csv")
    except FileNotFoundError:
        print("Error: cbb_ratings.csv not found. Did the ratings script fail?")
        return

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
    json_df = json_df.rename(columns={
        'Blended_AdjEM': 'AdjEM',
        'Current_AdjEM': 'Pure_AdjEM'
    })
    
    # Select only necessary columns to keep file size down
    cols_to_keep = ['Rank', 'Team', 'AdjEM', 'Pure_AdjEM', 'AdjO', 'AdjD', 'AdjT']
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
    
    # Step 3: Build Website
    generate_pages()