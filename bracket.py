import pandas as pd
import datetime
import os
import numpy as np

# --- CONFIGURATION ---
RATINGS_FILE = "cbb_ratings.csv"
CONFERENCE_FILE = "cbb_conferences.csv"
OUTPUT_HTML = "bracketology.html"

# --- MODEL COEFFICIENTS ---
# PASTE THE OUTPUT FROM train_bracket_model.py HERE
# These are placeholder weights based on typical committee behavior
SCORING_WEIGHTS = {
    'ADJEM': 5.5253779546839645,       # Base Power Metric (Mapped from Blended_AdjEM)
    'WAB': 16.542519618901224,         # Base Resume Metric
    'Q1_W': 6.528047173692391,       # Bonus for Elite Wins
    'Q1_L': 0.0,        # No penalty for quality losses
    'Q2_W': 5.355022855830827,        # Slight bonus
    'Q2_L': 0.0,       # Slight penalty
    'Q3_W': 0.0,        # Expected win
    'Q3_L': 0.0,       # Bad loss
    'Q4_W': 0.0,        # Expected win
    'Q4_L': 0.0,       # Catastrophic loss
    'T100_W': 9.306541120747148,     # Depth of wins
    'T100_L': 0.0     # Depth of losses
}

# --- MANUAL AUTO-BIDS ---
# Format: "Conference Name": "Team Name"
MANUAL_AQS = {
    # "Ivy": "Princeton",
}

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

def generate_bracket():
    print("Generating Bracketology...")

    # 1. Load Data
    try:
        df = pd.read_csv(RATINGS_FILE)
    except FileNotFoundError:
        print(f"Error: {RATINGS_FILE} not found.")
        return

    try:
        df_conf = pd.read_csv(CONFERENCE_FILE)
        df_conf['Team'] = df_conf['Team'].replace(NAME_MAPPING)
    except FileNotFoundError:
        print(f"Error: {CONFERENCE_FILE} not found. Cannot determine Auto-Bids.")
        return

    # 2. Merge Ratings and Conferences
    df = df.merge(df_conf, on='Team', how='left')
    df['Conference'] = df['Conference'].fillna('Unknown')

    # 3. CALCULATE SELECTION SCORE (New Model)
    # ----------------------------------------------------
    # Map the CSV column 'Blended_AdjEM' to the model feature 'ADJEM'
    df['ADJEM'] = df['Current_AdjEM']
    
    # Initialize score
    df['Selection_Score'] = 0.0
    
    # Apply weights
    for feature, weight in SCORING_WEIGHTS.items():
        if feature in df.columns:
            df['Selection_Score'] += df[feature] * weight
        else:
            print(f"Warning: Feature '{feature}' missing from ratings. Treated as 0.")
    # ----------------------------------------------------
    
    # 4. Determine Automatic Qualifiers (AQs)
    df['AQ'] = False
    conferences = df['Conference'].dropna().unique()
    aq_teams_list = []
    
    for conf in conferences:
        if conf == 'Unknown': continue
        if conf in MANUAL_AQS:
            forced_team = MANUAL_AQS[conf]
            if forced_team in df['Team'].values:
                aq_teams_list.append(forced_team)
                continue
        
        # Fallback: Highest Selection Score in conference wins AQ
        conf_teams = df[df['Conference'] == conf]
        if not conf_teams.empty:
            top_team = conf_teams.sort_values('Selection_Score', ascending=False).iloc[0]['Team']
            aq_teams_list.append(top_team)
            
    df.loc[df['Team'].isin(aq_teams_list), 'AQ'] = True
    
    # 5. Select the Field (68 Teams)
    field = []
    
    # A. Automatic Bids
    auto_bids = df[df['AQ'] == True].copy()
    for _, row in auto_bids.iterrows():
        row['Bid_Type'] = 'Auto'
        field.append(row)
        
    # B. At-Large Bids
    at_large_pool = df[df['AQ'] == False].sort_values('Selection_Score', ascending=False)
    num_at_large = 68 - len(auto_bids)
    at_large_bids = at_large_pool.head(num_at_large).copy()
    
    # Bubble Teams
    first_4_out = at_large_pool.iloc[num_at_large:num_at_large+4]
    next_4_out = at_large_pool.iloc[num_at_large+4:num_at_large+8]
    
    for _, row in at_large_bids.iterrows():
        row['Bid_Type'] = 'At-Large'
        field.append(row)
        
    last_4_in = at_large_bids.tail(4)

    # 6. Seed the Field (Custom 68-team logic)
    field_df = pd.DataFrame(field)
    field_df = field_df.sort_values('Selection_Score', ascending=False).reset_index(drop=True)
    field_df['Overall_Rank'] = field_df.index + 1
    
    seeds = []
    for s in range(1, 11): seeds.extend([s] * 4) 
    seeds.extend([11] * 6) 
    for s in range(12, 16): seeds.extend([s] * 4) 
    seeds.extend([16] * 6) 
    
    if len(field_df) <= len(seeds):
        field_df['Seed'] = seeds[:len(field_df)]
    else:
        seeds.extend([16] * (len(field_df) - len(seeds)))
        field_df['Seed'] = seeds

    # 7. Generate HTML
    html = generate_html(field_df, last_4_in, first_4_out, next_4_out)
    
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
        
    print(f"Bracketology generated: {OUTPUT_HTML}")

def generate_html(field_df, l4i, f4o, n4o):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    rows = ""
    for _, row in field_df.iterrows():
        type_class = "text-emerald-700 bg-emerald-50" if row['Bid_Type'] == 'Auto' else "text-blue-700 bg-blue-50"
        seed_color = "text-slate-900" if row['Seed'] <= 4 else "text-slate-500"
        
        rows += f"""
        <tr class="hover:bg-slate-50 border-b border-gray-100 transition-colors">
            <td class="px-5 py-4 font-bold text-center {seed_color} border-r border-gray-100 text-sm">{int(row['Seed'])}</td>
            <td class="px-5 py-4 font-medium text-slate-900 border-r border-gray-100 text-sm">{row['Team']}</td>
            <td class="px-5 py-4 text-center border-r border-gray-100"><span class="px-2.5 py-1 rounded-full text-xs font-bold {type_class}">{row['Bid_Type']}</span></td>
            <td class="px-5 py-4 text-center text-slate-500 text-sm">{row['Conference']}</td>
            <td class="px-5 py-4 text-right font-mono font-bold text-slate-800 text-sm">{row['Selection_Score']:.2f}</td>
            <td class="px-5 py-4 text-right font-mono text-purple-600 font-bold text-sm bg-purple-50/30">{row['WAB']:.2f}</td>
        </tr>
        """
        
    def make_bubble_list(df, border_color):
        h = ""
        for _, r in df.iterrows():
            h += f"""
            <div class="flex justify-between items-center py-3 border-b border-gray-100 last:border-0 hover:bg-gray-50 px-2 rounded-lg transition-colors">
                <div>
                    <div class="font-bold text-slate-900 text-sm">{r['Team']}</div>
                    <div class="text-xs text-slate-400">{r['Conference']}</div>
                </div>
                <div class="text-right">
                    <div class="text-xs font-mono font-bold text-purple-600">WAB: {r['WAB']:.2f}</div>
                    <div class="text-[10px] text-slate-400 font-mono">Score: {r['Selection_Score']:.1f}</div>
                </div>
            </div>
            """
        return h

    template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Bracketology | CHan Analytics</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style> 
            body {{ font-family: 'Inter', sans-serif; background-color: #f1f5f9; }} 
            .font-mono {{ font-family: 'Roboto Mono', monospace; }}
            #mobile-menu {{ display: none; }}
            #mobile-menu.open {{ display: block; }}
        </style>
    </head>
    <body class="text-slate-900 min-h-screen flex flex-col">
        
        <nav class="bg-slate-900 text-white shadow-lg sticky top-0 z-50">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex items-center justify-between h-16">
                    <div class="flex items-center">
                        <span class="font-bold text-xl tracking-tight text-white">CHan Analytics</span>
                        <div class="hidden md:block ml-10">
                            <div class="flex items-baseline space-x-4">
                                <a href="index.html" class="text-gray-300 hover:bg-slate-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">Rankings</a>
                                <a href="schedule.html" class="text-gray-300 hover:bg-slate-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">Today's Games</a>
                                <a href="matchup.html" class="text-gray-300 hover:bg-slate-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">Matchup Simulator</a>
                                <a href="bracketology.html" class="bg-slate-800 text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">Bracketology</a>
                            </div>
                        </div>
                    </div>
                    
                    <div class="flex items-center">
                        <div class="hidden md:block text-xs text-slate-400 mr-4">Updated: <span class="text-white">{now}</span></div>
                        <div class="-mr-2 flex md:hidden">
                            <button type="button" onclick="document.getElementById('mobile-menu').classList.toggle('open')" class="bg-slate-800 inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-white hover:bg-slate-700 focus:outline-none">
                                <svg class="h-6 w-6" stroke="currentColor" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" /></svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="md:hidden" id="mobile-menu">
                <div class="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                    <a href="index.html" class="text-gray-300 hover:bg-slate-700 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Rankings</a>
                    <a href="schedule.html" class="text-gray-300 hover:bg-slate-700 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Today's Games</a>
                    <a href="matchup.html" class="text-gray-300 hover:bg-slate-700 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Matchup Simulator</a>
                    <a href="bracketology.html" class="bg-slate-900 text-white block px-3 py-2 rounded-md text-base font-medium">Bracketology</a>
                </div>
            </div>
        </nav>

        <main class="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <h1 class="text-3xl font-bold text-slate-900 mb-2">Bracketology Projection</h1>
            <p class="text-slate-500 mb-8 max-w-2xl">
                Projections based on a weighted composite of efficiency, resume, and quality wins.
                <br class="hidden sm:inline">Automatic bids awarded to the highest rated team in each conference.
            </p>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
                <div class="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden">
                    <div class="absolute top-0 left-0 w-full h-1 bg-emerald-500"></div>
                    <h3 class="font-bold text-emerald-700 uppercase tracking-widest text-xs mb-4">Last 4 In</h3>
                    {make_bubble_list(l4i, "emerald")}
                </div>
                <div class="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden">
                    <div class="absolute top-0 left-0 w-full h-1 bg-orange-500"></div>
                    <h3 class="font-bold text-orange-700 uppercase tracking-widest text-xs mb-4">First 4 Out</h3>
                    {make_bubble_list(f4o, "orange")}
                </div>
                <div class="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden">
                    <div class="absolute top-0 left-0 w-full h-1 bg-red-500"></div>
                    <h3 class="font-bold text-red-700 uppercase tracking-widest text-xs mb-4">Next 4 Out</h3>
                    {make_bubble_list(n4o, "red")}
                </div>
            </div>

            <div class="bg-white shadow-xl rounded-2xl overflow-hidden border border-slate-200">
                <div class="overflow-x-auto">
                    <table class="w-full text-left whitespace-nowrap">
                        <thead class="bg-slate-50 border-b border-gray-200 text-xs uppercase tracking-wider text-slate-500 font-bold">
                            <tr>
                                <th class="px-6 py-4 text-center border-r border-gray-200">Seed</th>
                                <th class="px-6 py-4 border-r border-gray-200">Team</th>
                                <th class="px-6 py-4 text-center border-r border-gray-200">Bid</th>
                                <th class="px-6 py-4 text-center border-r border-gray-200">Conf</th>
                                <th class="px-6 py-4 text-right">Score</th>
                                <th class="px-6 py-4 text-right text-purple-700">WAB</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-100 bg-white text-sm">
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </main>
        
        <footer class="bg-white border-t border-gray-200 mt-auto py-6">
            <div class="max-w-6xl mx-auto px-4 text-center text-slate-400 text-xs">
                Automated Analysis System &bull; Powered by Python & GitHub Actions
            </div>
        </footer>
    </body>
    </html>
    """
    return template

if __name__ == "__main__":
    generate_bracket()