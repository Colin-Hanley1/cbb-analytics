import pandas as pd
import datetime
import os

# --- CONFIGURATION ---
RATINGS_FILE = "cbb_ratings.csv"
CONFERENCE_FILE = "cbb_conferences.csv"
OUTPUT_HTML = "bracketology.html"

# --- MANUAL AUTO-BIDS ---
# Format: "Conference Name": "Team Name" (Must match mapped names)
# Leave empty {} to use highest rated team automatically.
# Example: "Ivy": "Princeton", "Big Ten": "Purdue"
'''
MANUAL_AQS = {
    "SEC": "Florida",
    "B12": "Houston",
    "ACC": "Duke",
    "BE": "St. John's",
    "B10": "Michigan",
    "WCC": "Gonzaga",
    "Amer": "Memphis",
    "MVC": "Drake",
    "A10": "VCU",
    "BW": "UC San Diego",
    "MWC": "Colorado St.",
    "Slnd": "McNeese",
    "CUSA": "Liberty",
    "Ivy": "Yale",
    "BSth": "High Point",
    "MAC": "Akron",
    "CAA": "UNC Wilmington",
    "ASun": "Lipscomb",
    "WAC": "Grand Canyon",
    "SC": "Wofford",
    "SB": "Troy",
    "Horz": "Robert Morris",
    "BSky": "Montana",
    "Sum": "Nebraska Omaha",
    "NEC": "Saint Francis",
    "AE": "Bryant",
    "SWAC": "Alabama St.",
    "OVC": "Southeast Missouri",
    "PL": "American",
    "MEAC": "Norfolk St.",
    "MAAC": "Mount St. Mary's"   
}
'''
MANUAL_AQS={}

# --- NAME MAPPING (Must match your other scripts) ---
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

    # 3. Calculate Selection Score
    df['Selection_Score'] = (df['WAB'] * 4) + df['Blended_AdjEM']
    
    # 4. Determine Automatic Qualifiers (AQs) with Manual Override
    df['AQ'] = False
    
    # Get all unique conferences from the data
    conferences = df['Conference'].dropna().unique()
    
    aq_teams_list = []
    
    for conf in conferences:
        if conf == 'Unknown': continue
        
        # A. Check Manual Overrides first
        if conf in MANUAL_AQS:
            forced_team = MANUAL_AQS[conf]
            # Verify team exists in our data
            if forced_team in df['Team'].values:
                aq_teams_list.append(forced_team)
                continue
            else:
                print(f"Warning: Manual AQ '{forced_team}' for '{conf}' not found in ratings. Reverting to auto-select.")
        
        # B. Fallback: Highest rated team (AdjEM) in conference
        conf_teams = df[df['Conference'] == conf]
        if not conf_teams.empty:
            top_team = conf_teams.sort_values('Blended_AdjEM', ascending=False).iloc[0]['Team']
            aq_teams_list.append(top_team)
            
    # Mark AQs in DataFrame
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
    
    # We need 68 teams total. Remove count of AQs from 68 to get At-Large spots.
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
    
    # Sort S-Curve
    field_df = field_df.sort_values('Selection_Score', ascending=False).reset_index(drop=True)
    field_df['Overall_Rank'] = field_df.index + 1
    
    # Assign Seeds (1-10 have 4 teams; 11 and 16 have 6 teams; 12-15 have 4 teams)
    seeds = []
    for s in range(1, 11): seeds.extend([s] * 4) # 1-10
    seeds.extend([11] * 6) # 11 (First Four)
    for s in range(12, 16): seeds.extend([s] * 4) # 12-15
    seeds.extend([16] * 6) # 16 (First Four)
    
    # Handle list length mismatches (safety)
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
        type_class = "bg-green-100 text-green-800" if row['Bid_Type'] == 'Auto' else "bg-blue-100 text-blue-800"
        score_val = row['Selection_Score']
        
        rows += f"""
        <tr class="border-b border-gray-100 hover:bg-gray-50 transition-colors">
            <td class="px-6 py-4 font-bold text-center text-slate-700">{int(row['Seed'])}</td>
            <td class="px-6 py-4 font-medium text-slate-900">{row['Team']}</td>
            <td class="px-6 py-4 text-center text-xs font-bold uppercase"><span class="px-2 py-1 rounded {type_class}">{row['Bid_Type']}</span></td>
            <td class="px-6 py-4 text-right text-slate-500 text-sm">{row['Conference']}</td>
            <td class="px-6 py-4 text-right font-mono font-bold text-slate-900">{score_val:.2f}</td>
            <td class="px-6 py-4 text-right font-mono text-sm text-slate-400">{row['WAB']:.2f}</td>
        </tr>
        """
        
    def make_bubble_list(df):
        h = ""
        for _, r in df.iterrows():
            h += f"""
            <div class="flex justify-between items-center py-2 border-b border-gray-100 last:border-0">
                <div>
                    <span class="font-bold text-slate-800">{r['Team']}</span>
                    <span class="text-xs text-slate-400 ml-1">{r['Conference']}</span>
                </div>
                <div class="text-right">
                    <div class="text-xs font-bold text-slate-600">WAB: {r['WAB']:.2f}</div>
                    <div class="text-[10px] text-slate-400">Score: {r['Selection_Score']:.1f}</div>
                </div>
            </div>
            """
        return h

    template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bracketology | CHan Analytics</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style> body {{ font-family: 'Inter', sans-serif; }} .font-mono {{ font-family: 'Roboto Mono', monospace; }} </style>
    </head>
    <body class="bg-slate-50 text-slate-900 font-sans min-h-screen flex flex-col">
        
        <nav class="bg-slate-900 text-white shadow-lg sticky top-0 z-50">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex items-center justify-between h-16">
                    <div class="flex items-center space-x-4">
                        <span class="font-bold text-xl tracking-tight text-white">CHan Analytics</span>
                        <a href="index.html" class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Rankings</a>
                        <a href="bracketology.html" class="bg-slate-800 text-white px-3 py-2 rounded-md text-sm font-medium">Bracketology</a>
                    </div>
                    <div class="text-xs text-slate-400">Updated: {now}</div>
                </div>
            </div>
        </nav>

        <main class="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <h1 class="text-3xl font-bold text-slate-900 mb-2">Bracketology Projection</h1>
            <p class="text-slate-500 mb-8 max-w-2xl">
                Projections based on a composite "Selection Score" (4x WAB + AdjEM). 
                <br>Automatic bids awarded to highest rated team in conference (or manual override).
            </p>

            <!-- Bubble Watch -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
                <div class="bg-white p-5 rounded-xl shadow-sm border-t-4 border-green-500">
                    <h3 class="font-bold text-green-700 uppercase tracking-wide text-sm mb-4">Last 4 In</h3>
                    {make_bubble_list(l4i)}
                </div>
                <div class="bg-white p-5 rounded-xl shadow-sm border-t-4 border-orange-500">
                    <h3 class="font-bold text-orange-700 uppercase tracking-wide text-sm mb-4">First 4 Out</h3>
                    {make_bubble_list(f4o)}
                </div>
                <div class="bg-white p-5 rounded-xl shadow-sm border-t-4 border-red-500">
                    <h3 class="font-bold text-red-700 uppercase tracking-wide text-sm mb-4">Next 4 Out</h3>
                    {make_bubble_list(n4o)}
                </div>
            </div>

            <!-- The Field Table -->
            <div class="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
                <div class="overflow-x-auto">
                    <table class="w-full text-left whitespace-nowrap">
                        <thead class="bg-slate-100 text-xs uppercase text-slate-500 font-bold border-b border-gray-200">
                            <tr>
                                <th class="px-6 py-4 text-center">Seed</th>
                                <th class="px-6 py-4">Team</th>
                                <th class="px-6 py-4 text-center">Bid</th>
                                <th class="px-6 py-4 text-right">Conf</th>
                                <th class="px-6 py-4 text-right">Score</th>
                                <th class="px-6 py-4 text-right">WAB</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-100">
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </main>
        
        <footer class="bg-white border-t border-gray-200 mt-auto py-6">
            <div class="max-w-6xl mx-auto px-4 text-center text-slate-400 text-xs">
                Automated Analysis System
            </div>
        </footer>
    </body>
    </html>
    """
    return template

if __name__ == "__main__":
    generate_bracket()