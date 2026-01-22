import pandas as pd
import datetime
import os
import numpy as np

# --- CONFIGURATION ---
RATINGS_FILE = "cbb_ratings.csv"
CONFERENCE_FILE = "cbb_conferences.csv"
OUTPUT_HTML = "bracketology.html"

# --- MODEL COEFFICIENTS (Keep your weights here) ---
SCORING_WEIGHTS = {
    "ADJEM": 3.1455920563817816,
    "RQ": 7.501438184436439,
    "Q1_W": 6.7633764359705095,
    "Q1_L": -2.2095077635548015,
    "Q2_W": 4.653701084972598,
    "Q2_L": -2.423209667527029,
    "Q3_W": 0.07221419803078077,
    "Q3_L": -6.999764853540507,
    "Q4_L": -16.48658423831865
}

# --- MANUAL AUTO-BIDS ---
MANUAL_AQS = {}

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
    # Use Current_AdjEM (Pure) if available, else fallback to AdjEM
    if 'Current_AdjEM' in df.columns:
        df['ADJEM'] = df['Current_AdjEM']
    else:
        df['ADJEM'] = df['AdjEM']  # Fallback

    # Initialize score
    df['Selection_Score'] = 0.0

    # Apply weights
    for feature, weight in SCORING_WEIGHTS.items():
        if feature in df.columns:
            df['Selection_Score'] += df[feature] * weight
        # Missing feature treated as 0

    # 4. Determine Automatic Qualifiers (AQs)
    df['AQ'] = False
    conferences = df['Conference'].dropna().unique()
    aq_teams_list = []

    for conf in conferences:
        if conf == 'Unknown':
            continue

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
    for s in range(1, 11):
        seeds.extend([s] * 4)
    seeds.extend([11] * 6)
    for s in range(12, 16):
        seeds.extend([s] * 4)
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
        logo_path = f"logos/{row['Team']}.png"

        # Styles
        type_class = (
            "text-emerald-700 bg-emerald-50 border border-emerald-100"
            if row['Bid_Type'] == 'Auto'
            else "text-blue-700 bg-blue-50 border border-brand-100"
        )

        # Seed Color
        seed_style = "text-slate-900 font-extrabold" if row['Seed'] <= 4 else "text-slate-500 font-bold"

        rows += f"""
        <tr class="hover:bg-slate-50 transition-colors group border-b border-slate-50">
            <!-- SEED -->
            <td class="px-6 py-4 text-center border-r border-slate-100/50 w-16">
                <span class="text-lg {seed_style}">{int(row['Seed'])}</span>
            </td>

            <!-- TEAM (With Logo) -->
            <td class="px-6 py-4 border-r border-slate-100/50">
                <div class="flex items-center gap-3">
                    <img src="{logo_path}" class="w-8 h-8 object-contain" onerror="this.style.display='none'">
                    <div>
                        <div class="font-bold text-slate-900 text-base">{row['Team']}</div>
                        <div class="text-[10px] text-slate-400 font-mono tracking-wide uppercase">{row['Conference']}</div>
                    </div>
                </div>
            </td>

            <!-- BID TYPE -->
            <td class="px-6 py-4 text-center border-r border-slate-100/50">
                <span class="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider {type_class}">{row['Bid_Type']}</span>
            </td>

            <!-- SCORE -->
            <td class="px-6 py-4 text-right font-mono font-bold text-slate-800 text-base border-r border-slate-100/50">
                {row['Selection_Score']:.1f}
            </td>

            <!-- RQ/WAB -->
            <td class="px-6 py-4 text-right">
                <span class="font-mono text-sm font-bold text-purple-600 bg-purple-50 px-2 py-1 rounded-md border border-purple-100">
                    {row['RQ']:.2f}
                </span>
            </td>
        </tr>
        """

    def make_bubble_list(df, color_theme):
        colors = {
            "emerald": {"text": "text-emerald-700", "bg": "bg-emerald-500"},
            "orange": {"text": "text-orange-700", "bg": "bg-orange-500"},
            "red": {"text": "text-rose-700", "bg": "bg-rose-500"}
        }
        c = colors[color_theme]

        h = ""
        for _, r in df.iterrows():
            logo = f"logos/{r['Team']}.png"

            h += f"""
            <div class="flex items-center justify-between py-3 border-b border-slate-100 last:border-0 hover:bg-slate-50 px-3 rounded-lg transition-all group">
                <div class="flex items-center gap-3">
                    <img src="{logo}" class="w-6 h-6 object-contain opacity-80 group-hover:opacity-100 transition-opacity" onerror="this.style.display='none'">
                    <div>
                        <div class="font-bold text-slate-800 text-sm group-hover:text-brand-600 transition-colors">{r['Team']}</div>
                        <div class="text-[10px] text-slate-400 font-mono">{r['Conference']}</div>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-xs font-bold font-mono text-purple-600">{r['RQ']:.2f}</div>
                    <div class="text-[10px] text-slate-400 font-mono">Score: {r['Selection_Score']:.1f}</div>
                </div>
            </div>
            """
        return h

    template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <link rel="shortcut icon" type="image/x-icon" href="favicon.ico" />
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bracketology | CHan Analytics</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

        <script>
            tailwind.config = {{
                theme: {{
                    extend: {{
                        fontFamily: {{
                            sans: ['Inter', 'sans-serif'],
                            mono: ['JetBrains Mono', 'monospace'],
                        }},
                        colors: {{
                            brand: {{ 500: '#6366f1', 600: '#4f46e5', 900: '#312e81' }}
                        }}
                    }}
                }}
            }};
        </script>

        <style> 
            body {{ -webkit-font-smoothing: antialiased; }} 
            #mobile-menu {{ display: none; transition: all 0.3s; }}
            #mobile-menu.open {{ display: block; }}

            /* Smooth scroll for normal anchor navigation */
            html {{ scroll-behavior: smooth; }}
        </style>

        <script>
            // Sticky-navbar-aware smooth scroll to #topTableAnchor
            function scrollToTableTop(e) {{
                if (e) e.preventDefault();

                const anchor = document.getElementById('topTableAnchor');
                if (!anchor) return;

                const nav = document.getElementById('siteNav');
                const navH = nav ? nav.getBoundingClientRect().height : 0;

                // Extra spacing so content isn't glued to the nav
                const pad = 12;

                const y = anchor.getBoundingClientRect().top + window.pageYOffset - navH - pad;
                window.scrollTo({{ top: Math.max(0, y), behavior: 'smooth' }});

                // Optional: keep URL clean (remove hash)
                history.replaceState(null, '', window.location.pathname);
            }}

            // If user loads with #topTableAnchor, apply offset-correct scroll once
            window.addEventListener('load', () => {{
                if (window.location.hash === '#topTableAnchor') {{
                    setTimeout(() => scrollToTableTop(), 0);
                }}
            }});
        </script>
    </head>

    <body class="bg-slate-50 text-slate-900 min-h-screen flex flex-col selection:bg-brand-500 selection:text-white">

        <!-- Navbar -->
        <nav id="siteNav" class="bg-slate-900/95 backdrop-blur-md text-white shadow-lg sticky top-0 z-50 border-b border-slate-700/50">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex items-center justify-between h-16">
                    <div class="flex items-center gap-3">
                        <a href="#topTableAnchor" onclick="scrollToTableTop(event)" class="flex items-center">
                            <img
                              src="wlogo.png"
                              alt="CHan Analytics"
                              class="h-6 sm:h-7 w-auto object-contain select-none"
                              style="max-height: 28px"
                            />
                        </a>

                        <div class="hidden md:flex ml-8 space-x-1">
                            <a href="index.html" class="text-slate-300 hover:text-white hover:bg-slate-800 px-3 py-2 rounded-md text-sm font-medium transition-colors">Rankings</a>
                            <a href="schedule.html" class="text-slate-300 hover:text-white hover:bg-slate-800 px-3 py-2 rounded-md text-sm font-medium transition-colors">Today's Games</a>
                            <a href="matchup.html" class="text-slate-300 hover:text-white hover:bg-slate-800 px-3 py-2 rounded-md text-sm font-medium transition-colors">Matchup Simulator</a>
                            <a href="bracketology.html" class="bg-slate-800 text-white px-3 py-2 rounded-md text-sm font-medium transition-colors border border-slate-700">Bracketology</a>
                        </div>
                    </div>

                    <div class="flex items-center gap-4">
                        <div class="hidden md:block text-xs font-mono text-slate-400 bg-slate-800 px-3 py-1.5 rounded-full border border-slate-700">
                            <span class="text-brand-500">●</span> Updated: <span class="text-slate-200">{now}</span>
                        </div>
                        <div class="-mr-2 flex md:hidden">
                            <button type="button" onclick="document.getElementById('mobile-menu').classList.toggle('open')" class="text-slate-400 hover:text-white p-2">
                                <svg class="h-6 w-6" stroke="currentColor" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" /></svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="md:hidden" id="mobile-menu">
                <div class="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-slate-900 border-t border-slate-800">
                    <a href="index.html" class="text-slate-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Rankings</a>
                    <a href="schedule.html" class="text-slate-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Today's Games</a>
                    <a href="matchup.html" class="text-slate-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Matchup Simulator</a>
                    <a href="bracketology.html" class="bg-slate-900 text-white block px-3 py-2 rounded-md text-base font-medium">Bracketology</a>
                </div>
            </div>
        </nav>

        <main class="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div id="topTableAnchor" style="scroll-margin-top: 88px;"></div>

            <div class="mb-8 text-center sm:text-left border-b border-slate-200 pb-6">
                <h1 class="text-3xl font-bold text-slate-900 tracking-tight">Bracketology</h1>
                <p class="mt-2 text-slate-500 max-w-2xl text-sm">
                    Projection based on <strong>Resume Quality</strong> (RQ, SOS, Q1/Q2 Performance) and <strong>Predictive Metrics</strong> (AdjEM).
                    <span class="block mt-1 text-xs text-slate-400">Automatic bids awarded to highest rated team in conference.</span>
                </p>
            </div>

            <!-- Bubble Watch -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
                <div class="bg-white p-1 rounded-2xl shadow-lg shadow-emerald-100 border border-slate-200">
                    <div class="bg-emerald-50/50 p-4 rounded-xl h-full">
                        <h3 class="font-bold text-emerald-700 uppercase tracking-widest text-xs mb-4 flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-emerald-500"></span> Last 4 In
                        </h3>
                        {make_bubble_list(l4i, "emerald")}
                    </div>
                </div>
                <div class="bg-white p-1 rounded-2xl shadow-lg shadow-orange-100 border border-slate-200">
                    <div class="bg-orange-50/50 p-4 rounded-xl h-full">
                        <h3 class="font-bold text-orange-700 uppercase tracking-widest text-xs mb-4 flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-orange-500"></span> First 4 Out
                        </h3>
                        {make_bubble_list(f4o, "orange")}
                    </div>
                </div>
                <div class="bg-white p-1 rounded-2xl shadow-lg shadow-rose-100 border border-slate-200">
                    <div class="bg-rose-50/50 p-4 rounded-xl h-full">
                        <h3 class="font-bold text-rose-700 uppercase tracking-widest text-xs mb-4 flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-rose-500"></span> Next 4 Out
                        </h3>
                        {make_bubble_list(n4o, "red")}
                    </div>
                </div>
            </div>

            <!-- The Field Table -->
            <div class="bg-white shadow-xl shadow-slate-200/50 rounded-2xl overflow-hidden border border-slate-200">
                <div class="overflow-x-auto">
                    <table class="w-full text-left whitespace-nowrap text-sm">
                        <thead class="bg-slate-50 border-b border-slate-200 text-xs uppercase tracking-wider text-slate-500 font-bold">
                            <tr>
                                <th class="px-6 py-4 text-center border-r border-slate-200 w-20">Seed</th>
                                <th class="px-6 py-4 border-r border-slate-200">Team</th>
                                <th class="px-6 py-4 text-center border-r border-slate-200">Bid</th>
                                <th class="px-6 py-4 text-right border-r border-slate-200">Score</th>
                                <th class="px-6 py-4 text-right text-purple-700">RQ</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 bg-white">
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </main>

        <footer class="mt-auto py-8 text-center border-t border-slate-200 bg-white">
            <p class="text-slate-400 text-xs font-mono">CHan Analytics &bull; Automated Data System</p>
        </footer>
    </body>
    </html>
    """
    return template

if __name__ == "__main__":
    generate_bracket()
