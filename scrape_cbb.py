import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
import random
import re
import os

# --- CONFIGURATION: NAME MAPPING ---
# Maps Sports-Reference names to your preferred naming convention
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
   "Tarleton State": "Tarleton St." ,
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

TODAY_NAME_MAPPING = {
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
   "Tarleton State": "Tarleton St." ,
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
   "Mississippi Valley State": "Mississippi Valley St.",
   "ETSU": "East Tennessee St.",
   "UNC": "North Carolina",
   "UMass": "Massachusetts",
   "Southern Miss": "Southern Mississippi",
   "UT-Martin": "Tennessee Martin",
   "UC-Riverside": "UC Riverside",
   "UConn": "Connecticut",
   "UMass-Lowell": "UMass Lowell",
   "UIC": "Illinois Chicago",
   "Pitt": "Pittsburgh",
   "Ole Miss": "Mississippi",
   "UC-Davis": "UC Davis",
   "UC-Irvine": "UC Irvine",
   "UCSB": "UC Santa Barbara",
   "SIU-Edwardsville": "SIUE",
   "St. Joseph's": "Saint Joseph's", 
   "IU Indianapolis": "IU Indy"
   
   
}

class CBBScraper:
    def __init__(self):
        self.delay = 3.0 
        self.headers = {
            # Updated to a newer User Agent to avoid detection
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        self.base_url = "https://www.sports-reference.com"
        
        self.stat_fields = [
            'mp', 'fg', 'fga', 'fg_pct', 'fg3', 'fg3a', 'fg3_pct', 
            'ft', 'fta', 'ft_pct', 'orb', 'drb', 'trb', 
            'ast', 'stl', 'blk', 'tov', 'pf', 'pts'
        ]

    def clean_team_name(self, name_str):
        name_str = str(name_str).replace('Table', '')
        name_str = re.sub(r'\s*\(\d+-\d+\)', '', name_str)
        name_str = re.sub(r'\s*\(\d+\)', '', name_str)
        name_str = name_str.strip()
        if name_str in NAME_MAPPING: return NAME_MAPPING[name_str]
        return name_str

    def get_todays_schedule(self):
        """Scrapes the matchups for the current date with robust fallback logic."""
        
        # Use Local System Time (Simplest for desktop running)
        now = datetime.now()
        url = f"{self.base_url}/cbb/boxscores/index.cgi?month={now.month}&day={now.day}&year={now.year}"
        print(f"Fetching schedule for: {now.strftime('%Y-%m-%d')} ({url})")
        
        try:
            response = requests.get(url, headers=self.headers)
            # Check for redirect (The URL changes if SR redirects you)
            if response.url != url and "boxscores/index.cgi" not in response.url:
                 # Parse the new URL to see if the date changed
                 print(f"  > Notice: URL redirected to {response.url}")
        except Exception as e:
            print(f"  > Connection Error: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check "No Games Found" message
        content_div = soup.find('div', id='content')
        if content_div and "No games found" in content_div.text:
             print("  > Site explicitly says 'No games found'.")
             return []

        games = []
        summary_divs = soup.find_all('div', class_='game_summary')
        
        print(f"  > Found {len(summary_divs)} game summary blocks.")
        
        for div in summary_divs:
            if 'gender-f' in div.get('class', []): continue
            
            table = div.find('table')
            if not table: continue
            
            rows = table.find_all('tr')
            if len(rows) < 2: continue
            
            try:
                # Robust extraction: Try Link first, failover to Text
                def get_name(row):
                    link = row.find('a')
                    if link: return link.text.strip()
                    # If no link (Non-D1), get text from first cell
                    td = row.find('td')
                    if td: return td.text.strip()
                    return "Unknown"

                teamA_raw = get_name(rows[0])
                teamB_raw = get_name(rows[1])
                
                # Clean names
                # Use standard clean_team_name if TODAY mapping is empty/missing
                tA = self.clean_team_name(teamA_raw)
                tB = self.clean_team_name(teamB_raw)
                
                games.append({'away': tA, 'home': tB})
            except Exception as e:
                print(f"  > Error parsing a game row: {e}")
                continue
                
        print(f"  > Successfully extracted {len(games)} matchups.")
        return games

    # ... (Keep get_mens_game_links, parse_box_score, run methods as they were) ...
    # Be sure to keep the 'run' method for your historical scrapes!
    
    def get_mens_game_links(self, date_obj):
        # Reuse previous logic or paste from older file
        url = f"{self.base_url}/cbb/boxscores/index.cgi?month={date_obj.month}&day={date_obj.day}&year={date_obj.year}"
        try:
            response = requests.get(url, headers=self.headers)
        except: return []
        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        games = soup.find_all('div', class_='gender-m')
        for game in games:
            link_cell = game.find('td', class_='gamelink')
            if link_cell:
                anchor = link_cell.find('a')
                if anchor and 'href' in anchor.attrs:
                    links.append(self.base_url + anchor['href'])
        return links

    def parse_box_score(self, url, date_str):
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 429:
                time.sleep(10)
                response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
        except: return []

        scorebox = soup.find('div', class_='scorebox')
        if not scorebox: return []
        
        team_names = [a.text for a in scorebox.find_all('a', itemprop="name")]
        team_names = [self.clean_team_name(n) for n in team_names]
        tables = soup.find_all('table', id=lambda x: x and x.startswith('box-score-basic'))

        if len(tables) < 2 or len(team_names) < 2: return []
            
        game_data = []
        for i, table in enumerate(tables[:2]):
            t_name = team_names[i]
            o_name = team_names[1] if i == 0 else team_names[0]
            loc_val = 1 if i == 1 else -1

            tfoot = table.find('tfoot')
            if not tfoot: continue
            row = tfoot.find('tr')
            
            team_stats = {'date': date_str, 'team': t_name, 'opponent': o_name, 'location_value': loc_val}
            for stat in self.stat_fields:
                cell = row.find('td', {'data-stat': stat})
                if cell and cell.text.strip():
                    try: team_stats[stat] = float(cell.text)
                    except: team_stats[stat] = 0
                else: team_stats[stat] = 0
            
            # Simple Poss calc
            fga, orb, tov, fta, pts = team_stats.get('fga',0), team_stats.get('orb',0), team_stats.get('tov',0), team_stats.get('fta',0), team_stats.get('pts',0)
            poss = fga - orb + tov + (0.475 * fta)
            team_stats['possessions'] = round(poss, 2)
            team_stats['raw_off_eff'] = round((pts/poss)*100, 2) if poss > 0 else 0
            
            game_data.append(team_stats)
        
        time.sleep(self.delay)
        return game_data

    def run(self, start_date, end_date, output_filename="cbb_scores.csv"):
        # Simple loop for historical data
        curr = start_date
        while curr <= end_date:
            d_str = curr.strftime('%Y-%m-%d')
            links = self.get_mens_game_links(curr)
            print(f"Date: {d_str} | Games: {len(links)}")
            
            day_data = []
            for l in links:
                day_data.extend(self.parse_box_score(l, d_str))
            
            if day_data:
                df = pd.DataFrame(day_data)
                exists = os.path.isfile(output_filename)
                df.to_csv(output_filename, mode='a', header=not exists, index=False)
            
            curr += timedelta(days=1)

if __name__ == "__main__":
    # Test the schedule fetcher immediately
    s = CBBScraper()
    games = s.get_todays_schedule()
    print("\n--- Games Found ---")
    for g in games:
        print(f"{g['away']} @ {g['home']}")