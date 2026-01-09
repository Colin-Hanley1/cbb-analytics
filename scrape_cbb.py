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
    "Tarleton State": "Tarleton St.",
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
    "Tarleton State": "Tarleton St.",
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
        self.delay = 4.0
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        self.base_url = "https://www.sports-reference.com"
        self.stat_fields = [
            'mp', 'fg', 'fga', 'fg_pct', 'fg3', 'fg3a', 'fg3_pct',
            'ft', 'fta', 'ft_pct', 'orb', 'drb', 'trb',
            'ast', 'stl', 'blk', 'tov', 'pf', 'pts'
        ]

    def get_todays_schedule(self):
        """Scrapes the matchups for the current date."""
        now = datetime.now()
        # Note: Sports-Reference determines 'today' based on their server time.
        url = f"{self.base_url}/cbb/boxscores/index.cgi?month={now.month}&day={now.day}&year={now.year}"
        print(f"Fetching schedule for today: {now.strftime('%Y-%m-%d')}")
        
        try:
            response = requests.get(url, headers=self.headers)
        except:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        games = []
        
        # Look for game summaries
        summary_divs = soup.find_all('div', class_='game_summary')
        if summary_divs is None:
            return games
        
        for div in summary_divs:
            # Skip women's games if present
            if 'gender-f' in div.get('class', []): continue
            
            table = div.find('table')
            if not table: continue
            
            rows = table.find_all('tr')
            if len(rows) < 2: continue
            
            # Extract team names
            # Usually row 0 is team A, row 1 is team B
            try:
                teamA_raw = rows[0].find('a').text
                teamB_raw = rows[1].find('a').text
                
                # Check for location (Neutral site logic is hard to scrape perfectly here,
                # we will assume Home/Away based on listing order or simple logic)
                # SR usually lists Away first (top), Home second (bottom)
                games.append({
                    'away': self.clean_team_name2(teamA_raw),
                    'home': self.clean_team_name2(teamB_raw)
                })
            except:
                continue
        return games

    def clean_team_name(self, name_str):
        # 1. Clean raw text
        name_str = name_str.replace('Table', '')
        name_str = re.sub(r'\s*\(\d+-\d+\)', '', name_str)  # Remove record like (7-1)
        name_str = re.sub(r'\s*\(\d+\)', '', name_str)      # Remove rank like (25)
        name_str = name_str.strip()
        
        # 2. Apply Mapping
        # If the cleaned name is in our dictionary, swap it.
        # Otherwise, keep it as is.
        if name_str in NAME_MAPPING:
            return NAME_MAPPING[name_str]
        return name_str

    def clean_team_name2(self, name_str):
        # 1. Clean raw text
        name_str = name_str.replace('Table', '')
        name_str = re.sub(r'\s*\(\d+-\d+\)', '', name_str)  # Remove record like (7-1)
        name_str = re.sub(r'\s*\(\d+\)', '', name_str)      # Remove rank like (25)
        name_str = name_str.strip()
        
        # 2. Apply Mapping
        # If the cleaned name is in our dictionary, swap it.
        # Otherwise, keep it as is.
        if name_str in TODAY_NAME_MAPPING:
            return TODAY_NAME_MAPPING[name_str]
        return name_str

    def get_mens_game_links(self, date_obj):
        url = f"{self.base_url}/cbb/boxscores/index.cgi?month={date_obj.month}&day={date_obj.day}&year={date_obj.year}"
        print(f"Fetching schedule for: {date_obj.strftime('%Y-%m-%d')}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
        except Exception as e:
            print(f" Error fetching schedule: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        games = soup.find_all('div', class_='gender-m')

        for game in games:
            # --- D1 FILTERING LOGIC ---
            teams_table = game.find('table', class_='teams')
            if not teams_table:
                continue

            team_rows = teams_table.find_all('tr', limit=2)
            is_d1_matchup = True
            
            if len(team_rows) < 2:
                is_d1_matchup = False
            else:
                for row in team_rows:
                    td = row.find('td')
                    if td:
                        anchor = td.find('a')
                        if not anchor or 'href' not in anchor.attrs or '/cbb/schools/' not in anchor['href']:
                            is_d1_matchup = False
                            break
                    else:
                        is_d1_matchup = False

            if not is_d1_matchup:
                continue
            # ---------------------------

            link_cell = game.find('td', class_='gamelink')
            if link_cell:
                anchor = link_cell.find('a')
                if anchor and 'href' in anchor.attrs:
                    links.append(self.base_url + anchor['href'])
        
        return links

    def parse_box_score(self, url, date_str):
        print(f" Scraping game: {url}...")
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 429:
                print(" !! RATE LIMITED. Sleeping for 60 seconds.")
                time.sleep(60)
                response = requests.get(url, headers=self.headers)
            
            soup = BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f" Error fetching box score: {e}")
            return []

        scorebox = soup.find('div', class_='scorebox')
        team_names = []
        if scorebox:
            team_names = [a.text for a in scorebox.find_all('a', itemprop="name")]
        
        tables = soup.find_all('table', id=lambda x: x and x.startswith('box-score-basic'))

        if len(tables) < 2:
            print(" Could not find both box score tables.")
            return []

        if len(team_names) < 2:
            team_names = []
            for table in tables[:2]:
                caption = table.find('caption')
                if caption:
                    raw_name = caption.text
                    if 'Basic Box' in raw_name:
                        raw_name = raw_name.split('Basic Box')[0]
                    team_names.append(raw_name)
                else:
                    team_names.append("Unknown Team")

        # Apply the cleaning AND mapping here
        team_names = [self.clean_team_name(n) for n in team_names]
        game_data = []

        for i, table in enumerate(tables[:2]):
            team_name = team_names[i]
            opponent_name = team_names[1] if i == 0 else team_names[0]
            is_home = 1 if i == 1 else 0

            tfoot = table.find('tfoot')
            if not tfoot:
                continue
            
            row = tfoot.find('tr')
            team_stats = {
                'date': date_str,
                'team': team_name,
                'opponent': opponent_name,
                'location_value': 1 if is_home else -1
            }

            for stat in self.stat_fields:
                cell = row.find('td', {'data-stat': stat})
                if cell and cell.text.strip():
                    try:
                        val = float(cell.text)
                        if val.is_integer():
                            val = int(val)
                        team_stats[stat] = val
                    except ValueError:
                        team_stats[stat] = 0
                else:
                    team_stats[stat] = 0

            fga = team_stats.get('fga', 0)
            orb = team_stats.get('orb', 0)
            tov = team_stats.get('tov', 0)
            fta = team_stats.get('fta', 0)
            pts = team_stats.get('pts', 0)

            possessions = fga - orb + tov + (0.475 * fta)
            raw_off_eff = (pts / possessions * 100) if possessions > 0 else 0

            team_stats['possessions'] = round(possessions, 2)
            team_stats['raw_off_eff'] = round(raw_off_eff, 2)

            game_data.append(team_stats)

        time.sleep(self.delay + random.random())
        return game_data

    def run(self, start_date, end_date, output_filename="cbb_scores.csv"):
        current_date = start_date
        total_games_scraped = 0

        # Loop through each day
        while current_date <= end_date:
            day_games = []  # Reset list for the new day
            links = self.get_mens_game_links(current_date)
            print(f" > Found {len(links)} D1 vs D1 matchups.")
            i = 0
            for link in links:
                i += 1
                print(f"  [{i}/{len(links)}]", end="")
                game_stats = self.parse_box_score(link, current_date.strftime('%Y-%m-%d'))
                
                day_games.extend(game_stats)
            
            # --- SAVE DATA AT THE END OF THE DAY ---
            if day_games:
                df_day = pd.DataFrame(day_games)
                # Check if file exists. If it doesn't, we write headers.
                # If it does, we append (mode='a') and skip headers.
                file_exists = os.path.isfile(output_filename)
                
                try:
                    df_day.to_csv(output_filename, mode='a', header=not file_exists, index=False)
                    print(f" [SAVED] Appended {len(day_games)} rows to {output_filename}")
                    total_games_scraped += len(day_games)
                except Exception as e:
                    print(f" !! ERROR SAVING FILE: {e}")
            else:
                print(" No data to save for this date.")
            # ---------------------------------------

            current_date += timedelta(days=1)
        
        print(f"\nJob Complete. Total rows saved: {total_games_scraped}")


if __name__ == "__main__":
    scraper = CBBScraper()
    output_filename = "cbb_scores.csv"

    # 1. Calculate Yesterday's Date
    # Sports-Reference updates overnight, so we always want "Yesterday" relative to now.
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    sd = datetime(2026,1,6)
    ed = datetime(2026,1,8)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    print(f"Checking scrape status for: {yesterday_str}")

    # 2. Check if Yesterday is already in the CSV
    already_scraped = False
    if os.path.exists(output_filename):
        try:
            # We read the file to check existing dates
            # (If file is huge, reading unique dates is still reasonably fast for this size)
            existing_df = pd.read_csv(output_filename)
            if 'date' in existing_df.columns:
                existing_dates = set(existing_df['date'].astype(str).unique())
                if yesterday_str in existing_dates:
                    already_scraped = True
        except Exception as e:
            print(f"Warning: Could not read {output_filename} to verify dates. Proceeding with scrape.")

    # 3. Execute Scrape if needed
    if already_scraped:
        print(f" > Data for {yesterday_str} is already present in {output_filename}. Skipping scrape.")
    else:
        print(f" > Data for {yesterday_str} not found. Starting scraper...")
        scraper.run(yesterday, yesterday, output_filename=output_filename)