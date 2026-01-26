import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
import random
import re
import os

# --- CONFIGURATION: NAME MAPPING ---
NAME_MAPPING ={
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
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
        self.base_url = "https://www.sports-reference.com"

    # --------------------------
    # Name cleaning / mapping
    # --------------------------
    def clean_team_name(self, name_str: str) -> str:
        if name_str is None:
            return ""
        s = str(name_str)
        s = s.replace("Table", "")
        s = re.sub(r"\s*\(\d+-\d+\)", "", s)  # remove record (7-1)
        s = re.sub(r"\s*\(\d+\)", "", s)      # remove rank (25)
        s = s.strip()
        return NAME_MAPPING.get(s, s)

    # --------------------------
    # Schedule scraping helpers
    # --------------------------
    def _schedule_url(self, date_obj: datetime) -> str:
        return f"{self.base_url}/cbb/boxscores/index.cgi?month={date_obj.month}&day={date_obj.day}&year={date_obj.year}"

    def scrape_schedule_for_date(self, date_obj: datetime):
        """
        Scrapes ALL men's games listed on Sports-Reference for a given date.
        Returns a list of matchup dicts: {date, away, home, neutral?, game_url?}
        """
        url = self._schedule_url(date_obj)
        date_str = date_obj.strftime("%Y-%m-%d")
        print(f"Fetching schedule for: {date_str}")

        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            if resp.status_code == 429:
                print(" !! RATE LIMITED. Sleeping for 60 seconds.")
                time.sleep(60)
                resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f" Error fetching schedule page: {e}")
            return []

        soup = BeautifulSoup(resp.content, "html.parser")

        # Sports-Reference typically separates men's games under divs with class 'gender-m'
        games = soup.find_all("div", class_="gender-m")
        matchups = []

        for gdiv in games:
            teams_table = gdiv.find("table", class_="teams")
            if not teams_table:
                continue

            rows = teams_table.find_all("tr")
            # The first two meaningful rows should be the two teams
            team_rows = []
            for r in rows:
                a = r.find("a")
                if a and a.text:
                    team_rows.append(r)
                if len(team_rows) >= 2:
                    break
            if len(team_rows) < 2:
                continue

            # Extract team names
            try:
                away_raw = team_rows[0].find("a").text
                home_raw = team_rows[1].find("a").text
            except Exception:
                continue

            away = self.clean_team_name(away_raw)
            home = self.clean_team_name(home_raw)

            # Attempt to capture a game link (sometimes preview / boxscore)
            game_url = None
            link_cell = gdiv.find("td", class_="gamelink")
            if link_cell:
                a = link_cell.find("a")
                if a and a.get("href"):
                    game_url = self.base_url + a["href"]

            # Neutral detection on SR index pages is inconsistent.
            # Heuristic: if the game summary text contains "at " with a venue and no clear home listing,
            # SR still lists teams top/bottom, so we keep away/home but allow neutral=1 if we see "Neutral" keyword.
            neutral_flag = 0
            text_blob = gdiv.get_text(" ", strip=True).lower()
            if "neutral" in text_blob:
                neutral_flag = 1

            matchups.append({
                "date": date_str,
                "away": away,
                "home": home,
                "neutral": int(neutral_flag),
                "game_url": game_url if game_url else ""
            })

        # polite delay
        time.sleep(self.delay + random.random())
        return matchups

    def run_schedule(self, start_date: datetime, end_date: datetime, output_filename="schedule.csv"):
        """
        Writes a season schedule into schedule.csv in TEAM-ROW format:
          date, team, opponent, location_value, neutral, game_url, game_id

        For each matchup we write TWO rows:
          away-team row: location_value = -1 (unless neutral -> 0)
          home-team row: location_value =  1 (unless neutral -> 0)

        Also dedupes using a canonical game_id.
        """
        current_date = start_date
        all_rows = []
        total_rows = 0

        while current_date <= end_date:
            matchups = self.scrape_schedule_for_date(current_date)

            for m in matchups:
                date_str = m["date"]
                away = m["away"]
                home = m["home"]
                neutral = int(m.get("neutral", 0))
                game_url = m.get("game_url", "")

                # canonical game id (order invariant)
                a, b = (away, home) if away <= home else (home, away)
                game_id = f"{date_str}__{a}__{b}"

                if neutral == 1:
                    away_loc = 0
                    home_loc = 0
                else:
                    away_loc = -1
                    home_loc = 1

                # Away row
                all_rows.append({
                    "date": date_str,
                    "team": away,
                    "opponent": home,
                    "location_value": away_loc,
                    "neutral": neutral,
                    "game_url": game_url,
                    "game_id": game_id
                })

                # Home row
                all_rows.append({
                    "date": date_str,
                    "team": home,
                    "opponent": away,
                    "location_value": home_loc,
                    "neutral": neutral,
                    "game_url": game_url,
                    "game_id": game_id
                })

            current_date += timedelta(days=1)

        if not all_rows:
            print("No schedule rows found for the requested range.")
            return

        df_new = pd.DataFrame(all_rows)

        # Deduplicate within this run (just in case)
        df_new = df_new.drop_duplicates(subset=["team", "game_id"], keep="first")

        # If file exists, merge/dedupe against existing
        if os.path.exists(output_filename):
            try:
                df_old = pd.read_csv(output_filename)
                # Ensure required columns exist
                for col in ["date", "team", "opponent", "location_value", "neutral", "game_url", "game_id"]:
                    if col not in df_old.columns:
                        df_old[col] = ""
                df_all = pd.concat([df_old, df_new], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["team", "game_id"], keep="first")
            except Exception as e:
                print(f"Warning: Could not read existing {output_filename} ({e}). Overwriting.")
                df_all = df_new
        else:
            df_all = df_new

        # Sort nicely
        df_all["date_dt"] = pd.to_datetime(df_all["date"], errors="coerce")
        df_all = df_all.sort_values(["date_dt", "team"], ascending=[True, True]).drop(columns=["date_dt"])

        try:
            df_all.to_csv(output_filename, index=False)
            total_rows = len(df_all)
            print(f"[SAVED] {output_filename} rows: {total_rows}")
        except Exception as e:
            print(f"!! ERROR SAVING FILE: {e}")

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    scraper = CBBScraper()

    # Choose the season date range you want.
    # You can adjust these to match your season window.
    # For example: Nov 1 -> Apr 15
    season_start = datetime(2026, 1, 25)
    season_end   = datetime(2026, 4, 15)

    scraper.run_schedule(season_start, season_end, output_filename="schedule.csv")
