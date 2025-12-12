import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://www.sports-reference.com"
SCHOOLS_URL = f"{BASE_URL}/cbb/schools/"
SEASON = 2026
OUTPUT_DIR = "schedules"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CBB-Schedule-Scraper/1.0)"
}


def get_d1_schools_2026():
    print("Fetching schools list...")

    resp = requests.get(SCHOOLS_URL, headers=HEADERS)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # ✅ Find the correct table via its caption
    table = None
    for t in soup.find_all("table"):
        caption = t.find("caption")
        if caption and "Men's Schools" in caption.text:
            table = t
            break

    if table is None:
        raise RuntimeError("Could not locate Men's Schools table.")

    schools = []

    for row in table.tbody.find_all("tr"):
        school_cell = row.find("td", {"data-stat": "school_name"})
        year_max = row.find("td", {"data-stat": "year_max"})

        if not school_cell or not year_max:
            continue

        # ✅ Filter to active D1 teams in 2026
        if year_max.text.strip() != "2026":
            continue

        link = school_cell.find("a")
        if not link:
            continue

        schools.append({
            "Team": link.text.strip(),
            "BaseURL": BASE_URL + link["href"]
        })

    print(f"Found {len(schools)} active D1 programs.")
    return pd.DataFrame(schools)




def scrape_team_schedule(team, base_url):
    """
    Scrape 2025–26 schedule for a single team.
    """
    schedule_url = f"{base_url}{SEASON}-schedule.html"
    resp = requests.get(schedule_url, headers=HEADERS)

    if resp.status_code != 200:
        print(f"  ⚠️  Schedule not found for {team}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="schedule")

    if table is None:
        print(f"  ⚠️  No schedule table for {team}")
        return None

    games = []

    for row in table.tbody.find_all("tr"):
        if row.get("class") == ["thead"]:
            continue

        opp_cell = row.find("td", {"data-stat": "opp_name"})
        loc_cell = row.find("td", {"data-stat": "game_location"})
        date_cell = row.find("td", {"data-stat": "date_game"})

        if not opp_cell:
            continue

        opponent = opp_cell.text.strip()
        raw_loc = loc_cell.text.strip() if loc_cell else ""

        # Sports Reference location logic
        if raw_loc == "@":
            loc = "A"
        elif raw_loc == "N":
            loc = "N"
        else:
            loc = "H"

        games.append({
            "Opponent": opponent,
            "Location": loc
        })

    return pd.DataFrame(games)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    schools = get_d1_schools_2026()

    for i, row in schools.iterrows():
        team = row["Team"]
        url = row["BaseURL"]

        print(f"[{i+1}/{len(schools)}] Scraping {team}...")
        sched = scrape_team_schedule(team, url)

        if sched is not None and not sched.empty:
            safe_name = (
                team.replace(" ", "_")
                    .replace(".", "")
                    .replace("'", "")
                    .replace("&", "and")
            )
            out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{SEASON}.csv")
            sched.to_csv(out_path, index=False)

        # Polite delay – Sports Reference is strict
        time.sleep(3)

    print("\n✅ All schedules scraped successfully.")


if __name__ == "__main__":
    main()
