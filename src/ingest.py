import os
import sys
import json
import requests

# Ensure we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.puntos import SPRINT_ROUNDS_2026

BASE_URL = "https://api.jolpi.ca/ergast/f1"
ACTUAL_DIR = "data/actual"

def fetch_data(endpoint):
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, params={"limit": 100})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching {url}: {response.status_code}")
        return None

def save_json(data, filename):
    if data:
        os.makedirs(ACTUAL_DIR, exist_ok=True)
        path = os.path.join(ACTUAL_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Saved {path}")

def fetch_calendar(year=2026):
    data = fetch_data(f"{year}.json")
    save_json(data, f"calendar_{year}.json")

def fetch_standings(year=2026, round=None):
    if round:
        data = fetch_data(f"{year}/{round}/driverStandings.json")
        save_json(data, f"standings_r{round:02d}.json")
    else:
        data = fetch_data(f"{year}/driverStandings.json")
        save_json(data, f"standings_current.json")

def fetch_results(year=2026, round=1):
    data = fetch_data(f"{year}/{round}/results.json")
    save_json(data, f"results_r{round:02d}.json")

def fetch_sprint(year=2026, round=1):
    data = fetch_data(f"{year}/{round}/sprint.json")
    save_json(data, f"sprint_r{round:02d}.json")

if __name__ == "__main__":
    print("Ingesting data from Jolpica API...")
    fetch_calendar(2026)
    fetch_standings(2026)
    
    # Get the latest standings to see how many rounds have passed
    data = fetch_data("2026/driverStandings.json")
    if data:
        try:
            standings_list = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if standings_list:
                latest_round = int(standings_list[0]["round"])
                for r in range(1, latest_round + 1):
                    fetch_results(2026, r)
                    if r in SPRINT_ROUNDS_2026:
                        fetch_sprint(2026, r)
            else:
                print("No standings available yet for 2026.")
        except Exception as e:
            print(f"Error parsing standings to fetch results: {e}")
