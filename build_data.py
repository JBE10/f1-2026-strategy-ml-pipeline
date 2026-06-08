#!/usr/bin/env python3
"""
Build the data/raw CSV files and update data/actual from the Jolpica API.
This replaces the Kaggle/Ergast CSV download.
"""
import os
import sys
import json
import time
import requests
import csv

BASE_URL = "https://api.jolpi.ca/ergast/f1"
RAW_DIR = "data/raw"
ACTUAL_DIR = "data/actual"

# Sprint rounds for 2026
SPRINT_ROUNDS_2026 = [2, 4, 5, 9, 12, 16]

def fetch(endpoint, retries=3):
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params={"limit": 1000}, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                print(f"  Rate limited, waiting 5s...")
                time.sleep(5)
            else:
                print(f"  HTTP {r.status_code} for {url}")
                return None
        except Exception as e:
            print(f"  Error: {e}, retrying...")
            time.sleep(2)
    return None

def save_json(data, filename):
    if data:
        os.makedirs(ACTUAL_DIR, exist_ok=True)
        path = os.path.join(ACTUAL_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"  Saved {path}")

# ─── STEP 1: Build circuits.csv and races.csv ──────────────────────────────
def build_races_and_circuits():
    print("\n=== Building races.csv and circuits.csv ===")
    
    all_races = []
    circuits = {}
    circuit_id_counter = 1
    race_id_counter = 1
    
    for year in [2023, 2024, 2025, 2026]:
        print(f"  Fetching {year} calendar...")
        data = fetch(f"{year}.json")
        if not data:
            print(f"  FAILED to fetch {year} calendar!")
            continue
        
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        print(f"  Got {len(races)} races for {year}")
        
        for race in races:
            circuit_ref = race["Circuit"]["circuitId"]
            
            if circuit_ref not in circuits:
                circuits[circuit_ref] = {
                    "circuitId": circuit_id_counter,
                    "circuitRef": circuit_ref,
                    "name": race["Circuit"]["circuitName"],
                    "location": race["Circuit"]["Location"]["locality"],
                    "country": race["Circuit"]["Location"]["country"],
                    "lat": race["Circuit"]["Location"]["lat"],
                    "lng": race["Circuit"]["Location"]["long"],
                    "alt": "",
                    "url": race["Circuit"]["url"]
                }
                circuit_id_counter += 1
            
            all_races.append({
                "raceId": race_id_counter,
                "year": int(race["season"]),
                "round": int(race["round"]),
                "circuitId": circuits[circuit_ref]["circuitId"],
                "name": race["raceName"],
                "date": race["date"],
                "time": race.get("time", ""),
                "url": race["url"],
                "fp1_date": "", "fp1_time": "",
                "fp2_date": "", "fp2_time": "",
                "fp3_date": "", "fp3_time": "",
                "quali_date": "", "quali_time": "",
                "sprint_date": "", "sprint_time": ""
            })
            race_id_counter += 1
        
        time.sleep(0.5)
    
    # Write circuits.csv
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, "circuits.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["circuitId", "circuitRef", "name", "location", "country", "lat", "lng", "alt", "url"])
        w.writeheader()
        for c in circuits.values():
            w.writerow(c)
    print(f"  circuits.csv: {len(circuits)} circuits")
    
    # Write races.csv
    with open(os.path.join(RAW_DIR, "races.csv"), 'w', newline='') as f:
        fields = ["raceId", "year", "round", "circuitId", "name", "date", "time", "url",
                  "fp1_date", "fp1_time", "fp2_date", "fp2_time", "fp3_date", "fp3_time",
                  "quali_date", "quali_time", "sprint_date", "sprint_time"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_races:
            w.writerow(r)
    print(f"  races.csv: {len(all_races)} races")
    
    return all_races, circuits

# ─── STEP 2: Build results.csv, drivers.csv, constructors.csv, status.csv ──
def build_results_and_drivers(all_races, circuits):
    print("\n=== Building results.csv, drivers.csv, constructors.csv, status.csv, sprint_results.csv ===")
    
    all_results = []
    all_sprint_results = []
    drivers = {}
    constructors = {}
    statuses = {"Finished": 1}
    driver_id_counter = 1
    constructor_id_counter = 1
    status_id_counter = 2
    result_id_counter = 1
    sprint_result_id_counter = 1
    
    for race in all_races:
        year = race["year"]
        rnd = race["round"]
        race_id = race["raceId"]
        
        # Only fetch 2023-2025 historical + already-completed 2026 races
        if year == 2026:
            continue  # 2026 actual data handled separately by ingest.py
        
        print(f"  Fetching {year} R{rnd}...")
        data = fetch(f"{year}/{rnd}/results.json")
        if not data:
            time.sleep(1)
            continue
        
        races_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races_data:
            time.sleep(0.3)
            continue
        
        results = races_data[0].get("Results", [])
        
        for res in results:
            drv_ref = res["Driver"]["driverId"]
            if drv_ref not in drivers:
                drivers[drv_ref] = {
                    "driverId": driver_id_counter,
                    "driverRef": drv_ref,
                    "number": res["Driver"].get("permanentNumber", ""),
                    "code": res["Driver"].get("code", ""),
                    "forename": res["Driver"].get("givenName", ""),
                    "surname": res["Driver"].get("familyName", ""),
                    "dob": res["Driver"].get("dateOfBirth", ""),
                    "nationality": res["Driver"].get("nationality", ""),
                    "url": res["Driver"].get("url", "")
                }
                driver_id_counter += 1
            
            con_ref = res["Constructor"]["constructorId"]
            if con_ref not in constructors:
                constructors[con_ref] = {
                    "constructorId": constructor_id_counter,
                    "constructorRef": con_ref,
                    "name": res["Constructor"].get("name", ""),
                    "nationality": res["Constructor"].get("nationality", ""),
                    "url": res["Constructor"].get("url", "")
                }
                constructor_id_counter += 1
            
            status_text = res.get("status", "Finished")
            if status_text not in statuses:
                statuses[status_text] = status_id_counter
                status_id_counter += 1
            
            pos = res.get("position", "\\N")
            grid = res.get("grid", "0")
            
            all_results.append({
                "resultId": result_id_counter,
                "raceId": race_id,
                "driverId": drivers[drv_ref]["driverId"],
                "constructorId": constructors[con_ref]["constructorId"],
                "number": res.get("number", ""),
                "grid": grid,
                "position": pos,
                "positionText": res.get("positionText", pos),
                "positionOrder": pos if pos != "\\N" else "20",
                "points": res.get("points", "0"),
                "laps": res.get("laps", "0"),
                "time": res.get("Time", {}).get("time", "") if isinstance(res.get("Time"), dict) else "",
                "milliseconds": res.get("Time", {}).get("millis", "") if isinstance(res.get("Time"), dict) else "",
                "fastestLap": res.get("FastestLap", {}).get("lap", "") if isinstance(res.get("FastestLap"), dict) else "",
                "rank": res.get("FastestLap", {}).get("rank", "") if isinstance(res.get("FastestLap"), dict) else "",
                "fastestLapTime": res.get("FastestLap", {}).get("Time", {}).get("time", "") if isinstance(res.get("FastestLap"), dict) else "",
                "fastestLapSpeed": res.get("FastestLap", {}).get("AverageSpeed", {}).get("speed", "") if isinstance(res.get("FastestLap"), dict) else "",
                "statusId": statuses[status_text]
            })
            result_id_counter += 1
        
        # Check for sprint results
        sprint_data = fetch(f"{year}/{rnd}/sprint.json")
        if sprint_data:
            sprint_races = sprint_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if sprint_races:
                sprint_results = sprint_races[0].get("SprintResults", [])
                for res in sprint_results:
                    drv_ref = res["Driver"]["driverId"]
                    con_ref = res["Constructor"]["constructorId"]
                    
                    # Ensure driver/constructor exist
                    if drv_ref not in drivers:
                        drivers[drv_ref] = {
                            "driverId": driver_id_counter,
                            "driverRef": drv_ref,
                            "number": res["Driver"].get("permanentNumber", ""),
                            "code": res["Driver"].get("code", ""),
                            "forename": res["Driver"].get("givenName", ""),
                            "surname": res["Driver"].get("familyName", ""),
                            "dob": res["Driver"].get("dateOfBirth", ""),
                            "nationality": res["Driver"].get("nationality", ""),
                            "url": res["Driver"].get("url", "")
                        }
                        driver_id_counter += 1
                    if con_ref not in constructors:
                        constructors[con_ref] = {
                            "constructorId": constructor_id_counter,
                            "constructorRef": con_ref,
                            "name": res["Constructor"].get("name", ""),
                            "nationality": res["Constructor"].get("nationality", ""),
                            "url": res["Constructor"].get("url", "")
                        }
                        constructor_id_counter += 1
                    
                    status_text = res.get("status", "Finished")
                    if status_text not in statuses:
                        statuses[status_text] = status_id_counter
                        status_id_counter += 1
                    
                    pos = res.get("position", "\\N")
                    all_sprint_results.append({
                        "resultId": sprint_result_id_counter,
                        "raceId": race_id,
                        "driverId": drivers[drv_ref]["driverId"],
                        "constructorId": constructors[con_ref]["constructorId"],
                        "number": res.get("number", ""),
                        "grid": res.get("grid", "0"),
                        "position": pos,
                        "positionText": res.get("positionText", pos),
                        "positionOrder": pos if pos != "\\N" else "20",
                        "points": res.get("points", "0"),
                        "laps": res.get("laps", "0"),
                        "time": res.get("Time", {}).get("time", "") if isinstance(res.get("Time"), dict) else "",
                        "milliseconds": res.get("Time", {}).get("millis", "") if isinstance(res.get("Time"), dict) else "",
                        "fastestLap": "",
                        "rank": "",
                        "fastestLapTime": "",
                        "fastestLapSpeed": "",
                        "statusId": statuses[status_text]
                    })
                    sprint_result_id_counter += 1
        
        time.sleep(0.4)  # Rate limit
    
    # Write CSVs
    res_fields = ["resultId", "raceId", "driverId", "constructorId", "number", "grid",
                  "position", "positionText", "positionOrder", "points", "laps", "time",
                  "milliseconds", "fastestLap", "rank", "fastestLapTime", "fastestLapSpeed", "statusId"]
    
    with open(os.path.join(RAW_DIR, "results.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=res_fields)
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"  results.csv: {len(all_results)} results")
    
    with open(os.path.join(RAW_DIR, "sprint_results.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=res_fields)
        w.writeheader()
        for r in all_sprint_results:
            w.writerow(r)
    print(f"  sprint_results.csv: {len(all_sprint_results)} results")
    
    drv_fields = ["driverId", "driverRef", "number", "code", "forename", "surname", "dob", "nationality", "url"]
    with open(os.path.join(RAW_DIR, "drivers.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=drv_fields)
        w.writeheader()
        for d in drivers.values():
            w.writerow(d)
    print(f"  drivers.csv: {len(drivers)} drivers")
    
    con_fields = ["constructorId", "constructorRef", "name", "nationality", "url"]
    with open(os.path.join(RAW_DIR, "constructors.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=con_fields)
        w.writeheader()
        for c in constructors.values():
            w.writerow(c)
    print(f"  constructors.csv: {len(constructors)} constructors")
    
    with open(os.path.join(RAW_DIR, "status.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["statusId", "status"])
        w.writeheader()
        for status, sid in statuses.items():
            w.writerow({"statusId": sid, "status": status})
    print(f"  status.csv: {len(statuses)} statuses")

# ─── STEP 3: Update actual 2026 data ──────────────────────────────────────
def update_actual_2026():
    print("\n=== Updating 2026 actual data ===")
    
    # Fetch latest calendar
    print("  Fetching 2026 calendar...")
    cal_data = fetch("2026.json")
    save_json(cal_data, "calendar_2026.json")
    
    # Fetch latest standings
    print("  Fetching current standings...")
    standings_data = fetch("2026/driverStandings.json")
    save_json(standings_data, "standings_current.json")
    
    # Determine how many rounds completed
    if standings_data:
        try:
            lists = standings_data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if lists:
                latest_round = int(lists[0]["round"])
                print(f"  Latest completed round: {latest_round}")
                
                for r in range(1, latest_round + 1):
                    print(f"  Fetching R{r} results...")
                    res_data = fetch(f"2026/{r}/results.json")
                    save_json(res_data, f"results_r{r:02d}.json")
                    time.sleep(0.3)
                    
                    if r in SPRINT_ROUNDS_2026:
                        print(f"  Fetching R{r} sprint...")
                        spr_data = fetch(f"2026/{r}/sprint.json")
                        save_json(spr_data, f"sprint_r{r:02d}.json")
                        time.sleep(0.3)
                
                return latest_round
        except Exception as e:
            print(f"  Error: {e}")
    
    return 0

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║  F1 2026 - Data Builder & Updater        ║")
    print("╚══════════════════════════════════════════╝")
    
    # Step 1: Build calendar/circuits
    all_races, circuits = build_races_and_circuits()
    
    # Step 2: Build historical results (2023-2025)
    build_results_and_drivers(all_races, circuits)
    
    # Step 3: Update 2026 actual data
    latest = update_actual_2026()
    
    print(f"\n{'='*50}")
    print(f"✅ Data build complete!")
    print(f"   Historical races: {len([r for r in all_races if r['year'] < 2026])}")
    print(f"   2026 races (calendar): {len([r for r in all_races if r['year'] == 2026])}")
    print(f"   2026 latest round: {latest}")
    print(f"   Circuits: {len(circuits)}")
    print(f"\nNext step: python src/pipeline.py")
