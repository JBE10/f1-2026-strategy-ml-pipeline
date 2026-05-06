import os
import json
import pandas as pd

RAW_DIR = "data/raw"
ACTUAL_DIR = "data/actual"

def _build_race_key(year, rnd):
    try:
        y = int(year)
        r = int(rnd)
    except Exception:
        return None
    return f"{y}-{r:02d}"

def _is_number(value):
    try:
        float(value)
        return True
    except Exception:
        return False

def get_historical_results():
    """Retorna df de results.csv unidos con races.csv"""
    results_path = os.path.join(RAW_DIR, "results.csv")
    races_path = os.path.join(RAW_DIR, "races.csv")
    drivers_path = os.path.join(RAW_DIR, "drivers.csv")
    constructors_path = os.path.join(RAW_DIR, "constructors.csv")
    status_path = os.path.join(RAW_DIR, "status.csv")
    
    if not os.path.exists(results_path) or not os.path.exists(races_path):
        return pd.DataFrame()
        
    df_res = pd.read_csv(results_path)
    df_rac = pd.read_csv(races_path)
    
    df = df_res.merge(df_rac[['raceId', 'year', 'round', 'name']], on='raceId', how='left')
    
    if os.path.exists(drivers_path):
        df_drv = pd.read_csv(drivers_path)
        df = df.merge(df_drv[['driverId', 'driverRef']], on='driverId', how='left')
        df['driverId'] = df['driverRef']
        df = df.drop(columns=['driverRef'])

    if os.path.exists(constructors_path):
        df_con = pd.read_csv(constructors_path)
        df = df.merge(df_con[['constructorId', 'constructorRef']], on='constructorId', how='left')
        df['constructorId'] = df['constructorRef']
        df = df.drop(columns=['constructorRef'])

    if os.path.exists(status_path):
        df_status = pd.read_csv(status_path)
        df = df.merge(df_status[['statusId', 'status']], on='statusId', how='left')

    df['position'] = pd.to_numeric(df.get('position'), errors='coerce')
    if 'positionOrder' in df.columns:
        df['position'] = df['position'].fillna(df['positionOrder'])
    df['event'] = 'race'
    df['raceKey'] = df.apply(lambda x: _build_race_key(x['year'], x['round']), axis=1)
        
    return df

def get_historical_sprints():
    """Retorna df de sprint_results.csv unidos con races.csv"""
    sprint_path = os.path.join(RAW_DIR, "sprint_results.csv")
    races_path = os.path.join(RAW_DIR, "races.csv")
    drivers_path = os.path.join(RAW_DIR, "drivers.csv")
    constructors_path = os.path.join(RAW_DIR, "constructors.csv")
    status_path = os.path.join(RAW_DIR, "status.csv")
    
    if not os.path.exists(sprint_path) or not os.path.exists(races_path):
        return pd.DataFrame()
        
    df_spr = pd.read_csv(sprint_path)
    df_rac = pd.read_csv(races_path)
    
    df = df_spr.merge(df_rac[['raceId', 'year', 'round', 'name']], on='raceId', how='left')
    
    if os.path.exists(drivers_path):
        df_drv = pd.read_csv(drivers_path)
        df = df.merge(df_drv[['driverId', 'driverRef']], on='driverId', how='left')
        df['driverId'] = df['driverRef']
        df = df.drop(columns=['driverRef'])

    if os.path.exists(constructors_path):
        df_con = pd.read_csv(constructors_path)
        df = df.merge(df_con[['constructorId', 'constructorRef']], on='constructorId', how='left')
        df['constructorId'] = df['constructorRef']
        df = df.drop(columns=['constructorRef'])

    if os.path.exists(status_path):
        df_status = pd.read_csv(status_path)
        df = df.merge(df_status[['statusId', 'status']], on='statusId', how='left')

    df['position'] = pd.to_numeric(df.get('position'), errors='coerce')
    if 'positionOrder' in df.columns:
        df['position'] = df['position'].fillna(df['positionOrder'])
    df['event'] = 'sprint'
    df['raceKey'] = df.apply(lambda x: _build_race_key(x['year'], x['round']), axis=1)
        
    return df

def get_2026_actual_standings():
    """Lee el json de standings actual y lo convierte a formato útil"""
    path = os.path.join(ACTUAL_DIR, "standings_current.json")
    if not os.path.exists(path):
        return pd.DataFrame()
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    try:
        lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        if not lists:
            return pd.DataFrame()
            
        standings = lists[0]["DriverStandings"]
        
        records = []
        for s in standings:
            driver_id = s["Driver"]["driverId"]
            points = float(s["points"])
            wins = int(s["wins"])
            records.append({
                "driverId": driver_id,
                "points": points,
                "wins": wins
            })
            
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error parsing actual standings: {e}")
        return pd.DataFrame()

def get_2026_actual_results():
    """Lee los jsons de resultados en actual/ y los estandariza"""
    records = []
    if not os.path.exists(ACTUAL_DIR):
        return pd.DataFrame()
        
    for f in os.listdir(ACTUAL_DIR):
        if f.startswith("results_r") and f.endswith(".json"):
            path = os.path.join(ACTUAL_DIR, f)
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            try:
                races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if not races: continue
                race = races[0]
                rnd = int(race["round"])
                results = race.get("Results", [])
                
                for r in results:
                    # En la API status puede ser Finished, +1 Lap, etc.
                    # Asignamos position, y si es un número se usa
                    pos = r.get("position", "22")
                    if _is_number(pos):
                        pos_num = int(float(pos))
                    else:
                        pos_num = 22

                    constructor_id = r.get("Constructor", {}).get("constructorId", None)
                    status = r.get("status", None)
                        
                    records.append({
                        "year": 2026,
                        "round": rnd,
                        "driverId": r["Driver"]["driverId"],
                        "constructorId": constructor_id,
                        "position": pos_num,
                        "grid": int(r.get("grid", 0)),
                        "points": float(r["points"]),
                        "status": status,
                        "event": "race",
                        "raceKey": _build_race_key(2026, rnd)
                    })
            except Exception as e:
                print(f"Error parseando {f}: {e}")
                
    return pd.DataFrame(records)

def get_2026_actual_sprints():
    """Lee los jsons de sprint en actual/ y los estandariza"""
    records = []
    if not os.path.exists(ACTUAL_DIR):
        return pd.DataFrame()

    for f in os.listdir(ACTUAL_DIR):
        if f.startswith("sprint_r") and f.endswith(".json"):
            path = os.path.join(ACTUAL_DIR, f)
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            try:
                races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if not races:
                    continue
                race = races[0]
                rnd = int(race["round"])
                results = race.get("SprintResults", [])

                for r in results:
                    pos = r.get("position", "22")
                    if _is_number(pos):
                        pos_num = int(float(pos))
                    else:
                        pos_num = 22

                    constructor_id = r.get("Constructor", {}).get("constructorId", None)
                    status = r.get("status", None)

                    records.append({
                        "year": 2026,
                        "round": rnd,
                        "driverId": r["Driver"]["driverId"],
                        "constructorId": constructor_id,
                        "position": pos_num,
                        "points": float(r.get("points", 0)),
                        "status": status,
                        "event": "sprint",
                        "raceKey": _build_race_key(2026, rnd)
                    })
            except Exception as e:
                print(f"Error parseando {f}: {e}")

    return pd.DataFrame(records)

def get_2026_calendar():
    """Retorna las rondas 2026 desde races.csv o json"""
    races_path = os.path.join(RAW_DIR, "races.csv")
    if os.path.exists(races_path):
        df_rac = pd.read_csv(races_path)
        return df_rac[df_rac['year'] == 2026].copy()
    return pd.DataFrame()
