import fastf1
import pandas as pd
import os

cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

ACTUAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'actual')
os.makedirs(ACTUAL_DIR, exist_ok=True)

def fetch_pace_for_race(year, round_num):
    out_path = os.path.join(ACTUAL_DIR, f'race_pace_{year}_r{round_num}.csv')
    if os.path.exists(out_path):
        print(f"✅ {year} R{round_num} ya existe, omitiendo...")
        return
        
    print(f"📡 Descargando {year} R{round_num}...")
    try:
        session = fastf1.get_session(year, round_num, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        
        if len(session.laps) == 0:
            print(f"⚠️ No hay datos de vueltas para {year} R{round_num}.")
            return
            
        valid_laps = session.laps.pick_quicklaps().pick_track_status('1')
        
        paces = []
        for driver in valid_laps['Driver'].unique():
            driver_laps = valid_laps.pick_driver(driver)
            if len(driver_laps) > 5:
                median_time = driver_laps['LapTime'].median()
                paces.append({
                    'Driver': driver,
                    'Median_Pace_s': median_time.total_seconds(),
                    'Valid_Laps': len(driver_laps)
                })
                
        if not paces:
            print(f"⚠️ No hay suficientes vueltas válidas para {year} R{round_num}.")
            return
            
        df_pace = pd.DataFrame(paces).sort_values('Median_Pace_s')
        df_pace['Pace_Rank'] = df_pace['Median_Pace_s'].rank(method='min')
        
        df_pace.to_csv(out_path, index=False)
        print(f"💾 Guardado: {out_path}")
    except Exception as e:
        print(f"❌ Error en {year} R{round_num}: {e}")

def main():
    # Descargaremos el histórico del 2025 para entrenar al modelo
    # Limitamos a las últimas 10 carreras para no hacer el proceso demasiado largo
    print("Iniciando backfilling de telemetría (2025)...")
    year = 2025
    schedule = fastf1.get_event_schedule(year)
    
    # Filtramos eventos de carrera y tomamos las últimas 10 rondas
    race_events = schedule[schedule['EventFormat'] != 'testing']
    if len(race_events) > 10:
        race_events = race_events.tail(10)
        
    for _, event in race_events.iterrows():
        rnd = event['RoundNumber']
        if rnd > 0:
            fetch_pace_for_race(year, rnd)
            
if __name__ == "__main__":
    main()
