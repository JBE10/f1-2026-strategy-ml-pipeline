import fastf1
import pandas as pd
import os

# Configurar caché (FastF1 descarga mucha data, es vital cachearla)
cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def calculate_race_pace(year, round_num):
    print(f"📡 Intentando conectar con los servidores de F1 Live Timing para {year} Ronda {round_num}...")
    try:
        # 'R' significa Carrera (Race). Podríamos usar 'Q' para Qualifying.
        session = fastf1.get_session(year, round_num, 'R')
        # Cargamos datos de tiempos por vuelta. Desactivamos telemetría pesada (acelerador/freno) por ahora.
        session.load(telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"⚠️ No se pudieron obtener datos para {year}. Puede que la carrera no exista en la API real o haya un error de conexión.")
        print("🔙 Fallback: Cargando datos históricos de Miami 2024 (Ronda 6) para demostración de Machine Learning...")
        year, round_num = 2024, 6
        session = fastf1.get_session(year, round_num, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        
    laps = session.laps
    
    # FILTRO DE MACHINE LEARNING:
    # Para saber la velocidad "real" del coche, debemos limpiar el ruido:
    # 1. pick_quicklaps(): Elimina vueltas de entrada/salida a boxes (in/out laps).
    # 2. pick_track_status('1'): Solo vueltas bajo Bandera Verde (sin Safety Car ni Virtual Safety Car).
    print("🧹 Limpiando ruido (Safety Cars, Vueltas de entrada a Boxes)...")
    valid_laps = laps.pick_quicklaps().pick_track_status('1')
    
    paces = []
    for driver in valid_laps['Driver'].unique():
        driver_laps = valid_laps.pick_driver(driver)
        if len(driver_laps) > 5: # Exigimos al menos 5 vueltas representativas
            mean_time = driver_laps['LapTime'].mean()
            median_time = driver_laps['LapTime'].median()
            paces.append({
                'Driver': driver,
                'Median_Pace_s': median_time.total_seconds(),
                'Valid_Laps': len(driver_laps)
            })
            
    # Convertir a DataFrame y ordenar del más rápido al más lento
    df_pace = pd.DataFrame(paces).sort_values('Median_Pace_s')
    
    # Calcular el "Delta" (diferencia en segundos por vuelta respecto al más rápido)
    fastest_pace = df_pace['Median_Pace_s'].min()
    df_pace['Delta_to_Leader_s'] = (df_pace['Median_Pace_s'] - fastest_pace).round(3)
    
    print("\n🏎️  RITMO DE CARRERA PURO (Mediana de vueltas válidas):")
    print(df_pace[['Driver', 'Median_Pace_s', 'Delta_to_Leader_s', 'Valid_Laps']].to_string(index=False))
    
    # Guardar este feature para el modelo de ML (Agregado)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base_dir, 'data', 'actual', f'race_pace_{year}_r{round_num}.csv')
    df_pace.to_csv(out_path, index=False)
    
    # NUEVO: Guardar TODAS las vueltas válidas para visualización detallada en el Dashboard
    laps_path = os.path.join(base_dir, 'data', 'actual', f'valid_laps_{year}_r{round_num}.csv')
    # Seleccionamos columnas útiles para ahorrar espacio
    df_laps_export = valid_laps[['Driver', 'LapNumber', 'LapTime', 'Stint', 'Compound', 'TyreLife']].copy()
    # Convertir LapTime a segundos (float)
    df_laps_export['LapTime_s'] = df_laps_export['LapTime'].dt.total_seconds()
    df_laps_export.drop(columns=['LapTime'], inplace=True)
    df_laps_export.to_csv(laps_path, index=False)
    
    print(f"\n💾 Agregados guardados en: {out_path}")
    print(f"💾 Vueltas detalladas guardadas en: {laps_path}")
    
    return df_pace

if __name__ == "__main__":
    # Intentamos cargar Miami (Ronda 4 en el calendario 2026 planificado)
    calculate_race_pace(2026, 4)
