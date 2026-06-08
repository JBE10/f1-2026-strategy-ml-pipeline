import fastf1
import pandas as pd
import os
import logging

# Silenciar logs de requests y urllib3 para evitar tracebacks ruidosos en fechas futuras
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# Configurar caché (FastF1 descarga mucha data, es vital cachearla)
cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def calculate_race_pace(year, round_num):
    print(f"📡 Cargando datos de telemetría para {year} Ronda {round_num}...")
    original_year = year
    original_round = round_num
    
    # Si es una fecha futura, usamos directamente el fallback cacheado para evitar errores de red
    if year >= 2026:
        if round_num == 6:
            print(f"ℹ️  Nota: {year} es una fecha futura. Usando Mónaco 2024 como base de simulación (Dato cacheado).")
            year, round_num = 2024, 8
        else:
            print(f"ℹ️  Nota: {year} es una fecha futura. Usando Japón 2024 como base de simulación (Dato cacheado).")
            year, round_num = 2024, 4

    try:
        session = fastf1.get_session(year, round_num, 'R')
        # Intentamos cargar. Si falla la red pero hay cache, fastf1 suele seguir adelante con warnings.
        session.load(telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"⚠️ Error al cargar sesión: {e}")
        return None
        
    if not hasattr(session, 'laps') or len(session.laps) == 0:
        print("❌ Error: No se pudieron cargar vueltas para esta sesión.")
        return None

    laps = session.laps
    
    # FILTRO DE MACHINE LEARNING:
    # Para saber la velocidad "real" del coche, debemos limpiar el ruido:
    # 1. pick_quicklaps(): Elimina vueltas de entrada/salida a boxes (in/out laps).
    # 2. pick_track_status('1'): Solo vueltas bajo Bandera Verde (sin Safety Car ni Virtual Safety Car).
    print("🧹 Limpiando ruido (Safety Cars, Vueltas de entrada a Boxes)...")
    valid_laps = laps.pick_quicklaps().pick_track_status('1')
    
    paces = []
    for driver in valid_laps['Driver'].unique():
        driver_laps = valid_laps.pick_drivers(driver)
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
    out_path = os.path.join(base_dir, 'data', 'actual', f'race_pace_{original_year}_r{original_round}.csv')
    df_pace.to_csv(out_path, index=False)
    
    # NUEVO: Guardar TODAS las vueltas válidas para visualización detallada en el Dashboard
    laps_path = os.path.join(base_dir, 'data', 'actual', f'valid_laps_{original_year}_r{original_round}.csv')
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
    # Intentamos cargar Mónaco (Ronda 6 en el calendario 2026 planificado)
    calculate_race_pace(2026, 6)
