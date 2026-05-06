import fastf1
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.tire_model import TireDegradationModel

cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def extract_lap_data(year, round_num):
    print(f"📡 Descargando datos de vueltas y neumáticos para {year} R{round_num}...")
    try:
        session = fastf1.get_session(year, round_num, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
        
        # Filtramos vueltas limpias (sin safety car, sin entrada a boxes)
        valid_laps = laps.pick_quicklaps().pick_track_status('1')
        
        records = []
        for _, row in valid_laps.iterrows():
            compound = row['Compound']
            if pd.isna(compound) or compound not in ['SOFT', 'MEDIUM', 'HARD']:
                continue
                
            tyre_life = row['TyreLife']
            if pd.isna(tyre_life):
                continue
                
            lap_time = row['LapTime'].total_seconds()
            if pd.isna(lap_time):
                continue
                
            records.append({
                'Driver': row['Driver'],
                'Compound': compound,
                'TyreLife': float(tyre_life),
                'LapTime_s': float(lap_time)
            })
            
        return pd.DataFrame(records)
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

def main():
    # Bajamos un par de carreras representativas (ej. del 2024/2025) para que el modelo
    # aprenda cómo mueren los neumáticos Pirelli.
    df_all = pd.DataFrame()
    
    # Usaremos las primeras 5 carreras del 2024 para este entrenamiento base
    for rnd in [1, 2, 3, 4, 5]:
        df = extract_lap_data(2024, rnd)
        if not df.empty:
            df_all = pd.concat([df_all, df], ignore_index=True)
            
    if df_all.empty:
        print("No se encontraron datos.")
        return
        
    # Filtrar outliers (tiempos muy altos por errores que no fueron registrados como bandera amarilla)
    q1 = df_all['LapTime_s'].quantile(0.05)
    q3 = df_all['LapTime_s'].quantile(0.95)
    df_filtered = df_all[(df_all['LapTime_s'] >= q1) & (df_all['LapTime_s'] <= q3)]
    
    print(f"\n📊 Datos extraídos: {len(df_filtered)} vueltas limpias.")
    
    model = TireDegradationModel()
    model.fit(df_filtered)
    
    # Hacemos una prueba rápida de predicción para mostrar en consola
    print("\n🏎️ Predicción de degradación simulada (Auto con Ritmo Base de 90.0s):")
    for comp in ['SOFT', 'HARD']:
        print(f"--- Compuesto {comp} ---")
        for lap in [1, 10, 20]:
            pred = model.predict_lap_time(comp, lap, base_pace=90.0)
            print(f"Vuelta {lap} del neumático: {pred:.2f} segundos")
    
    os.makedirs('data/output', exist_ok=True)
    model.save('data/output/tire_model.pkl')
    print("\n✅ Modelo de neumáticos entrenado y guardado en data/output/tire_model.pkl")

if __name__ == "__main__":
    main()
