import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.standardize import get_historical_results
from src.models.ml_model import MLModel

def run_backtest_2025(df_all):
    print("\n" + "="*60)
    print("🏎️  BACKTEST MASIVO: TEMPORADA COMPLETA 2025")
    print("="*60)
    
    # Todas las carreras de 2025 en orden
    df_2025 = df_all[df_all['year'] == 2025].copy()
    rondas_2025 = sorted(df_2025['round'].unique())
    
    maes_por_ronda = []
    
    for r in rondas_2025:
        # El modelo SOLO puede ver el pasado (Data Leakage Prevention)
        # Entrenamos con 2023, 2024 y las carreras de 2025 ANTERIORES a la ronda actual
        df_train = df_all[
            (df_all['year'] < 2025) | 
            ((df_all['year'] == 2025) & (df_all['round'] < r))
        ].copy()
        
        df_truth = df_2025[df_2025['round'] == r].copy()
        
        if df_truth.empty:
            continue
            
        print(f"[{r:02d}/{len(rondas_2025)}] Entrenando y simulando Ronda {r}...", end=" ")
        
        # Instanciar y entrenar
        model = MLModel(years_train=[2023, 2024, 2025])
        
        # Suprimimos los prints internos del modelo para no inundar la consola
        import contextlib
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                model.fit(df_train)
        
        active_drivers = df_truth['driverId'].tolist()
        
        # Extraer la Posición de Parrilla (Grid) Real de cada piloto para esta carrera
        starting_grids = {}
        for _, row in df_truth.iterrows():
            d_id = row['driverId']
            g_val = pd.to_numeric(row.get('grid'), errors='coerce')
            if pd.isna(g_val) or g_val == 0:
                g_val = 20.0
            starting_grids[d_id] = float(g_val)
            
        B = 200 # Reducimos iteraciones para que el backtest corra rápido
        resultados_sim = {d: [] for d in active_drivers}
        
        for _ in range(B):
            orden = model.sample_race(active_drivers, event_type='race', starting_grids=starting_grids)
            for pos_0_idx, d in enumerate(orden):
                resultados_sim[d].append(pos_0_idx + 1)
                
        # Calcular MAE de esta ronda
        comparacion = []
        for d in active_drivers:
            pos_real = df_truth[df_truth['driverId'] == d]['position'].values[0]
            pos_sim = np.mean(resultados_sim[d])
            comparacion.append(abs(pos_real - pos_sim))
            
        mae_ronda = np.mean(comparacion)
        print(f"✅ MAE: {mae_ronda:.2f}")
        maes_por_ronda.append((r, mae_ronda))
        
    print("\n" + "*"*60)
    print("📊 RESULTADOS FINALES DE CONVERGENCIA 2025")
    print("*"*60)
    
    mitad_1 = [m for r, m in maes_por_ronda if r <= len(rondas_2025)/2]
    mitad_2 = [m for r, m in maes_por_ronda if r > len(rondas_2025)/2]
    
    print(f"📉 MAE Primera Mitad del Año : {np.mean(mitad_1):.2f} posiciones de error")
    print(f"📉 MAE Segunda Mitad del Año : {np.mean(mitad_2):.2f} posiciones de error")
    print(f"🏁 MAE Promedio Global 2025  : {np.mean([m for _, m in maes_por_ronda]):.2f} posiciones de error")

def main():
    df_hist = get_historical_results()
    if df_hist.empty:
        print("No se encontró el dataset histórico.")
        return
        
    run_backtest_2025(df_hist)

if __name__ == '__main__':
    main()
