import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.standardize import get_historical_results, get_2026_actual_results
from src.models.ml_model import MLModel

def run_backtest_for_round(target_round, df_hist, df_act):
    print(f"\n{'='*50}")
    print(f"🏁 Iniciando Backtest: GP 2026 - Ronda {target_round}")
    print(f"{'='*50}")
    
    # Ocultar Ronda N y posteriores del set de entrenamiento
    df_train_act = df_act[df_act['round'] < target_round].copy()
    df_truth = df_act[df_act['round'] == target_round].copy()
    
    if df_truth.empty:
        print(f"No se encontraron resultados reales de la Ronda {target_round} para comparar.")
        return None
        
    df_train = pd.concat([df_hist, df_train_act], ignore_index=True)
    
    # Entrenar el modelo
    print(f"🤖 Entrenando XGBoost con historial hasta Ronda {target_round - 1}...")
    model = MLModel()
    model.fit(df_train)
    
    # Obtener predicciones
    active_drivers = df_truth['driverId'].tolist()
    
    B = 1000
    resultados_sim = {d: [] for d in active_drivers}
    
    for _ in range(B):
        orden = model.sample_race(active_drivers, event_type='race')
        for pos_0_idx, d in enumerate(orden):
            resultados_sim[d].append(pos_0_idx + 1)
            
    # Comparar
    comparacion = []
    for d in active_drivers:
        pos_real = df_truth[df_truth['driverId'] == d]['position'].values[0]
        pos_sim = np.mean(resultados_sim[d])
        comparacion.append({
            'Piloto': d.upper().replace('_', ' '),
            'Pos Real': int(pos_real),
            'Pred ML': round(pos_sim, 1),
            'Error': abs(pos_real - pos_sim)
        })
        
    df_comp = pd.DataFrame(comparacion).sort_values('Pos Real')
    
    mae = df_comp['Error'].mean()
    print(f"\n📊 RESULTADOS RONDA {target_round}:")
    print(df_comp.to_string(index=False))
    print(f"\n📉 Error Medio Absoluto (MAE) Ronda {target_round}: {mae:.2f} posiciones.")
    return mae

def main():
    df_hist = get_historical_results()
    df_act = get_2026_actual_results()
    
    rondas = sorted(df_act['round'].unique())
    
    maes = []
    for r in rondas:
        mae = run_backtest_for_round(r, df_hist, df_act)
        if mae is not None:
            maes.append((r, mae))
            
    print("\n" + "*"*50)
    print("📈 RESUMEN DEL BACKTEST TEMPORADA 2026")
    print("*"*50)
    for r, mae in maes:
        print(f"Ronda {r}: MAE = {mae:.2f} posiciones")
        
    if maes:
        print(f"\n--> MAE Promedio Global 2026: {np.mean([m for _, m in maes]):.2f} posiciones")

if __name__ == '__main__':
    main()
