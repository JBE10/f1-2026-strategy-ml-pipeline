import os
import sys
import pandas as pd
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.standardize import (
    get_historical_results,
    get_historical_sprints,
    get_2026_actual_results,
    get_2026_actual_sprints,
    get_2026_actual_standings
)
from src.models.ml_model import MLModel
from src.simulate import run_simulation, get_one_simulation_path
from config.puntos import SPRINT_ROUNDS_2026

def main():
    print("Iniciando pipeline de simulación...")
    
    # 1. Cargar datos
    print("Cargando datos históricos y actuales...")
    df_hist = get_historical_results()
    df_hist_sprint = get_historical_sprints()
    df_act = get_2026_actual_results()
    df_act_sprint = get_2026_actual_sprints()
    
    # Unir historial y 2026 para entrenar el modelo
    df_sources = [df_hist, df_hist_sprint, df_act, df_act_sprint]
    df_sources = [df for df in df_sources if df is not None and not df.empty]
    if df_sources:
        df_train = pd.concat(df_sources, ignore_index=True)
    else:
        print("No hay datos para entrenar el modelo.")
        return

    standings = get_2026_actual_standings()
    
    # 2. Determinar pilotos activos y rondas restantes
    if not standings.empty:
        active_drivers = standings['driverId'].tolist()
    else:
        # Fallback a los de 2026 en resultados
        if not df_act.empty:
            active_drivers = df_act['driverId'].unique().tolist()
        else:
            active_drivers = [
                'antonelli', 'norris', 'leclerc', 'piastri', 'russell', 
                'max_verstappen', 'hamilton', 'colapinto', 'sainz', 'albon', 
                'bearman', 'bortoleto', 'ocon', 'arvid_lindblad', 'stroll', 
                'alonso', 'perez', 'bottas', 'hulkenberg', 'gasly', 
                'lawson', 'hadjar'
            ]
            
    # Determinar última ronda corrida
    last_round = 0
    if not df_act.empty:
        last_round = df_act['round'].max()
        
    from src.standardize import get_2026_calendar
    from config.circuits import get_circuit_features
    df_cal = get_2026_calendar()
    
    rounds_left = []
    for r in range(last_round + 1, 23): # 22 carreras en 2026
        # Buscar circuitId en el calendario
        circuit_id = "unknown"
        if not df_cal.empty:
            row_cal = df_cal[df_cal['round'] == r]
            if not row_cal.empty:
                # En races.csv de Kaggle, circuitId es un número, pero nosotros mapeamos circuitRef.
                # Sin embargo, en standardize.py ya hicimos el merge.
                # Vamos a obtener el circuitRef si es posible.
                c_id = row_cal.iloc[0]['circuitId']
                # Necesitamos circuits.csv para mapear circuitId a circuitRef
                df_cir = pd.read_csv("data/raw/circuits.csv")
                circuit_ref = df_cir[df_cir['circuitId'] == c_id]['circuitRef'].iloc[0]
                circuit_id = circuit_ref
        
        c_feats = get_circuit_features(circuit_id)
        rounds_left.append({
            'round': r,
            'has_sprint': r in SPRINT_ROUNDS_2026,
            'is_street': c_feats['is_street'],
            'overtaking_difficulty': c_feats['overtaking_difficulty']
        })
        
    print(f"Pilotos activos: {len(active_drivers)}")
    print(f"Rondas restantes: {len(rounds_left)}")
    
    # 3. Entrenar modelo
    print("Entrenando modelo de Machine Learning (XGBoost)...")
    model = MLModel()
    model.fit(df_train)
    
    # 4. Simulación General
    B = 2000 # Iteraciones MC
    SEED = 42
    ENSEMBLE_SEEDS = [41, 42, 43, 44, 45]
    B_ENSEMBLE = 1000

    # Solo carreras principales (excluyendo sprints) para el historial de posiciones del desempate
    df_act_races = df_act[df_act['event'] == 'race'] if (not df_act.empty and 'event' in df_act.columns) else df_act

    print(f"Ejecutando Monte Carlo con B={B}...")
    df_metrics = run_simulation(standings, model, rounds_left, active_drivers, B=B, seed=SEED, actual_results=df_act_races)

    df_ensemble = pd.DataFrame()
    if ENSEMBLE_SEEDS and B_ENSEMBLE > 0:
        print(f"Ejecutando ensemble de semillas: {len(ENSEMBLE_SEEDS)} x B={B_ENSEMBLE}...")
        frames = []
        for seed in ENSEMBLE_SEEDS:
            df_seed = run_simulation(standings, model, rounds_left, active_drivers, B=B_ENSEMBLE, seed=seed, actual_results=df_act_races)
            df_seed['seed'] = seed
            frames.append(df_seed)
        if frames:
            df_all = pd.concat(frames, ignore_index=True)
            df_ensemble = (
                df_all
                .groupby('driverId', as_index=False)[['p_campeon', 'p_top3', 'expected_points']]
                .mean()
            )
    
    # 5. Generar 1 Universo de Ejemplo
    print("Generando un universo simulado paso a paso...")
    history, final_standings = get_one_simulation_path(standings, model, rounds_left, active_drivers, seed=44)
    sample_data = {
        'history': history,
        'final_standings': final_standings
    }
    
    # 6. Guardar resultados
    os.makedirs('data/output', exist_ok=True)
    df_metrics = df_metrics.sort_values('p_campeon', ascending=False)
    out_path = 'data/output/probabilidades.csv'
    df_metrics.to_csv(out_path, index=False)

    if not df_ensemble.empty:
        df_ensemble = df_ensemble.sort_values('p_campeon', ascending=False)
        df_ensemble.to_csv('data/output/probabilidades_ensemble.csv', index=False)
    
    with open('data/output/sample_path.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=4)

    model_config = {
        'model': 'MLModel_XGBoost',
        'years_train': model.years_train,
        'seed': SEED,
        'B': B,
        'ensemble_seeds': ENSEMBLE_SEEDS,
        'B_ensemble': B_ENSEMBLE,
        'ensemble_total_runs': B_ENSEMBLE * len(ENSEMBLE_SEEDS) if ENSEMBLE_SEEDS else 0
    }
    with open('data/output/model_config.json', 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=4)
        
    print(f"Resultados guardados en data/output/")
    print("\nTop 5 favoritos al título:")
    print(df_metrics.head(5))

if __name__ == "__main__":
    main()
