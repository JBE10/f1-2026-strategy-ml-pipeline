import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.puntos import asignar_puntos, sort_key_desempate

def run_simulation(current_standings, model, rounds_left, active_drivers, B=10000, seed=42, actual_results=None):
    """
    Ejecuta el Monte Carlo y retorna un dict con las probabilidades de ser campeón.
    """
    np.random.seed(seed)
    
    base_points = {}
    base_positions = {}
    for d in active_drivers:
        base_points[d] = 0
        base_positions[d] = []
        
    if current_standings is not None and not current_standings.empty:
        for _, row in current_standings.iterrows():
            d = row['driverId']
            if d in active_drivers:
                base_points[d] = row.get('points', 0)
                
    # Cargar todas las posiciones obtenidas en carreras reales ya disputadas
    if actual_results is not None and not actual_results.empty:
        for _, row in actual_results.iterrows():
            d = row['driverId']
            if d in active_drivers:
                pos = pd.to_numeric(row.get('position'), errors='coerce')
                if not pd.isna(pos):
                    base_positions[d].append(int(pos))
                    
    results = []
    driver_metrics = {}
    
    for b in range(B):
        sim_points = base_points.copy()
        sim_positions = {k: v.copy() for k, v in base_positions.items()}
        
        for rnd in rounds_left:
            is_st = rnd.get('is_street', 0)
            diff = rnd.get('overtaking_difficulty', 2)
            
            orden_carrera = model.sample_race(active_drivers, event_type='race', is_street=is_st, overtaking_difficulty=diff)
            pts_carrera = asignar_puntos(orden_carrera, tipo='race')
            
            for i, d in enumerate(orden_carrera, start=1):
                sim_points[d] += pts_carrera.get(d, 0)
                sim_positions[d].append(i)
                
            if rnd.get('has_sprint', False):
                orden_sprint = model.sample_race(active_drivers, event_type='sprint', is_street=is_st, overtaking_difficulty=diff)
                pts_sprint = asignar_puntos(orden_sprint, tipo='sprint')
                for d in orden_sprint:
                    sim_points[d] += pts_sprint.get(d, 0)
                    
        stats_list = []
        for d in active_drivers:
            stats = {
                'driverId': d,
                'points': sim_points[d],
                'positions': sim_positions[d]
            }
            stats_list.append(stats)
            
        stats_list.sort(key=sort_key_desempate, reverse=True)
        
        # Registrar resultados de esta iteración
        campeon = stats_list[0]['driverId']
        results.append(campeon)
        for i, d_stat in enumerate(stats_list):
            d = d_stat['driverId']
            if d not in driver_metrics:
                driver_metrics[d] = {'top3': 0, 'pts': []}
            if i < 3:
                driver_metrics[d]['top3'] += 1
            driver_metrics[d]['pts'].append(d_stat['points'])
            
    # Calcular probabilidades y esperanzas
    metrics_list = []
    for d in active_drivers:
        p_campeon = results.count(d) / B
        p_top3 = driver_metrics[d]['top3'] / B if d in driver_metrics else 0
        mean_pts = np.mean(driver_metrics[d]['pts']) if d in driver_metrics else 0
        metrics_list.append({
            'driverId': d,
            'p_campeon': p_campeon,
            'p_top3': p_top3,
            'expected_points': round(mean_pts, 1)
        })
        
    return pd.DataFrame(metrics_list)

def get_one_simulation_path(current_standings, model, rounds_left, active_drivers, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    base_points = {}
    for d in active_drivers:
        base_points[d] = 0
        
    if current_standings is not None and not current_standings.empty:
        for _, row in current_standings.iterrows():
            d = row['driverId']
            if d in active_drivers:
                base_points[d] = row.get('points', 0)
                
    sim_points = base_points.copy()
    history = []
    
    for rnd in rounds_left:
        step_data = {'round': rnd['round'], 'sprint_winner': None, 'podium': []}
        is_st = rnd.get('is_street', 0)
        diff = rnd.get('overtaking_difficulty', 2)
        
        if rnd.get('has_sprint', False):
            orden_sprint = model.sample_race(active_drivers, event_type='sprint', is_street=is_st, overtaking_difficulty=diff)
            pts_sprint = asignar_puntos(orden_sprint, tipo='sprint')
            for d in orden_sprint:
                sim_points[d] += pts_sprint.get(d, 0)
            step_data['sprint_winner'] = orden_sprint[0]
            
        orden_carrera = model.sample_race(active_drivers, event_type='race', is_street=is_st, overtaking_difficulty=diff)
        pts_carrera = asignar_puntos(orden_carrera, tipo='race')
        for d in orden_carrera:
            sim_points[d] += pts_carrera.get(d, 0)
            
        step_data['podium'] = orden_carrera[:3]
        history.append(step_data)
        
    final_standings = sorted(sim_points.items(), key=lambda x: x[1], reverse=True)
    return history, final_standings
