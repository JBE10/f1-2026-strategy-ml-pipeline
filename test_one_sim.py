import sys
import pandas as pd
import numpy as np

from src.standardize import (
    get_historical_results,
    get_historical_sprints,
    get_2026_actual_results,
    get_2026_actual_sprints,
    get_2026_actual_standings
)
from src.models.factorial import FactorialModel
from config.puntos import asignar_puntos, SPRINT_ROUNDS_2026

df_hist = get_historical_results()
df_hist_sprint = get_historical_sprints()
df_act = get_2026_actual_results()
df_act_sprint = get_2026_actual_sprints()

df_sources = [df_hist, df_hist_sprint, df_act, df_act_sprint]
df_sources = [df for df in df_sources if df is not None and not df.empty]
df_train = pd.concat(df_sources, ignore_index=True) if df_sources else pd.DataFrame()

standings = get_2026_actual_standings()
active_drivers = standings['driverId'].tolist()

model = FactorialModel()
model.fit(df_train)

base_points = {}
for _, row in standings.iterrows():
    if row['driverId'] in active_drivers:
        base_points[row['driverId']] = row['points']

last_round = df_act['round'].max()
rounds_left = [{'round': r, 'has_sprint': r in SPRINT_ROUNDS_2026} for r in range(last_round + 1, 23)]

np.random.seed(44)
sim_points = base_points.copy()

print("🏁 ESTADO ACTUAL TRAS RONDA", last_round)
top5_actual = sorted(sim_points.keys(), key=lambda k: sim_points[k], reverse=True)[:5]
for i, d in enumerate(top5_actual, 1):
    print(f"   {i}º {d.upper().replace('_', ' ')}: {sim_points[d]} pts")
print("-" * 50)

for rnd in rounds_left:
    print(f"▶ Ronda {rnd['round']}", end="")
    if rnd['has_sprint']:
        orden_sprint = model.sample_race(active_drivers, event_type='sprint')
        pts_sprint = asignar_puntos(orden_sprint, 'sprint')
        for d in orden_sprint: sim_points[d] += pts_sprint.get(d, 0)
        print(f" (+Sprint. Ganador: {orden_sprint[0].upper().replace('_', ' ')})")
    else:
        print()
        
    orden_carrera = model.sample_race(active_drivers, event_type='race')
    pts_carrera = asignar_puntos(orden_carrera, 'race')
    for d in orden_carrera: sim_points[d] += pts_carrera.get(d, 0)
    
    print(f"   Podio Carrera: 1º {orden_carrera[0].upper().replace('_', ' ')} | 2º {orden_carrera[1].upper().replace('_', ' ')} | 3º {orden_carrera[2].upper().replace('_', ' ')}")

print("-" * 50)
print("🏆 POSICIONES FINALES DEL CAMPEONATO (Universo #44)")
top5_final = sorted(sim_points.keys(), key=lambda k: sim_points[k], reverse=True)[:5]
for i, d in enumerate(top5_final, 1):
    print(f"   {i}º {d.upper().replace('_', ' ')}: {sim_points[d]} pts")
