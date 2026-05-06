import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import os

DRIVER_CODE_MAP = {
    'ANT': 'antonelli', 'NOR': 'norris', 'LEC': 'leclerc', 'PIA': 'piastri',
    'RUS': 'russell', 'VER': 'max_verstappen', 'HAM': 'hamilton', 'COL': 'colapinto',
    'SAI': 'sainz', 'ALB': 'albon', 'BEA': 'bearman', 'BOR': 'bortoleto',
    'OCO': 'ocon', 'LIN': 'arvid_lindblad', 'STR': 'stroll', 'ALO': 'alonso',
    'PER': 'perez', 'BOT': 'bottas', 'HUL': 'hulkenberg', 'GAS': 'gasly',
    'LAW': 'lawson', 'HAD': 'hadjar'
}

class MLModel:
    def __init__(self, years_train=None, window=10, dnf_noise=2.0, default_dnf=0.08):
        self.years_train = years_train or [2023, 2024, 2025, 2026]
        self.window = window
        self.model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
        
        # Guardaremos el último estado conocido para la inferencia
        self.current_driver_features = {}
        self.current_team_features = {}
        self.active_driver_team = {}
        
        # Estadísticas de DNF
        self.driver_dnf = {}
        self.team_dnf = {}
        self.default_dnf = default_dnf
        self.dnf_noise = dnf_noise
        
        # Telemetría de FastF1
        self.real_pace_rank = {}
        self._load_telemetry()

    def _load_telemetry(self):
        pace_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'actual', 'race_pace_2026_r4.csv')
        if os.path.exists(pace_path):
            print("🚀 Inyectando Telemetría de FastF1 (Ritmo de Carrera) en el modelo...")
            df_pace = pd.read_csv(pace_path)
            df_pace['Pace_Rank'] = df_pace['Median_Pace_s'].rank(method='min')
            for _, row in df_pace.iterrows():
                code = row['Driver']
                d_id = DRIVER_CODE_MAP.get(code)
                if d_id:
                    self.real_pace_rank[d_id] = row['Pace_Rank']

    def _engineer_features(self, df):
        """
        Crea las features (variables) para el modelo de Machine Learning.
        """
        print("🔧 Armando variables (Feature Engineering) para el modelo de Machine Learning...")
        
        df = df.sort_values(['year', 'round']).reset_index(drop=True)
        features_list = []
        
        driver_history_pos = {}
        driver_history_grid = {}
        team_history_pos = {}
        
        for idx, row in df.iterrows():
            d_id = row['driverId']
            t_id = row.get('constructorId', None)
            
            d_hist_pos = driver_history_pos.get(d_id, [])
            d_hist_grid = driver_history_grid.get(d_id, [])
            t_hist_pos = team_history_pos.get(t_id, []) if t_id else []
            
            if len(d_hist_pos) > 0:
                d_recent_pos = d_hist_pos[-self.window:]
                d_mean_pos = np.mean(d_recent_pos)
                d_best_pos = np.min(d_recent_pos)
            else:
                d_mean_pos = 10.0
                d_best_pos = 10.0
                
            if len(d_hist_grid) > 0:
                d_recent_grid = d_hist_grid[-self.window:]
                d_mean_grid = np.mean(d_recent_grid)
            else:
                d_mean_grid = 10.0
                
            if len(t_hist_pos) > 0:
                t_recent_pos = t_hist_pos[-self.window:]
                t_mean_pos = np.mean(t_recent_pos)
            else:
                t_mean_pos = 10.0
                
            # Posición de parrilla de la carrera ACTUAL
            current_grid = pd.to_numeric(row.get('grid'), errors='coerce')
            if pd.isna(current_grid) or current_grid == 0:
                current_grid = 20.0
            else:
                current_grid = float(current_grid)
                
            features_list.append({
                'd_mean_pos': d_mean_pos,
                'd_best_pos': d_best_pos,
                'd_mean_grid': d_mean_grid, # PROXY de VELOCIDAD PURA
                't_mean_pos': t_mean_pos,
                'Starting_Grid': current_grid
            })
            
            pos = pd.to_numeric(row.get('position'), errors='coerce')
            pos_val = 20 if pd.isna(pos) else float(pos)
            
            # PACE PROXY: intentamos usar la telemetría real, si no, fallback a la grid
            race_key = f"{int(row['year'])}_{int(row['round'])}"
            if hasattr(self, 'historical_pace_rank') and race_key in self.historical_pace_rank and d_id in self.historical_pace_rank[race_key]:
                pace_val = float(self.historical_pace_rank[race_key][d_id])
            else:
                grid_val = pd.to_numeric(row.get('grid'), errors='coerce')
                if pd.isna(grid_val) or grid_val == 0:
                    pace_val = 20.0
                else:
                    pace_val = float(grid_val)
            
            if d_id not in driver_history_pos: driver_history_pos[d_id] = []
            driver_history_pos[d_id].append(pos_val)
            
            if d_id not in driver_history_grid: driver_history_grid[d_id] = []
            driver_history_grid[d_id].append(pace_val)
            
            if t_id:
                if t_id not in team_history_pos: team_history_pos[t_id] = []
                team_history_pos[t_id].append(pos_val)

        df_feats = pd.DataFrame(features_list)
        df_out = pd.concat([df.reset_index(drop=True), df_feats], axis=1)
        
        self.current_driver_features = {}
        for d_id, hist in driver_history_pos.items():
            recent = hist[-self.window:]
            grid_recent = driver_history_grid[d_id][-self.window:]
            self.current_driver_features[d_id] = {
                'd_mean_pos': np.mean(recent),
                'd_best_pos': np.min(recent),
                'd_mean_grid': np.mean(grid_recent)
            }
            
        self.current_team_features = {}
        for t_id, hist in team_history_pos.items():
            recent = hist[-self.window:]
            self.current_team_features[t_id] = {
                't_mean_pos': np.mean(recent)
            }
            
        return df_out


    def fit(self, df_results):
        if df_results.empty:
            return

        df = df_results.copy()
        
        # Determinar equipo activo
        self.active_driver_team = {}
        if 'constructorId' in df.columns:
            df_recent = df.sort_values(['year', 'round'])
            last_team = df_recent.groupby('driverId').tail(1)
            for _, row in last_team.iterrows():
                self.active_driver_team[row['driverId']] = row.get('constructorId', None)
                
        # Estimar DNF probabilístico
        self.driver_dnf = {}
        self.team_dnf = {}
        if 'status' in df.columns:
            df_event = df[df['event'] == 'race'] if 'event' in df.columns else df
            
            for driver_id, group in df_event.groupby('driverId'):
                statuses = group['status'].fillna('').astype(str).str.lower()
                if statuses.empty: continue
                finished_mask = statuses.str.contains('finished') | statuses.str.contains('lapped')
                dnf_rate = 1.0 - (finished_mask.sum() / len(statuses))
                self.driver_dnf[driver_id] = max(0.0, min(0.6, dnf_rate))
                
            if 'constructorId' in df_event.columns:
                for team_id, group in df_event.groupby('constructorId'):
                    statuses = group['status'].fillna('').astype(str).str.lower()
                    if statuses.empty: continue
                    finished_mask = statuses.str.contains('finished') | statuses.str.contains('lapped')
                    dnf_rate = 1.0 - (finished_mask.sum() / len(statuses))
                    self.team_dnf[team_id] = max(0.0, min(0.6, dnf_rate))

        # Filtrar años de entrenamiento
        df_train = df[df['year'].isin(self.years_train)].copy()
        if df_train.empty:
            df_train = df.copy()

        # Armar Variables
        df_engineered = self._engineer_features(df_train)
        
        # Limpiar Target (Variable a predecir)
        # Solo entrenamos el modelo sobre autos que terminaron la carrera, el DNF se simula aparte
        df_ml = df_engineered.copy()
        df_ml['position'] = pd.to_numeric(df_ml['position'], errors='coerce')
        df_ml = df_ml.dropna(subset=['position'])
        
        # Features X e Y (Target)
        X = df_ml[['d_mean_pos', 'd_best_pos', 'd_mean_grid', 't_mean_pos', 'Starting_Grid']]
        y = df_ml['position']
        
        print(f"🤖 Entrenando XGBoost con {len(X)} registros históricos...")
        self.model.fit(X, y)
        print("✅ Entrenamiento completado.")


    def _get_dnf_prob(self, driver_id, team_id=None):
        driver_rate = self.driver_dnf.get(driver_id, self.default_dnf)
        team_rate = self.team_dnf.get(team_id, self.default_dnf)
        return 0.5 * driver_rate + 0.5 * team_rate

    def sample_race(self, active_drivers, event_type='race', starting_grids=None):
        """
        Inferencia del modelo ML + Variabilidad estocástica
        """
        if starting_grids is None:
            starting_grids = {}
            
        scores = {}
        
        # 1. Armar matriz de inferencia (X) para los pilotos activos en su estado actual
        infer_data = []
        for d in active_drivers:
            t_id = self.active_driver_team.get(d)
            
            d_feats = self.current_driver_features.get(d, {'d_mean_pos': 10.0, 'd_best_pos': 10.0, 'd_mean_grid': 10.0})
            t_feats = self.current_team_features.get(t_id, {'t_mean_pos': 10.0}) if t_id else {'t_mean_pos': 10.0}
            
            # 🚀 Inyectar Ritmo de Carrera Real de FastF1: 
            # Reemplaza o promedia la posición de parrilla con la velocidad pura demostrada en pista
            mean_grid = d_feats['d_mean_grid']
            if d in self.real_pace_rank:
                mean_grid = (mean_grid * 0.3) + (self.real_pace_rank[d] * 0.7) # 70% peso a la telemetría actual
            
            actual_grid = starting_grids.get(d, mean_grid)
            
            infer_data.append([
                d_feats['d_mean_pos'],
                d_feats['d_best_pos'],
                mean_grid,
                t_feats['t_mean_pos'],
                actual_grid
            ])
            
        X_infer = pd.DataFrame(infer_data, columns=['d_mean_pos', 'd_best_pos', 'd_mean_grid', 't_mean_pos', 'Starting_Grid'])
        
        # 2. El modelo predice la Posición Esperada Pura
        expected_positions = self.model.predict(X_infer)
        
        # 3. Monte Carlo: Añadir variabilidad para simular una carrera y no ser determinista
        for i, d in enumerate(active_drivers):
            pred_pos = expected_positions[i]
            
            # El ruido modela imprevistos: paradas de box lentas, peleas en pista, desgaste.
            # Los pilotos de punta (pred_pos bajo) suelen tener menos varianza que los del medio del pelotón
            variance = 1.0 + (pred_pos * 0.1) 
            final_pos = pred_pos + np.random.normal(0, variance)
            
            # Chequear DNF (Abandono)
            team_id = self.active_driver_team.get(d)
            dnf_prob = self._get_dnf_prob(d, team_id)
            if event_type == 'sprint':
                dnf_prob *= 0.6
                
            if np.random.rand() < dnf_prob:
                final_pos += (25.0 + np.random.normal(0, self.dnf_noise)) # Lo mandamos al final
                
            scores[d] = final_pos
            
        # Ordenar (Menor score significa posición 1, 2, 3...)
        orden = sorted(scores.keys(), key=lambda k: scores[k], reverse=False)
        return orden
