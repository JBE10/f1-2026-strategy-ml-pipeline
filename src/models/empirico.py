import numpy as np
import pandas as pd

class EmpiricalModel:
    def __init__(self, years_train=None, weight_mapping=None):
        self.years_train = years_train or [2023, 2024, 2025, 2026]
        # Ponderación manual: 2026 pesa muchísimo más que 2023 para reflejar el auto actual
        self.weight_mapping = weight_mapping or {2023: 1, 2024: 2, 2025: 4, 2026: 20}
        self.driver_profiles = {}
        self.fallback_profile = {}
        
    def _get_weight(self, year):
        return self.weight_mapping.get(int(year), 1)
        
    def fit(self, df_results):
        """
        Calcula perfiles empíricos de llegadas ponderados por año.
        """
        if df_results.empty:
            self.fallback_profile = {'positions': list(range(1, 23)), 'weights': [1/22]*22}
            return
            
        df_train = df_results[df_results['year'].isin(self.years_train)].copy()
        if df_train.empty:
            df_train = df_results.copy()
            
        for driver_id, group in df_train.groupby('driverId'):
            valid = group[pd.to_numeric(group['position'], errors='coerce').notna()]
            if not valid.empty:
                positions = pd.to_numeric(valid['position']).astype(int).tolist()
                years = valid['year'].tolist()
                
                weights = [self._get_weight(y) for y in years]
                sum_w = sum(weights)
                probs = [w/sum_w for w in weights]
                
                self.driver_profiles[driver_id] = {'positions': positions, 'weights': probs}
                
        valid_all = df_train[pd.to_numeric(df_train['position'], errors='coerce').notna()]
        if not valid_all.empty:
            all_positions = pd.to_numeric(valid_all['position']).astype(int).tolist()
            all_years = valid_all['year'].tolist()
            
            weights = [self._get_weight(y) for y in all_years]
            sum_w = sum(weights)
            probs = [w/sum_w for w in weights]
            
            self.fallback_profile = {'positions': all_positions, 'weights': probs}
        else:
            self.fallback_profile = {'positions': list(range(1, 23)), 'weights': [1/22]*22}
            
    def sample_race(self, active_drivers, event_type='race', **kwargs):
        """
        Simula una carrera ponderando los resultados recientes.
        """
        scores = {}
        for d in active_drivers:
            profile = self.driver_profiles.get(d, self.fallback_profile)
            if not profile or 'positions' not in profile:
                profile = self.fallback_profile
                
            sampled_pos = np.random.choice(profile['positions'], p=profile['weights'])
            ruido = np.random.uniform(0, 0.99)
            scores[d] = sampled_pos + ruido
            
        orden = sorted(scores.keys(), key=lambda k: scores[k])
        return orden
