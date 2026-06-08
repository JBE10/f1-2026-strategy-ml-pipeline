import numpy as np
import pandas as pd
from trueskill import TrueSkill, Rating, rate

class TrueSkillModel:
    def __init__(
        self,
        years_train=None,
        mu0=25.0,
        sigma0=8.333,
        beta=4.167,
        tau=0.083,
        draw_probability=0.0,
        dnf_noise=2.0,
        default_dnf=0.08
    ):
        """
        Modelo Bayesiano de habilidad usando TrueSkill.
        
        Args:
            years_train: Años a incluir en el entrenamiento.
            mu0: Habilidad inicial media.
            sigma0: Incertidumbre inicial.
            beta: Varianza de la "performance" (impacto del azar en un día de carrera).
            tau: Ruido dinámico (cuánto cambia la habilidad real de una carrera a otra).
            draw_probability: Probabilidad de empate (casi 0 en F1, salvo DNF simultáneos).
        """
        self.years_train = years_train or [2023, 2024, 2025, 2026]
        self.env = TrueSkill(mu=mu0, sigma=sigma0, beta=beta, tau=tau, draw_probability=draw_probability)
        self.ratings = {}
        self.driver_dnf = {}
        self.team_dnf = {}
        self.active_driver_team = {}
        self.default_dnf = default_dnf
        self.dnf_noise = dnf_noise

    def fit(self, df_results):
        if df_results.empty:
            return

        df = df_results.copy()
        df = df[df['year'].isin(self.years_train)]
        if df.empty:
            df = df_results.copy()
            
        # Determinar equipo activo
        self.active_driver_team = {}
        if 'constructorId' in df.columns:
            df_recent = df.sort_values(['year', 'round'])
            last_team = df_recent.groupby('driverId').tail(1)
            for _, row in last_team.iterrows():
                self.active_driver_team[row['driverId']] = row.get('constructorId', None)
                
        # Estimar probabilidad empírica de DNF
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

        # Reconstruir historial cronológicamente
        if 'raceKey' not in df.columns:
            df['raceKey'] = df.apply(lambda x: f"{int(x['year'])}-{int(x['round']):02d}", axis=1)
            
        df = df.sort_values(['year', 'round'])
        
        # Iterar por cada carrera
        for race_key, race_df in df.groupby(['year', 'round'], sort=False):
            # Ignorar sprints para la actualización TrueSkill o ponderarlos menos
            if 'event' in race_df.columns and (race_df['event'] == 'sprint').all():
                continue
                
            drivers_in_race = race_df['driverId'].tolist()
            
            # Obtener posiciones (o max pos para DNF)
            max_pos = len(drivers_in_race)
            positions = []
            for _, row in race_df.iterrows():
                pos = pd.to_numeric(row.get('position'), errors='coerce')
                if pd.isna(pos):
                    positions.append(max_pos) # DNF
                else:
                    positions.append(int(pos))
                    
            # Crear grupos de rating
            rating_groups = []
            for d in drivers_in_race:
                if d not in self.ratings:
                    self.ratings[d] = self.env.create_rating()
                rating_groups.append((self.ratings[d],))
                
            # TrueSkill requiere que ranks comience en 0 para el 1er lugar (ranks más bajos son mejores)
            # Y usa "ranks" para resolver empates (mismo rank = empate)
            ranks = [p - 1 for p in positions]
            
            try:
                updated_ratings = self.env.rate(rating_groups, ranks=ranks)
                # Guardar ratings actualizados
                for i, d in enumerate(drivers_in_race):
                    self.ratings[d] = updated_ratings[i][0]
            except Exception as e:
                print(f"Error actualizando TrueSkill en {race_key}: {e}")

    def _get_dnf_prob(self, driver_id, team_id=None):
        driver_rate = self.driver_dnf.get(driver_id, self.default_dnf)
        team_rate = self.team_dnf.get(team_id, self.default_dnf)
        return 0.5 * driver_rate + 0.5 * team_rate

    def sample_race(self, active_drivers, event_type='race', **kwargs):
        """
        Simula una carrera muestreando de las distribuciones normales de TrueSkill.
        """
        scores = {}
        for d in active_drivers:
            # Obtener rating (o crear por defecto si no existe)
            rating = self.ratings.get(d, self.env.create_rating())
            
            # Muestrear el rendimiento en este evento particular: N(mu, sigma^2 + beta^2)
            # Pero podemos simplemente muestrear de N(mu, sigma) y añadir ruido
            # performance = N(rating.mu, rating.sigma)
            performance = np.random.normal(rating.mu, rating.sigma)
            
            # Añadir ruido aleatorio equivalente al 'beta' (azar del día de carrera)
            performance += np.random.normal(0, self.env.beta)
            
            # Simular posibilidad de DNF (que te tira al fondo)
            team_id = self.active_driver_team.get(d)
            dnf_prob = self._get_dnf_prob(d, team_id)
            if event_type == 'sprint':
                dnf_prob *= 0.6
                
            if np.random.rand() < dnf_prob:
                # Penalización severa para mandarlo al final del orden (mu es ~25 a 50)
                performance -= (50.0 + np.random.normal(0, self.dnf_noise))
                
            scores[d] = performance
            
        # Ordenar: Mayor score es mejor posición
        orden = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return orden
