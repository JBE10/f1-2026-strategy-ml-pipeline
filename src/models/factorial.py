import numpy as np
import pandas as pd


def _safe_norm(value, mean, std):
    if std == 0 or np.isnan(std):
        return 0.0
    return (value - mean) / std


class FactorialModel:
    def __init__(
        self,
        years_train=None,
        weight_mapping=None,
        recent_window=12,
        driver_weight=0.35,
        team_weight=0.55,
        form_weight=0.10,
        sigma_driver=0.35,
        sigma_team=0.30,
        sigma_noise=0.20,
        dnf_noise=0.40,
        default_dnf=0.08
    ):
        self.years_train = years_train or [2023, 2024, 2025, 2026]
        self.weight_mapping = weight_mapping or {2023: 1, 2024: 2, 2025: 4, 2026: 20}
        self.recent_window = recent_window
        self.driver_weight = driver_weight
        self.team_weight = team_weight
        self.form_weight = form_weight
        self.sigma_driver = sigma_driver
        self.sigma_team = sigma_team
        self.sigma_noise = sigma_noise
        self.dnf_noise = dnf_noise
        self.default_dnf = default_dnf

        self.driver_ratings = {}
        self.team_ratings = {}
        self.driver_form = {}
        self.driver_dnf = {}
        self.team_dnf = {}
        self.active_driver_team = {}
        self.global_stats = {}

    def _get_weight(self, year):
        return self.weight_mapping.get(int(year), 1)

    def _normalize_ratings(self, ratings):
        values = np.array(list(ratings.values()), dtype=float)
        if values.size == 0:
            return ratings, 0.0, 1.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            std = 1.0
        normalized = {k: (v - mean) / std for k, v in ratings.items()}
        return normalized, mean, std

    def fit(self, df_results):
        if df_results.empty:
            return

        df = df_results.copy()
        df = df[df['year'].isin(self.years_train)]
        if df.empty:
            df = df_results.copy()

        df = df[pd.to_numeric(df['position'], errors='coerce').notna()].copy()
        if df.empty:
            return

        df['position'] = pd.to_numeric(df['position']).astype(int)
        df['weight'] = df['year'].apply(self._get_weight)

        if 'event' not in df.columns:
            df['event'] = 'race'

        if 'raceKey' not in df.columns:
            df['raceKey'] = df.apply(lambda x: f"{int(x['year'])}-{int(x['round']):02d}", axis=1)

        df_event = df[df['event'] == 'race'].copy()
        if df_event.empty:
            df_event = df.copy()

        df_perf = df_event.copy()
        if 'status' in df_perf.columns:
            status_norm = df_perf['status'].fillna('').astype(str).str.lower()
            finish_mask = status_norm.str.contains('finished') | status_norm.str.contains('lapped')
            if finish_mask.any():
                df_perf = df_perf[finish_mask].copy()

        df_perf['inv_pos'] = 1.0 / df_perf['position'].clip(lower=1)

        use_delta = 'constructorId' in df_perf.columns and df_perf['constructorId'].notna().any()
        driver_scores = {}
        df_team = None

        if use_delta:
            df_team = df_perf.copy()
            df_team['team_count'] = df_team.groupby(['raceKey', 'constructorId'])['driverId'].transform('count')
            df_team['team_mean_pos'] = df_team.groupby(['raceKey', 'constructorId'])['position'].transform('mean')
            df_team['delta'] = 0.0
            mask = df_team['team_count'] >= 2
            df_team.loc[mask, 'delta'] = df_team.loc[mask, 'team_mean_pos'] - df_team.loc[mask, 'position']

            if mask.any():
                driver_scores = df_team.groupby('driverId').apply(
                    lambda g: np.average(g['delta'], weights=g['weight'])
                ).to_dict()
            else:
                use_delta = False

        if not use_delta:
            driver_scores = df_perf.groupby('driverId').apply(
                lambda g: np.average(g['inv_pos'], weights=g['weight'])
            ).to_dict()

        team_scores = {}
        if 'constructorId' in df_perf.columns:
            team_scores = df_perf.groupby('constructorId').apply(
                lambda g: np.average(g['inv_pos'], weights=g['weight'])
            ).to_dict()

        driver_scores_norm, driver_mean, driver_std = self._normalize_ratings(driver_scores)
        team_scores_norm, team_mean, team_std = self._normalize_ratings(team_scores)

        self.driver_ratings = driver_scores_norm
        self.team_ratings = team_scores_norm
        self.global_stats = {
            'driver_mean': driver_mean,
            'driver_std': driver_std,
            'team_mean': team_mean,
            'team_std': team_std
        }

        self.active_driver_team = {}
        if 'constructorId' in df.columns:
            df_recent = df.sort_values('year')
            last_team = df_recent.groupby('driverId').tail(1)
            for _, row in last_team.iterrows():
                self.active_driver_team[row['driverId']] = row.get('constructorId', None)

        self.driver_form = {}
        if use_delta and df_team is not None:
            for driver_id, group in df_team.groupby('driverId'):
                group_sorted = group.sort_values(['year', 'round'])
                recent = group_sorted.tail(self.recent_window)
                if recent.empty:
                    continue
                score = np.average(recent['delta'], weights=recent['weight'])
                self.driver_form[driver_id] = score
        else:
            for driver_id, group in df_perf.groupby('driverId'):
                group_sorted = group.sort_values(['year', 'round'])
                recent = group_sorted.tail(self.recent_window)
                if recent.empty:
                    continue
                score = np.average(recent['inv_pos'], weights=recent['weight'])
                self.driver_form[driver_id] = score

        form_values = np.array(list(self.driver_form.values()), dtype=float)
        if form_values.size:
            form_mean = np.mean(form_values)
            form_std = np.std(form_values) if np.std(form_values) != 0 else 1.0
        else:
            form_mean = 0.0
            form_std = 1.0

        for key in list(self.driver_form.keys()):
            self.driver_form[key] = _safe_norm(self.driver_form[key], form_mean, form_std)

        self.driver_dnf = {}
        if 'status' in df_event.columns:
            for driver_id, group in df_event.groupby('driverId'):
                statuses = group['status'].fillna('').astype(str).str.lower()
                if statuses.empty:
                    continue
                finished_mask = statuses.str.contains('finished') | statuses.str.contains('lapped')
                dnf_rate = 1.0 - (finished_mask.sum() / len(statuses))
                self.driver_dnf[driver_id] = max(0.0, min(0.6, dnf_rate))

        self.team_dnf = {}
        if 'status' in df_event.columns and 'constructorId' in df_event.columns:
            for team_id, group in df_event.groupby('constructorId'):
                statuses = group['status'].fillna('').astype(str).str.lower()
                if statuses.empty:
                    continue
                finished_mask = statuses.str.contains('finished') | statuses.str.contains('lapped')
                dnf_rate = 1.0 - (finished_mask.sum() / len(statuses))
                self.team_dnf[team_id] = max(0.0, min(0.6, dnf_rate))

    def _driver_team(self, driver_id):
        return self.active_driver_team.get(driver_id)

    def _get_rating(self, driver_id):
        return self.driver_ratings.get(driver_id, 0.0)

    def _get_team_rating(self, team_id):
        if not team_id:
            return 0.0
        return self.team_ratings.get(team_id, 0.0)

    def _get_form(self, driver_id):
        return self.driver_form.get(driver_id, 0.0)

    def _get_dnf_prob(self, driver_id, team_id=None):
        driver_rate = self.driver_dnf.get(driver_id, self.default_dnf)
        team_rate = self.team_dnf.get(team_id, self.default_dnf)
        return 0.5 * driver_rate + 0.5 * team_rate

    def sample_race(self, active_drivers, event_type='race'):
        scores = {}
        event_factor = 1.0 if event_type == 'race' else 0.85

        for d in active_drivers:
            team_id = self._driver_team(d)
            driver_score = self._get_rating(d)
            team_score = self._get_team_rating(team_id)
            form_score = self._get_form(d)

            noise = np.random.normal(0, self.sigma_noise)
            driver_noise = np.random.normal(0, self.sigma_driver)
            team_noise = np.random.normal(0, self.sigma_team)

            base = (
                self.driver_weight * (driver_score + driver_noise) +
                self.team_weight * (team_score + team_noise) +
                self.form_weight * form_score
            ) * event_factor

            dnf_prob = self._get_dnf_prob(d, team_id)
            if event_type == 'sprint':
                dnf_prob *= 0.6
            if np.random.rand() < dnf_prob:
                dnf_penalty = np.random.uniform(2.0, 3.5) + np.random.normal(0, self.dnf_noise)
                base -= dnf_penalty

            scores[d] = base + noise

        orden = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return orden
