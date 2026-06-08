import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simulate import run_simulation
from src.models.factorial import FactorialModel
from src.models.empirico import EmpiricalModel
from src.models.trueskill_model import TrueSkillModel
from src.models.ml_model import MLModel
from src.models.tire_model import TireDegradationModel

def test_factorial_model_compatibility():
    model = FactorialModel()
    active_drivers = ['hamilton', 'russell', 'norris']
    
    # Mock data to fit the model
    df_train = pd.DataFrame({
        'year': [2025, 2025, 2025],
        'round': [1, 1, 1],
        'driverId': ['hamilton', 'russell', 'norris'],
        'constructorId': ['mercedes', 'mercedes', 'mclaren'],
        'position': [1, 2, 3],
        'points': [25, 18, 15],
        'status': ['Finished', 'Finished', 'Finished'],
        'event': ['race', 'race', 'race']
    })
    model.fit(df_train)
    
    # This should NOT fail now when called with is_street/overtaking_difficulty
    res = model.sample_race(active_drivers, event_type='race', is_street=1, overtaking_difficulty=2)
    assert len(res) == len(active_drivers)

def test_empirical_model_compatibility():
    model = EmpiricalModel()
    active_drivers = ['hamilton', 'russell', 'norris']
    
    df_train = pd.DataFrame({
        'year': [2025, 2025, 2025],
        'round': [1, 1, 1],
        'driverId': ['hamilton', 'russell', 'norris'],
        'position': [1, 2, 3]
    })
    model.fit(df_train)
    
    # This should NOT fail now when called with is_street/overtaking_difficulty
    res = model.sample_race(active_drivers, event_type='race', is_street=1, overtaking_difficulty=2)
    assert len(res) == len(active_drivers)

def test_trueskill_model_compatibility():
    model = TrueSkillModel()
    active_drivers = ['hamilton', 'russell', 'norris']
    
    df_train = pd.DataFrame({
        'year': [2025, 2025, 2025],
        'round': [1, 1, 1],
        'driverId': ['hamilton', 'russell', 'norris'],
        'constructorId': ['mercedes', 'mercedes', 'mclaren'],
        'position': [1, 2, 3],
        'status': ['Finished', 'Finished', 'Finished']
    })
    model.fit(df_train)
    
    # This should NOT fail now when called with is_street/overtaking_difficulty
    res = model.sample_race(active_drivers, event_type='race', is_street=1, overtaking_difficulty=2)
    assert len(res) == len(active_drivers)

def test_tire_model_unfitted():
    model = TireDegradationModel()
    # Fit with data only containing 'MEDIUM'
    df_laps = pd.DataFrame({
        'Compound': ['MEDIUM'] * 20,
        'TyreLife': list(range(1, 21)),
        'LapTime_s': [90.0 + 0.1 * i for i in range(20)]
    })
    model.fit(df_laps)
    
    # SOFT model is not fitted, but should fall back safely without raising NotFittedError
    time_pred = model.predict_lap_time('SOFT', 5.0, 90.0)
    assert isinstance(time_pred, (float, np.float64))

def test_fia_tie_break_actual_results():
    # Verify we successfully load actual positions in run_simulation
    current_standings = pd.DataFrame({
        'driverId': ['hamilton', 'russell'],
        'points': [25, 25]
    })
    actual_results = pd.DataFrame({
        'driverId': ['hamilton', 'russell'],
        'position': [1, 2]
    })
    model = MLModel()
    # Fit with dummy data
    df_train = pd.DataFrame({
        'year': [2025, 2025],
        'round': [1, 1],
        'driverId': ['hamilton', 'russell'],
        'constructorId': ['mercedes', 'mercedes'],
        'position': [1, 2],
        'grid': [1, 2],
        'is_street': [0, 0],
        'overtaking_difficulty': [2, 2]
    })
    model.fit(df_train)
    
    rounds_left = [{'round': 2, 'has_sprint': False, 'is_street': 0, 'overtaking_difficulty': 2}]
    res_df = run_simulation(current_standings, model, rounds_left, ['hamilton', 'russell'], B=10, seed=42, actual_results=actual_results)
    assert 'p_campeon' in res_df.columns
