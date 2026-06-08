import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pickle
import os

class TireDegradationModel:
    def __init__(self):
        # Usaremos polinomios de grado 2 para modelar la curva de desgaste
        # (al principio se desgastan lento, luego caen abruptamente: "el precipicio")
        self.models = {
            'SOFT': make_pipeline(PolynomialFeatures(2), LinearRegression()),
            'MEDIUM': make_pipeline(PolynomialFeatures(2), LinearRegression()),
            'HARD': make_pipeline(PolynomialFeatures(2), LinearRegression())
        }
        self.is_trained = False
        self.trained_compounds = set()
        
    def fit(self, df_laps):
        # df_laps debe tener: 'Compound', 'TyreLife', 'LapTime_s'
        print("🔧 Entrenando curvas de degradación de neumáticos...")
        
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            df_comp = df_laps[df_laps['Compound'] == compound].copy()
            if len(df_comp) < 10:
                print(f"⚠️ Neumático {compound} omitido: solo {len(df_comp)} vueltas (mínimo 10).")
                continue
                
            X = df_comp[['TyreLife']]
            y = df_comp['LapTime_s']
            
            self.models[compound].fit(X, y)
            self.trained_compounds.add(compound)
            print(f"✅ Neumático {compound} entrenado con {len(df_comp)} vueltas.")
            
        self.is_trained = len(self.trained_compounds) > 0
        
    def predict_lap_time(self, compound, tyre_life, base_pace):
        """
        Predice el tiempo de vuelta considerando el desgaste.
        base_pace: tiempo de vuelta ideal de ese piloto con neumático nuevo.
        """
        if not self.is_trained:
            return base_pace + (tyre_life * 0.1) # Fallback tonto
            
        if compound not in self.trained_compounds:
            if 'MEDIUM' in self.trained_compounds:
                compound = 'MEDIUM'
            elif self.trained_compounds:
                compound = next(iter(self.trained_compounds))
            else:
                return base_pace + (tyre_life * 0.1)
            
        # El modelo predice un lap_time histórico general. 
        # Lo que nos importa es el DELTA (cuántos segundos más lento es en la vuelta N comparado con la 1).
        # Hacemos reshape a 2D array que es lo que espera sklearn: [[tyre_life]]
        time_lap_1 = self.models[compound].predict(np.array([[1.0]]))[0]
        time_lap_n = self.models[compound].predict(np.array([[float(tyre_life)]]))[0]
        
        degradation_penalty = max(0, time_lap_n - time_lap_1)
        
        # Efecto de combustible: el auto se vuelve más ligero y rápido cada vuelta
        # Promedio en F1: se ganan ~0.06s por vuelta por la quema de combustible
        fuel_effect = tyre_life * 0.06 
        
        return base_pace + degradation_penalty - fuel_effect

    def save(self, filepath="data/output/tire_model.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump(self.models, f)
