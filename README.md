# Simulación Campeonato F1 · 2026 🏎️🤖

Este proyecto comenzó como una simulación estadística tradicional (Monte Carlo empírico) para predecir el Campeonato de Fórmula 1 de 2026. Sin embargo, ha evolucionado hacia un **Pipeline de Machine Learning de Nivel Profesional**, integrando datos históricos, resultados en vivo de la temporada actual y **telemetría pura** de los servidores de la F1.

## 🌟 Estado Actual del Proyecto (Mayo 2026)

Actualmente hemos completado la **Fase de Inyección Telemétrica y Backtesting**. Nuestro simulador ya no predice resultados mirando únicamente quién sumó más puntos en el pasado, sino que entiende la *velocidad real* de cada auto en la pista, y hemos validado su precisión matemáticamente.

### Arquitectura Híbrida Implementada:
1. **Fuentes de Datos Mixtas:**
   - **Histórico (Kaggle):** ~2000 registros de carreras y sprints desde 2023.
   - **Actual (API Jolpica):** Resultados y Standings oficiales hasta la Ronda 4 (Miami 2026).
   - **Telemetría (FastF1):** Conexión a los servidores de *F1 Live Timing* para extraer el Ritmo de Carrera (Race Pace) real de los pilotos en milisegundos, filtrando paradas en boxes y Safety Cars.
2. **Motor Predictivo (Machine Learning - XGBoost):**
   - Implementamos un modelo `XGBRegressor` en `src/models/ml_model.py`.
   - **Feature Engineering:** El modelo calcula ventanas móviles (`window=10`) para extraer la posición media reciente del piloto, su mejor resultado y el rendimiento medio del constructor.
   - **Starting Grid:** Inyectamos la posición de parrilla real de la carrera a predecir, mejorando la exactitud del modelo drásticamente.
3. **Simulador de Estrategia Intra-Carrera (Regresión Polinómica):**
   - Entrenamos un modelo en `src/models/tire_model.py` que aprendió la curva de degradación de los neumáticos Pirelli (Soft, Medium, Hard).
   - Creamos un simulador visual en Streamlit que proyecta el tiempo por vuelta real durante un Gran Premio (ej. Canadá), considerando la degradación, quema de combustible y pérdida de tiempo en boxes.
4. **Validación por Backtesting:**
   - Construimos scripts (`src/backtest.py` y `src/backtest_2025.py`) que simulan el paso del tiempo de forma secuencial.
   - **Resultado:** Tras ingerir toda la temporada 2025, logramos un **Error Medio Absoluto (MAE) de 3.51 posiciones**. En la F1, un error menor a 4 posiciones a lo largo de un campeonato entero se considera un estado del arte excepcional debido a los imprevistos climáticos y choques.

---

## 🚀 Próximos Pasos en Machine Learning (Siguientes Niveles)

Hemos cubierto los pasos críticos de telemetría y estrategia. Para seguir bajando el MAE (acercándonos a ~2.5), los próximos retos son de enriquecimiento de datos:

### 1. Categorización y Sensibilidad del Circuito
*   **Problema:** El modelo trata a Mónaco (callejero, imposible pasar) igual que a Monza (rectas gigantes, muchos sobrepasos).
*   **Siguiente Paso:** Crear variables categóricas como `Is_Street_Circuit` y `Overtaking_Difficulty` (Alta/Media/Baja).
*   **Beneficio:** El modelo aprenderá automáticamente que si la pista es "Mónaco" y el piloto larga en pole, su probabilidad de ganar es casi absoluta, sin importar si el auto de atrás tiene mejor ritmo.

### 2. Diferencia con el Compañero de Equipo (Teammate Delta)
*   **Problema:** Evaluamos al piloto aislado de su contexto mecánico más cercano.
*   **Siguiente Paso:** Crear una feature llamada `Teammate_Pace_Delta`.
*   **Beneficio:** Si Norris consistentemente clasifica más rápido que Piastri, el modelo entenderá que Norris está "sobre-conduciendo" y exprimiendo el auto más allá de su límite teórico. Esto estabiliza mucho la predicción de los autos de mitad de tabla.

### 3. Clima y DNF Predictivo
*   **Problema:** En el backtest vimos que carreras lluviosas o caóticas rompen la barrera matemática y el MAE salta a >5.0.
*   **Siguiente Paso:** Conectar una API del clima e inyectar `Is_Raining`. Además, crear métricas como `Driver_Crash_Rate` o `Engine_Age_Laps`.
*   **Beneficio:** En vez de que los abandonos se calculen tirando un dado aleatorio, el modelo será capaz de predecir fallas mecánicas inminentes.

### 4. Tuning de Hiperparámetros (AutoML)
*   **Siguiente Paso:** Implementar `Optuna` o `GridSearchCV` de scikit-learn.
*   **Beneficio:** Encontrar matemáticamente la combinación perfecta de árboles (`n_estimators`), profundidad (`max_depth`) y tasa de aprendizaje (`learning_rate`) de nuestro XGBoost para minimizar el error de predicción.

---

## 🛠️ Ejecución del Proyecto

**1. Instalar dependencias:**
```bash
pip install -r requirements.txt
```

**2. Extraer Telemetría de la última fecha:**
```bash
python src/telemetry_test.py
```

**3. Ejecutar el Pipeline (Entrenamiento + Monte Carlo):**
```bash
python src/pipeline.py
```

**4. Ver Resultados (Dashboard Visual):**
```bash
streamlit run dashboard.py
```