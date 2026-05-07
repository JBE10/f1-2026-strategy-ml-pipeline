import os
import json
import pandas as pd
import streamlit as st

def main():
    st.set_page_config(page_title="Simulación F1 2026", layout="wide")
    st.title("🏆 Simulación F1 2026 — Estado Actual")
    
    probs_path = "data/output/probabilidades.csv"
    ensemble_path = "data/output/probabilidades_ensemble.csv"
    sample_path = "data/output/sample_path.json"
    config_path = "data/output/model_config.json"
    tire_model_path = "data/output/tire_model.pkl"
    
    if os.path.exists(probs_path):
        df_probs = pd.read_csv(probs_path)
        
        tabs = ["📊 Probabilidades del Campeonato", "📡 Telemetría (FastF1)", "⏱️ Estrategia (Canadá)", "🏎️ Simulación Carrera a Carrera (Universo #44)"]
        if os.path.exists(ensemble_path):
            tabs.insert(3, "🎲 Ensemble de Semillas")
        if os.path.exists(config_path):
            tabs.append("⚙️ Configuración del Modelo")

        tab_objs = st.tabs(tabs)
        tab1 = tab_objs[0]
        tab_telem = tab_objs[1]
        tab_strat = tab_objs[2]
        
        offset = 3
        tab_ensemble = None
        if os.path.exists(ensemble_path):
            tab_ensemble = tab_objs[offset]
            offset += 1
            
        tab2 = tab_objs[offset]
        tab_config = None
        if os.path.exists(config_path):
            tab_config = tab_objs[-1]
        
        with tab1:
            st.subheader("Favoritos al Campeonato")
            
            df_probs['p_campeon_pct'] = (df_probs['p_campeon'] * 100).round(1).astype(str) + '%'
            df_probs['p_top3_pct'] = (df_probs.get('p_top3', 0) * 100).round(1).astype(str) + '%'
            
            cols = st.columns(min(5, len(df_probs)))
            for i, col in enumerate(cols):
                if i < len(df_probs):
                    row = df_probs.iloc[i]
                    col.metric(label=row['driverId'].upper().replace('_', ' '), value=row['p_campeon_pct'])
            
            st.dataframe(df_probs[['driverId', 'p_campeon_pct', 'p_top3_pct', 'expected_points']], hide_index=True)
            st.bar_chart(data=df_probs.set_index('driverId')['p_campeon'])
            
        with tab_telem:
            st.subheader("📡 Telemetría en Vivo: Ritmo de Carrera Puro")
            st.markdown("Datos extraídos de los servidores de **F1 Live Timing (FastF1)**. Representa la mediana del tiempo por vuelta, limpiando paradas en boxes y Safety Cars.")
            
            pace_path = "data/actual/race_pace_2026_r4.csv"
            laps_path = "data/actual/valid_laps_2026_r4.csv"
            
            if os.path.exists(pace_path):
                df_pace = pd.read_csv(pace_path)
                
                col_metric, col_plot = st.columns([1, 2])
                with col_metric:
                    st.dataframe(df_pace[['Driver', 'Median_Pace_s', 'Delta_to_Leader_s']], hide_index=True)
                
                with col_plot:
                    st.markdown("#### Delta vs Líder (Segundos)")
                    df_bar = df_pace[['Driver', 'Delta_to_Leader_s']].set_index('Driver')
                    st.bar_chart(df_bar)
                
                if os.path.exists(laps_path):
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    
                    df_laps = pd.read_csv(laps_path)
                    
                    st.divider()
                    st.subheader("📊 Análisis Gráfico de Consistencia")
                    
                    col_box, col_line = st.columns(2)
                    
                    with col_box:
                        st.markdown("#### Distribución de Tiempos (Boxplot)")
                        fig_box, ax_box = plt.subplots(figsize=(10, 6))
                        # Ordenar por el más rápido según df_pace
                        order = df_pace['Driver'].tolist()
                        sns.boxplot(data=df_laps, x='Driver', y='LapTime_s', order=order, ax=ax_box, palette='viridis')
                        ax_box.set_ylabel("Tiempo de Vuelta (s)")
                        ax_box.set_xlabel("Piloto")
                        plt.xticks(rotation=45)
                        st.pyplot(fig_box)
                        st.caption("El ancho de la caja indica la consistencia. Cajas más pequeñas = piloto más constante.")

                    with col_line:
                        st.markdown("#### Evolución del Ritmo (Trend)")
                        selected_drivers = st.multiselect("Seleccionar Pilotos para comparar", order, default=order[:3])
                        if selected_drivers:
                            fig_line, ax_line = plt.subplots(figsize=(10, 6))
                            df_sub = df_laps[df_laps['Driver'].isin(selected_drivers)]
                            sns.lineplot(data=df_sub, x='LapNumber', y='LapTime_s', hue='Driver', ax=ax_line, marker='o')
                            ax_line.set_ylabel("Tiempo de Vuelta (s)")
                            ax_line.set_xlabel("Vuelta")
                            st.pyplot(fig_line)
                            st.caption("Muestra cómo varía el ritmo vuelta a vuelta (degradación y tráfico).")
            else:
                st.warning("No hay datos de telemetría extraídos. Ejecuta 'python src/telemetry_test.py'")

        with tab_strat:
            st.subheader("⏱️ Simulador de Estrategia Intra-Carrera (Canadá GP)")
            st.markdown("Modelo predictivo de degradación de neumáticos entrenado con ML (Regresión Polinómica).")
            
            if os.path.exists(tire_model_path):
                import pickle
                import numpy as np
                with open(tire_model_path, 'rb') as f:
                    tire_models = pickle.load(f)
                    
                col_inputs, col_chart = st.columns([1, 2])
                with col_inputs:
                    piloto = st.selectbox("Piloto", ["Lando Norris", "Andrea Antonelli", "Max Verstappen", "George Russell"])
                    base_pace = st.number_input("Ritmo Base Estimado (Segundos)", value=90.0, step=0.1)
                    comp_1 = st.selectbox("Neumático Inicial (Stint 1)", ["SOFT", "MEDIUM", "HARD"], index=1)
                    pit_lap = st.slider("Vuelta de Parada en Boxes (Pitstop)", min_value=5, max_value=65, value=25)
                    comp_2 = st.selectbox("Neumático Final (Stint 2)", ["SOFT", "MEDIUM", "HARD"], index=2)
                    
                with col_chart:
                    total_laps = 70
                    laps = list(range(1, total_laps + 1))
                    times = []
                    
                    for lap in laps:
                        if lap < pit_lap:
                            current_comp = comp_1
                            tyre_life = lap
                        elif lap == pit_lap:
                            current_comp = comp_2
                            tyre_life = 1
                        else:
                            current_comp = comp_2
                            tyre_life = lap - pit_lap + 1
                            
                        model = tire_models.get(current_comp, tire_models.get('MEDIUM'))
                        time_lap_1 = model.predict(np.array([[1.0]]))[0]
                        time_lap_n = model.predict(np.array([[float(tyre_life)]]))[0]
                        degradation = max(0, time_lap_n - time_lap_1)
                        fuel_effect = lap * 0.06
                        
                        lap_time = base_pace + degradation - fuel_effect
                        if lap == pit_lap:
                            lap_time += 20.0 # Pitstop penalty
                            
                        times.append(lap_time)
                        
                    df_strat = pd.DataFrame({'Vuelta': laps, 'Tiempo (s)': times}).set_index('Vuelta')
                    st.line_chart(df_strat)
                    
            else:
                st.warning("Falta entrenar el modelo de neumáticos. Ejecuta 'python src/train_tire_model.py'")
            
        if tab_ensemble is not None:
            with tab_ensemble:
                st.subheader("Ensemble de Monte Carlo (promedio de varias semillas)")
                df_ens = pd.read_csv(ensemble_path)
                df_ens['p_campeon_pct'] = (df_ens['p_campeon'] * 100).round(1).astype(str) + '%'
                df_ens['p_top3_pct'] = (df_ens.get('p_top3', 0) * 100).round(1).astype(str) + '%'
                st.dataframe(df_ens[['driverId', 'p_campeon_pct', 'p_top3_pct', 'expected_points']], hide_index=True)
                st.bar_chart(data=df_ens.set_index('driverId')['p_campeon'])

        with tab2:
            st.header("Ejemplo de una Simulación (Universo #44)")
            if os.path.exists(sample_path):
                with open(sample_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                history = data['history']
                final_standings = data['final_standings']
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🏁 Desarrollo del Campeonato")
                    for step in history:
                        rnd = step['round']
                        st.markdown(f"#### ▶ Ronda {rnd}")
                        
                        if step.get('sprint_winner'):
                            st.write(f"⏱️ **Sprint Ganador:** {step['sprint_winner'].upper().replace('_', ' ')}")
                            
                        podium = step['podium']
                        st.write(f"🏎️ **Podio:** 1º {podium[0].upper().replace('_', ' ')} | 2º {podium[1].upper().replace('_', ' ')} | 3º {podium[2].upper().replace('_', ' ')}")
                        st.divider()
                        
                with col2:
                    st.subheader("🏆 Posiciones Finales")
                    df_final = pd.DataFrame(final_standings, columns=['Piloto', 'Puntos'])
                    df_final['Piloto'] = df_final['Piloto'].apply(lambda x: x.upper().replace('_', ' '))
                    df_final.index = df_final.index + 1
                    st.dataframe(df_final.head(10))
            else:
                st.warning("No se encontró el universo de ejemplo. Ejecuta el pipeline.")

        if tab_config is not None:
            with tab_config:
                st.subheader("Configuración y parámetros usados")
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                st.json(cfg)
    else:
        st.warning("Aún no hay resultados de simulación. Ejecuta el pipeline primero.")

if __name__ == "__main__":
    main()
