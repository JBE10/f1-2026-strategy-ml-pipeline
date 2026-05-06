import pandas as pd
from src.standardize import (
    get_historical_results,
    get_historical_sprints,
    get_2026_actual_results,
    get_2026_actual_sprints
)
df_hist = get_historical_results()
df_hist_sprint = get_historical_sprints()
df_act = get_2026_actual_results()
df_act_sprint = get_2026_actual_sprints()
df_sources = [df_hist, df_hist_sprint, df_act, df_act_sprint]
df_sources = [df for df in df_sources if df is not None and not df.empty]
df_train = pd.concat(df_sources, ignore_index=True) if df_sources else pd.DataFrame()
print("Años en df_hist:", df_hist['year'].unique() if not df_hist.empty else "Vacio")
print("Años en df_act:", df_act['year'].unique() if not df_act.empty else "Vacio")
print("Años en df_train:", df_train['year'].unique() if not df_train.empty else "Vacio")
