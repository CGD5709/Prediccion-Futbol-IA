import pandas as pd

import pandas as pd

def calcular_acumulados_temporada(df):
    """
    Calcula los puntos y la diferencia de goles acumulada por cada equipo 
    en la temporada actual, usando el índice original para evitar problemas de merge.
    """
    df_temp = df.copy()
    
    # 1. Separar locales y visitantes
    df_local = df_temp[['id', 'fecha', 'anio_fin_temporada', 'local', 'puntos_local', 'goles_local', 'goles_visitante']].copy()
    df_local.rename(columns={'local': 'equipo', 'puntos_local': 'puntos'}, inplace=True)
    df_local['dif_goles_partido'] = df_local['goles_local'] - df_local['goles_visitante']
    df_local['juega_de'] = 'local'
    
    df_visitante = df_temp[['id', 'fecha', 'anio_fin_temporada', 'visitante', 'puntos_visitante', 'goles_visitante', 'goles_local']].copy()
    df_visitante.rename(columns={'visitante': 'equipo', 'puntos_visitante': 'puntos'}, inplace=True)
    df_visitante['dif_goles_partido'] = df_visitante['goles_visitante'] - df_visitante['goles_local']
    df_visitante['juega_de'] = 'visitante'
    
    # 2. Unimos y ordenamos cronológicamente, reseteando el índice para evitar duplicados
    df_historial = pd.concat([df_local, df_visitante])
    df_historial = df_historial.sort_values(by=['equipo', 'fecha']).reset_index(drop=True)
    
    # 3. Calculamos los acumulados por equipo y temporada, usando shift para no incluir el partido actual
    df_historial['puntos_acumulados'] = df_historial.groupby(['anio_fin_temporada', 'equipo'])['puntos'].transform(
        lambda x: x.shift(1).cumsum()
    ).fillna(0)
    
    df_historial['dif_goles_acumulada'] = df_historial.groupby(['anio_fin_temporada', 'equipo'])['dif_goles_partido'].transform(
        lambda x: x.shift(1).cumsum()
    ).fillna(0)
    
    # 4. Volvemos a separar en Local y Visitante para preparar el cruce
    cols_extraer = ['id', 'puntos_acumulados', 'dif_goles_acumulada']
    
    df_locales_final = df_historial[df_historial['juega_de'] == 'local'][cols_extraer]
    df_locales_final.rename(columns={'puntos_acumulados': 'pts_acum_local', 'dif_goles_acumulada': 'dif_goles_acum_local'}, inplace=True)
    
    df_visitantes_final = df_historial[df_historial['juega_de'] == 'visitante'][cols_extraer]
    df_visitantes_final.rename(columns={'puntos_acumulados': 'pts_acum_visitante', 'dif_goles_acumulada': 'dif_goles_acum_visitante'}, inplace=True)
    
    # 5. Cruzamos con el dataset original usando el 'id' único de cada partido
    df_resultado = df_temp.merge(df_locales_final, on='id', how='left')
    df_resultado = df_resultado.merge(df_visitantes_final, on='id', how='left')
    
    return df_resultado


def calcular_rachas_recientes(df, ventana):
    """
    Calcula los puntos sumados y el promedio de goles a favor/en contra
    en los últimos partidos (3 o 5), respetando la temporada actual
    e iniciando a 0 en el primer partido.
    """
    df_temp = df.copy()
    
    # 1. Separar locales y visitantes
    df_local = df_temp[['id', 'fecha', 'anio_fin_temporada', 'local', 'puntos_local', 'goles_local', 'goles_visitante']].copy()
    df_local.rename(columns={'local': 'equipo', 'puntos_local': 'puntos', 'goles_local': 'goles_favor', 'goles_visitante': 'goles_contra'}, inplace=True)
    df_local['juega_de'] = 'local'
    
    df_visitante = df_temp[['id', 'fecha', 'anio_fin_temporada', 'visitante', 'puntos_visitante', 'goles_visitante', 'goles_local']].copy()
    df_visitante.rename(columns={'visitante': 'equipo', 'puntos_visitante': 'puntos', 'goles_visitante': 'goles_favor', 'goles_local': 'goles_contra'}, inplace=True)
    df_visitante['juega_de'] = 'visitante'
    
    # 3. Unir, ordenar y resetear índice para evitar problemas de merge
    df_historial = pd.concat([df_local, df_visitante]).sort_values(by=['equipo', 'fecha']).reset_index(drop=True)
    
    # 4. Calcular las rachas usando rolling con shift para no incluir el partido actual, y fillna(0) para iniciar a 0
    # Puntos sumados en los últimos N partidos
    df_historial[f'pts_ultimos_{ventana}'] = df_historial.groupby(['anio_fin_temporada', 'equipo'])['puntos'].transform(
        lambda x: x.shift(1).rolling(ventana, min_periods=1).sum()
    ).fillna(0)
    
    # Promedio de goles a favor en los últimos N partidos
    df_historial[f'gf_media_{ventana}'] = df_historial.groupby(['anio_fin_temporada', 'equipo'])['goles_favor'].transform(
        lambda x: x.shift(1).rolling(ventana, min_periods=1).mean()
    ).fillna(0)
    
    # Promedio de goles en contra en los últimos N partidos
    df_historial[f'gc_media_{ventana}'] = df_historial.groupby(['anio_fin_temporada', 'equipo'])['goles_contra'].transform(
        lambda x: x.shift(1).rolling(ventana, min_periods=1).mean()
    ).fillna(0)
    
    # 5. Volver a separar para pegar en el dataset original
    cols_mantener = ['id', f'pts_ultimos_{ventana}', f'gf_media_{ventana}', f'gc_media_{ventana}']
    
    df_locales_final = df_historial[df_historial['juega_de'] == 'local'][cols_mantener]
    # Renombramos añadiendo el sufijo _local
    df_locales_final.columns = ['id', f'pts_ultimos_{ventana}_local', f'gf_media_{ventana}_local', f'gc_media_{ventana}_local']
    
    df_visitantes_final = df_historial[df_historial['juega_de'] == 'visitante'][cols_mantener]
    # Renombramos añadiendo el sufijo _visitante
    df_visitantes_final.columns = ['id', f'pts_ultimos_{ventana}_visitante', f'gf_media_{ventana}_visitante', f'gc_media_{ventana}_visitante']
    
    # 6. Cruzar (merge) usando el ID
    df_resultado = df_temp.merge(df_locales_final, on='id', how='left')
    df_resultado = df_resultado.merge(df_visitantes_final, on='id', how='left')
    
    return df_resultado

def calcular_rendimiento_local_visitante(df):
    """
    Calcula el promedio de puntos del equipo local jugando en casa 
    y del equipo visitante jugando fuera, estrictamente antes del partido actual.
    """
    df_temp = df.copy()
    
    # 1. RENDIMIENTO DEL EQUIPO LOCAL (EN CASA)
    df_local = df_temp[['id', 'fecha', 'anio_fin_temporada', 'local', 'puntos_local']].copy()
    df_local = df_local.sort_values(by=['local', 'fecha']).reset_index(drop=True)
    
    # Contamos el número de partidos jugados en casa hasta hoy para cada equipo y temporada
    df_local['partidos_jugados_casa'] = df_local.groupby(['anio_fin_temporada', 'local']).cumcount()
    
    # Suma de puntos conseguidos en casa hasta hoy
    df_local['pts_acum_casa'] = df_local.groupby(['anio_fin_temporada', 'local'])['puntos_local'].transform(
        lambda x: x.shift(1).cumsum()
    ).fillna(0)
    
    # Calculamos la media. Si es el primer partido, dividirá entre 0 (dará NaN). Lo rellenamos con 0.
    df_local['promedio_pts_casa'] = (df_local['pts_acum_casa'] / df_local['partidos_jugados_casa']).fillna(0)
    
    # 2. RENDIMIENTO DEL EQUIPO VISITANTE (FUERA)
    df_visit = df_temp[['id', 'fecha', 'anio_fin_temporada', 'visitante', 'puntos_visitante']].copy()
    df_visit = df_visit.sort_values(by=['visitante', 'fecha']).reset_index(drop=True)
    
    df_visit['partidos_jugados_fuera'] = df_visit.groupby(['anio_fin_temporada', 'visitante']).cumcount()
    
    df_visit['pts_acum_fuera'] = df_visit.groupby(['anio_fin_temporada', 'visitante'])['puntos_visitante'].transform(
        lambda x: x.shift(1).cumsum()
    ).fillna(0)
    
    df_visit['promedio_pts_fuera'] = (df_visit['pts_acum_fuera'] / df_visit['partidos_jugados_fuera']).fillna(0)
    
    # 3. RECORTAR Y PEGAR AL DATASET ORIGINAL
    cols_local = ['id', 'promedio_pts_casa']
    cols_visit = ['id', 'promedio_pts_fuera']
    
    df_resultado = df_temp.merge(df_local[cols_local], on='id', how='left')
    df_resultado = df_resultado.merge(df_visit[cols_visit], on='id', how='left')
    
    return df_resultado