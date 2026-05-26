import pandas as pd
import numpy as np

# ==========================================
# 1. DICCIONARIO DE TRADUCCIÓN DE EQUIPOS
# ==========================================

MAPEO_EQUIPOS = {
    # ── La Liga ──────────────────────────────────
    "Real Madrid":             "Real Madrid",     
    "FC Barcelona":            "Barcelona",
    "Atlético de Madrid":      "Ath Madrid",
    "Athletic Bilbao":         "Ath Bilbao",
    "Real Betis Balompié":     "Betis",
    "Valencia CF":             "Valencia",
    "Sevilla FC":              "Sevilla",
    "Villarreal CF":           "Villarreal",
    "Real Sociedad":           "Sociedad",
    "Celta de Vigo":           "Celta",
    "CA Osasuna":              "Osasuna",
    "RCD Espanyol Barcelona":  "Espanol",
    "RCD Espanyol":            "Espanol",         
    "RCD Mallorca":            "Mallorca",
    "Getafe CF":               "Getafe",
    "Deportivo Alavés":        "Alaves",
    "CD Alavés":               "Alaves",        
    "Levante UD":              "Levante",
    "UD Almería":              "Almeria",
    "Girona FC":               "Girona",
    "Rayo Vallecano":          "Vallecano",
    "UD Las Palmas":           "Las Palmas",
    "Real Valladolid CF":      "Valladolid",
    "Real Valladolid Deportivo": "Valladolid",     
    "SD Eibar":                "Eibar",
    "CD Leganés":              "Leganes",
    "Granada CF":              "Granada",
    "Cádiz CF":                "Cadiz",
    "Elche CF":                "Elche",
    "SD Huesca":               "Huesca",
    "Málaga CF":               "Malaga",
    "Real Zaragoza":           "Zaragoza",
    "Deportivo de La Coruña":  "La Coruna",
    "RC Deportivo La Coruña":  "La Coruna",        
    "RC Celta de Vigo":        "Celta",      
    "Racing de Santander":     "Santander",
    "Racing Santander":        "Santander",         
    "CD Tenerife":             "Tenerife",
    "Real Oviedo":             "Oviedo",
    "Albacete Balompié":       "Albacete",
    "CD Numancia":             "Numancia",
    "Hércules CF":             "Hercules",
    "Córdoba CF":              "Cordoba",
    "Gimnàstic de Tarragona":  "Gimnastic",
    "Real Sporting de Gijón":  "Sp Gijon",
    "Sporting de Gijón":       "Sp Gijon",         
    "Sporting Gijón":          "Sp Gijon",         
    "Real Murcia CF":          "Murcia",
    "Recreativo de Huelva":    "Recreativo",
    "Recreativo Huelva":       "Recreativo",       
    "Xerez CD":                "Xerez",
    "AD Rayo Vallecano":       "Vallecano",        
    # ── Premier League ───────────────────────
    "Arsenal FC":              "Arsenal",
    "Chelsea FC":              "Chelsea",
    "Liverpool FC":            "Liverpool",
    "Manchester City":         "Man City",
    "Manchester United":       "Man United",
    "Tottenham Hotspur":       "Tottenham",
    "Newcastle United":        "Newcastle",
    "Aston Villa":             "Aston Villa",
    "West Ham United":         "West Ham",
    "Brighton & Hove Albion":  "Brighton",
    "AFC Bournemouth":         "Bournemouth",
    "Wolverhampton Wanderers": "Wolves",
    "Fulham FC":               "Fulham",
    "Brentford FC":            "Brentford",
    "Crystal Palace":          "Crystal Palace",
    "Everton FC":              "Everton",
    "Nottingham Forest":       "Nott'm Forest",
    "Leicester City":          "Leicester",
    "Southampton FC":          "Southampton",
    "Leeds United":            "Leeds",
    "Burnley FC":              "Burnley",
    "Sheffield United":        "Sheffield United",
    "Watford FC":              "Watford",
    "Norwich City":            "Norwich",
    "West Bromwich Albion":    "West Brom",
    "Ipswich Town":            "Ipswich",
    "Stoke City":              "Stoke",
    "Sunderland AFC":          "Sunderland",
    "Swansea City":            "Swansea",
    "Cardiff City":            "Cardiff",
    "Hull City":               "Hull",
    "Middlesbrough FC":        "Middlesbrough",
    "Queens Park Rangers":     "QPR",
    "Reading FC":              "Reading",
    "Huddersfield Town":       "Huddersfield",
    "Wigan Athletic":          "Wigan",
    "Bolton Wanderers":        "Bolton",
    "Blackburn Rovers":        "Blackburn",
    "Birmingham City":         "Birmingham",
    "Charlton Athletic":       "Charlton",
    "Portsmouth FC":           "Portsmouth",
    "Derby County":            "Derby",
    "Blackpool FC":            "Blackpool",
    "Bradford City":           "Bradford",
    "Coventry City":           "Coventry",
}


# ==========================================
# 2. FUNCIONES AUXILIARES PRIVADAS
# ==========================================

def estandarizar_nombres(df, columna_equipo):
    df_limpio = df.copy()
    df_limpio[columna_equipo] = df_limpio[columna_equipo].replace(MAPEO_EQUIPOS)
    return df_limpio


def _anio_completo_str(x):
    """Convierte año abreviado ('22') a año completo (2022)."""
    if len(x) == 2:
        n = int(x)
        # Asumimos que 00-50 pertenecen a 2000s, y 51-99 a 1900s
        return 2000 + n if n <= 50 else 1900 + n
    return int(x)


def _cargar_contexto_traspasos(ruta_transfers, ruta_rendimiento):
    """
    Carga y prepara los datos comunes para funciones que usan traspasos
    para determinar la composición de plantillas.
    """
    df_transfers = pd.read_csv(ruta_transfers)
    df_transfers['transfer_date'] = pd.to_datetime(df_transfers['transfer_date'])

    df_rend = pd.read_csv(
        ruta_rendimiento, usecols=['team_id', 'team_name', 'season_name']
    )

    # Mapeo team_id → team_name (desde performances)
    team_id_to_name = (
        df_rend[['team_id', 'team_name']]
        .drop_duplicates('team_id', keep='last')
        .set_index('team_id')['team_name']
        .to_dict()
    )

    # Pseudo-equipos a excluir
    destinos_excluir = df_transfers[
        df_transfers['to_team_name'].isin(
            ['Retired', 'Without Club', 'Unknown', 'Career break']
        )
    ]['to_team_id'].unique()

    # Temporadas y equipos
    df_rend['season'] = (
        df_rend['season_name'].astype(str).str.split('/').str[0]
    )
    df_rend['season'] = df_rend['season'].apply(_anio_completo_str)
    temporadas_equipos = df_rend[['season', 'team_id']].drop_duplicates()

    return df_transfers, team_id_to_name, temporadas_equipos, destinos_excluir


def _construir_plantilla_temporada(
    df_transfers, team_id_to_name, equipos_temp, destinos_excluir, cutoff
):
    """
    Construye la plantilla de los equipos indicados para una temporada.
    Busca el último traspaso de cada jugador antes del cutoff.
    """
    cols_vacio = ['player_id', 'to_team_id', 'team_name']

    trans_pre = df_transfers[df_transfers['transfer_date'] < cutoff]
    if trans_pre.empty:
        return pd.DataFrame(columns=cols_vacio)

    ultimo_mov = (
        trans_pre
        .sort_values('transfer_date')
        .groupby('player_id')
        .last()
        .reset_index()
    )
    ultimo_mov = ultimo_mov[~ultimo_mov['to_team_id'].isin(destinos_excluir)]
    plantilla = ultimo_mov[
        ultimo_mov['to_team_id'].isin(equipos_temp)
    ][['player_id', 'to_team_id']].copy()

    if plantilla.empty:
        return pd.DataFrame(columns=cols_vacio)

    plantilla['team_name'] = plantilla['to_team_id'].map(team_id_to_name)
    plantilla = plantilla.dropna(subset=['team_name'])

    return plantilla


# ==========================================
# 3. VARIABLES DE PLANTILLA (INICIO DE TEMPORADA)
# ==========================================

def preparar_valor_plantilla_inicio_temporada(
    ruta_valuations, ruta_rendimiento, ruta_transfers
):
    """
    Calcula el valor de la plantilla al inicio de la temporada actual (hasta el 1 de agosto).

    """
    df_valuations = pd.read_csv(ruta_valuations)
    df_valuations['fecha_valoracion'] = pd.to_datetime(df_valuations['date_unix'])

    df_transfers, team_id_to_name, temporadas_equipos, destinos_excluir = \
        _cargar_contexto_traspasos(ruta_transfers, ruta_rendimiento)

    resultados = []

    for season_year in temporadas_equipos['season'].unique():
        cutoff = pd.Timestamp(f'{season_year}-08-01')
        equipos_temp = set(
            temporadas_equipos[
                temporadas_equipos['season'] == season_year
            ]['team_id']
        )

        plantilla = _construir_plantilla_temporada(
            df_transfers, team_id_to_name, equipos_temp, destinos_excluir, cutoff
        )
        if plantilla.empty:
            continue

        # Filtrar valoraciones previas al cutoff de los jugadores en la plantilla
        mask = (
            (df_valuations['player_id'].isin(plantilla['player_id'])) &
            (df_valuations['fecha_valoracion'] < cutoff)
        )
        vals_pre = df_valuations[mask]

        if vals_pre.empty:
            continue

        # Obtener última valoración de cada jugador
        ultima_val = (
            vals_pre
            .sort_values('fecha_valoracion')
            .groupby('player_id')['value']
            .last()
            .reset_index()
        )

        # Unir valoración con plantilla y sumar por equipo
        merged = pd.merge(ultima_val, plantilla, on='player_id', how='inner')
        por_equipo = merged.groupby('team_name')['value'].sum().reset_index()
        por_equipo['season'] = season_year
        resultados.append(por_equipo)

    df_resultado = pd.concat(resultados, ignore_index=True)
    df_resultado['season_join_key'] = df_resultado['season'] + 1

    df_resultado = estandarizar_nombres(df_resultado, 'team_name')
    df_resultado = df_resultado.rename(columns={
        'team_name': 'equipo',
        'value': 'valor_plantilla_inicio'
    })

    return df_resultado[['equipo', 'season_join_key', 'valor_plantilla_inicio']]


def preparar_altura_edad_plantilla(
    ruta_profiles, ruta_transfers, ruta_rendimiento
):
    """
    Calcula la altura media y la edad media de la plantilla al inicio de la temporada (hasta el 1 de agosto).
    """
    df_profiles = pd.read_csv(
        ruta_profiles, usecols=['player_id', 'height', 'date_of_birth']
    )
    df_profiles['date_of_birth'] = pd.to_datetime(
        df_profiles['date_of_birth'], errors='coerce'
    )

    df_transfers, team_id_to_name, temporadas_equipos, destinos_excluir = \
        _cargar_contexto_traspasos(ruta_transfers, ruta_rendimiento)

    resultados = []

    for season_year in temporadas_equipos['season'].unique():
        cutoff = pd.Timestamp(f'{season_year}-08-01')
        equipos_temp = set(
            temporadas_equipos[
                temporadas_equipos['season'] == season_year
            ]['team_id']
        )

        plantilla = _construir_plantilla_temporada(
            df_transfers, team_id_to_name, equipos_temp, destinos_excluir, cutoff
        )
        if plantilla.empty:
            continue

        merged = pd.merge(plantilla, df_profiles, on='player_id', how='inner')
        if merged.empty:
            continue

        # Altura media (excluyendo alturas faltantes o <= 0)
        m_h = merged[merged['height'] > 0]
        if not m_h.empty:
            h_eq = m_h.groupby('team_name')['height'].mean().reset_index()
            h_eq.columns = ['team_name', 'altura_media_plantilla']
        else:
            h_eq = pd.DataFrame(columns=['team_name', 'altura_media_plantilla'])

        # Edad media (a fecha 1 de agosto)
        merged['edad'] = (cutoff - merged['date_of_birth']).dt.days / 365.25
        m_e = merged[merged['edad'].between(14, 50, inclusive='both')]
        if not m_e.empty:
            e_eq = m_e.groupby('team_name')['edad'].mean().reset_index()
            e_eq.columns = ['team_name', 'edad_media_plantilla']
        else:
            e_eq = pd.DataFrame(columns=['team_name', 'edad_media_plantilla'])

        combinado = pd.merge(h_eq, e_eq, on='team_name', how='outer')
        combinado['season'] = season_year
        resultados.append(combinado)

    df_resultado = pd.concat(resultados, ignore_index=True)
    df_resultado['season_join_key'] = df_resultado['season'] + 1

    df_resultado = estandarizar_nombres(df_resultado, 'team_name')
    df_resultado = df_resultado.rename(columns={'team_name': 'equipo'})

    df_resultado['altura_media_plantilla'] = df_resultado['altura_media_plantilla'].round(1)
    df_resultado['edad_media_plantilla'] = df_resultado['edad_media_plantilla'].round(2)

    return df_resultado[[
        'equipo', 'season_join_key',
        'altura_media_plantilla', 'edad_media_plantilla'
    ]]


# ==========================================
# 4. VARIABLES DE LESIONES POR PARTIDO
# ==========================================

def _cargar_y_limpiar_lesiones(ruta_injuries, df_p):
    """Carga y limpia el dataset de lesiones, filtrando por el rango de fechas de los partidos."""
    df_injuries = pd.read_csv(ruta_injuries)
    df_injuries['from_date'] = pd.to_datetime(df_injuries['from_date'], errors='coerce')
    df_injuries['end_date'] = pd.to_datetime(df_injuries['end_date'], errors='coerce')
    
    # Rellenar end_date faltantes usando days_missed
    mask_end_nan = df_injuries['end_date'].isna() & df_injuries['days_missed'].notna()
    df_injuries.loc[mask_end_nan, 'end_date'] = (
        df_injuries.loc[mask_end_nan, 'from_date'] + pd.to_timedelta(df_injuries.loc[mask_end_nan, 'days_missed'], unit='D')
    )
    df_injuries = df_injuries.dropna(subset=['from_date'])

    # Pre-filtrar al rango temporal de los partidos
    fecha_min = df_p['fecha'].min()
    fecha_max = df_p['fecha'].max()

    return df_injuries[
        (df_injuries['from_date'] <= fecha_max) &
        ((df_injuries['end_date'] >= fecha_min) | df_injuries['end_date'].isna())
    ].copy()

def _asignar_equipo_y_valor_lesiones(
    df_injuries, ruta_transfers, ruta_valuations, ruta_rendimiento, equipos_train
):
    """
    Cruza el dataset de lesiones con los traspasos y valoraciones para asignar
    un equipo y un valor de mercado a cada jugador lesionado, en la fecha exacta
    en que se produjo la lesión.
    """
    df_valuations = pd.read_csv(ruta_valuations)
    df_valuations['fecha_valoracion'] = pd.to_datetime(df_valuations['date_unix'])

    df_transfers = pd.read_csv(ruta_transfers)
    df_transfers['transfer_date'] = pd.to_datetime(df_transfers['transfer_date'])

    df_rend = pd.read_csv(ruta_rendimiento, usecols=['team_id', 'team_name'])
    team_id_to_name = (
        df_rend.drop_duplicates('team_id', keep='last')
        .set_index('team_id')['team_name'].to_dict()
    )

    destinos_excluir = set(df_transfers[
        df_transfers['to_team_name'].isin(['Retired', 'Without Club', 'Unknown', 'Career break'])
    ]['to_team_id'].unique())

    trans_limpio = (
        df_transfers[~df_transfers['to_team_id'].isin(destinos_excluir)]
        [['player_id', 'transfer_date', 'to_team_id']]
        .dropna(subset=['transfer_date', 'player_id'])
        .sort_values('transfer_date')
    )

    inj_sorted = df_injuries.dropna(subset=['from_date', 'player_id']).sort_values('from_date')

    inj_con_equipo = pd.merge_asof(
        inj_sorted, trans_limpio,
        left_on='from_date', right_on='transfer_date',
        by='player_id', direction='backward'
    )

    inj_con_equipo['team_name'] = inj_con_equipo['to_team_id'].map(team_id_to_name)
    inj_con_equipo = inj_con_equipo.dropna(subset=['team_name'])
    inj_con_equipo = estandarizar_nombres(inj_con_equipo, 'team_name')
    inj_con_equipo = inj_con_equipo.rename(columns={'team_name': 'equipo'})

    inj_relevantes = inj_con_equipo[inj_con_equipo['equipo'].isin(equipos_train)].copy()

    vals_sorted = (
        df_valuations[['player_id', 'fecha_valoracion', 'value']]
        .dropna(subset=['fecha_valoracion', 'player_id'])
        .sort_values('fecha_valoracion')
    )
    inj_relevantes = inj_relevantes.dropna(subset=['from_date', 'player_id']).sort_values('from_date')

    inj_con_valor = pd.merge_asof(
        inj_relevantes, vals_sorted,
        left_on='from_date', right_on='fecha_valoracion',
        by='player_id', direction='backward'
    )
    inj_con_valor['value'] = inj_con_valor['value'].fillna(0)

    return inj_con_valor[['equipo', 'from_date', 'end_date', 'player_id', 'value']].copy()


def _lesiones_modo_preciso(inj_final, df_p, equipos_train):
    """Calcula las lesiones activas partido a partido (exacto, pero lento)."""
    results = []
    for equipo in sorted(equipos_train):
        team_inj = inj_final[inj_final['equipo'] == equipo]
        if team_inj.empty:
            continue

        team_matches = pd.concat([
            df_p[df_p['local'] == equipo][['id', 'fecha']].assign(role='local'),
            df_p[df_p['visitante'] == equipo][['id', 'fecha']].assign(role='visitante')
        ]).reset_index(drop=True)

        if team_matches.empty:
            continue

        m_dates = team_matches['fecha'].values
        i_from = team_inj['from_date'].values
        i_end = team_inj['end_date'].values.copy()
        
        # Las lesiones sin end_date se asumen activas indefinidamente
        nat_mask = np.isnat(i_end)
        if nat_mask.any():
            i_end = i_end.copy()
            i_end[nat_mask] = np.datetime64('2099-12-31')

        # Broadcasting para ver qué lesiones están activas en qué partido
        active = (m_dates[:, None] >= i_from[None, :]) & (m_dates[:, None] <= i_end[None, :])

        num_list = np.zeros(len(m_dates), dtype=int)
        coste_list = np.zeros(len(m_dates), dtype=float)

        i_pids = team_inj['player_id'].values
        i_vals = team_inj['value'].values

        for m_idx in range(len(m_dates)):
            mask = active[m_idx]
            if not mask.any():
                continue
            
            pids = i_pids[mask]
            vals = i_vals[mask]
            uniq_pids, uniq_idx = np.unique(pids, return_index=True)
            num_list[m_idx] = len(uniq_pids)
            coste_list[m_idx] = vals[uniq_idx].sum()

        team_result = team_matches[['id', 'role']].copy()
        team_result['num_lesionados'] = num_list
        team_result['coste_lesionados'] = coste_list
        results.append(team_result)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=['id', 'role', 'num_lesionados', 'coste_lesionados'])


def preparar_lesiones_por_partido(
    ruta_injuries, ruta_transfers, ruta_valuations,
    ruta_rendimiento, df_partidos
):
    """
    Calcula el número y coste de jugadores lesionados para cada partido.
    Sin data leakage: se usa el historial pasado de transferencias y valoraciones
    para determinar dónde juega y cuánto vale cada lesionado.
    """
    df_p = df_partidos[['id', 'fecha', 'local', 'visitante']].copy()
    df_p['fecha'] = pd.to_datetime(df_p['fecha'])
    equipos_train = set(df_p['local'].unique()) | set(df_p['visitante'].unique())

    df_injuries = _cargar_y_limpiar_lesiones(ruta_injuries, df_p)
    inj_final = _asignar_equipo_y_valor_lesiones(
        df_injuries, ruta_transfers, ruta_valuations, ruta_rendimiento, equipos_train
    )

    result = _lesiones_modo_preciso(inj_final, df_p, equipos_train)

    # Pivotar de formato largo (role=local/visitante) a formato ancho
    result_pivot = result.pivot_table(
        index='id', columns='role',
        values=['num_lesionados', 'coste_lesionados'],
        aggfunc='first'
    )
    result_pivot.columns = [f'{col}_{role}' for col, role in result_pivot.columns]
    result_pivot = result_pivot.reset_index()

    result_final = df_p[['id']].merge(result_pivot, on='id', how='left')
    
    for col in ['num_lesionados_local', 'num_lesionados_visitante']:
        result_final[col] = result_final.get(col, 0).fillna(0).astype(int)
        
    for col in ['coste_lesionados_local', 'coste_lesionados_visitante']:
        result_final[col] = result_final.get(col, 0.0).fillna(0)

    return result_final[[
        'id', 'num_lesionados_local', 'num_lesionados_visitante',
        'coste_lesionados_local', 'coste_lesionados_visitante'
    ]]
