import pandas as pd

# ==========================================
# 1. DICCIONARIO DE TRADUCCIÓN DE EQUIPOS
# ==========================================
# A la izquierda: El nombre exacto que viene en Transfermarkt (fuente externa).
# A la derecha: El nombre exacto que usa el 'train.csv'.
#
# IMPORTANTE: Debe cubrir TODOS los equipos presentes en train.csv.
# Los nombres de train.csv son: La Liga española + Premier League inglesa.

MAPEO_EQUIPOS = {
    # ── La Liga (España) ──────────────────────────────────
    "Real Madrid":             "Real Madrid",      # Coincide exacto en performances
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
    "RCD Espanyol":            "Espanol",          # Variante antigua
    "RCD Mallorca":            "Mallorca",
    "Getafe CF":               "Getafe",
    "Deportivo Alavés":        "Alaves",
    "CD Alavés":               "Alaves",           # Variante
    "Levante UD":              "Levante",
    "UD Almería":              "Almeria",
    "Girona FC":               "Girona",
    "Rayo Vallecano":          "Vallecano",
    "UD Las Palmas":           "Las Palmas",
    "Real Valladolid CF":      "Valladolid",
    "Real Valladolid Deportivo": "Valladolid",     # Variante
    "SD Eibar":                "Eibar",
    "CD Leganés":              "Leganes",
    "Granada CF":              "Granada",
    "Cádiz CF":                "Cadiz",
    "Elche CF":                "Elche",
    "SD Huesca":               "Huesca",
    "Málaga CF":               "Malaga",
    "Real Zaragoza":           "Zaragoza",
    "Deportivo de La Coruña":  "La Coruna",
    "RC Deportivo La Coruña":  "La Coruna",        # Variante
    "RC Celta de Vigo":        "Celta",            # Variante formal
    "Racing de Santander":     "Santander",
    "Racing Santander":        "Santander",         # Variante
    "CD Tenerife":             "Tenerife",
    "Real Oviedo":             "Oviedo",
    "Albacete Balompié":       "Albacete",
    "CD Numancia":             "Numancia",
    "Hércules CF":             "Hercules",
    "Córdoba CF":              "Cordoba",
    "Gimnàstic de Tarragona":  "Gimnastic",
    "Real Sporting de Gijón":  "Sp Gijon",
    "Sporting de Gijón":       "Sp Gijon",         # Variante
    "Sporting Gijón":          "Sp Gijon",         # Otra variante en Transfermarkt
    "Real Murcia CF":          "Murcia",
    "Recreativo de Huelva":    "Recreativo",
    "Recreativo Huelva":       "Recreativo",       # Variante en Transfermarkt
    "Xerez CD":                "Xerez",
    "AD Rayo Vallecano":       "Vallecano",        # Variante con AD
    # ── Premier League (Inglaterra) ───────────────────────
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


def estandarizar_nombres(df, columna_equipo):
    """Traduce nombres de Transfermarkt a nombres del train.csv."""
    df_limpio = df.copy()
    df_limpio[columna_equipo] = df_limpio[columna_equipo].replace(MAPEO_EQUIPOS)
    return df_limpio


def preparar_valor_plantilla_inicio_temporada(ruta_valuations, ruta_rendimiento):
    """
    Calcula el valor de la plantilla al INICIO de la temporada actual.

    Usa la última valoración conocida de cada jugador ANTES del 1 de agosto
    del año en que comienza la temporada. Las ligas empiezan entre el 5 y 17
    de agosto, así que el 1 de agosto garantiza que no se ha jugado ningún
    partido aún → cero data leakage.

    La cobertura es excelente gracias a la gran actualización de valoraciones
    que Transfermarkt publica en junio (~217k registros).

    Limitación conocida: la composición de la plantilla se obtiene del archivo
    de performances (que cubre toda la temporada), por lo que incluye jugadores
    fichados después del inicio. Esto es inevitable sin datos de traspasos.

    Parámetros:
    -----------
    ruta_valuations : str
        Ruta al CSV de valoraciones de mercado (player_market_value.csv).
    ruta_rendimiento : str
        Ruta al CSV de rendimiento de jugadores (player_performances.csv).

    Retorna:
    --------
    DataFrame con columnas: equipo, season_join_key, valor_plantilla_inicio
    """
    df_valuations = pd.read_csv(ruta_valuations)
    df_rendimiento = pd.read_csv(ruta_rendimiento)

    # --- PASO 1: Parsear fechas de valoración ---
    df_valuations['fecha_valoracion'] = pd.to_datetime(df_valuations['date_unix'])

    # --- PASO 2: Obtener composición de plantillas por temporada ---
    df_rendimiento['season'] = df_rendimiento['season_name'].astype(str).str.split('/').str[0]

    def _anio_completo(x):
        if len(x) == 2:
            n = int(x)
            return 2000 + n if n < 80 else 1900 + n
        return int(x)

    df_rendimiento['season'] = df_rendimiento['season'].apply(_anio_completo)
    df_equipos = df_rendimiento[['player_id', 'season', 'team_name']].drop_duplicates()

    # --- PASO 3: Para cada jugador-temporada, obtener su última valoración PRE-temporada ---
    # Cutoff: 1 de agosto del año de inicio de la temporada
    # Las ligas empiezan entre el 5-17 de agosto → el 1 de agosto es seguro
    # Ejemplo: temporada 2022/23 (season=2022) → cutoff = 1 Ago 2022
    resultados = []

    for season_year in df_equipos['season'].unique():
        cutoff = pd.Timestamp(f'{season_year}-08-01')

        # Jugadores de esta temporada y sus equipos
        jugadores_temp = df_equipos[df_equipos['season'] == season_year]

        # Valoraciones antes del cutoff para esos jugadores
        mask = (
            (df_valuations['player_id'].isin(jugadores_temp['player_id'])) &
            (df_valuations['fecha_valoracion'] < cutoff)
        )
        vals_pre = df_valuations[mask]

        if vals_pre.empty:
            continue

        # Última valoración de cada jugador antes del cutoff
        ultima_val = (vals_pre
                      .sort_values('fecha_valoracion')
                      .groupby('player_id')['value']
                      .last()
                      .reset_index())

        # Unir con equipo
        merged = pd.merge(ultima_val, jugadores_temp, on='player_id', how='inner')

        # Sumar por equipo
        por_equipo = merged.groupby('team_name')['value'].sum().reset_index()
        por_equipo['season'] = season_year
        resultados.append(por_equipo)

    df_resultado = pd.concat(resultados, ignore_index=True)

    # --- PASO 4: Preparar clave de unión ---
    # season = año de INICIO, anio_fin_temporada = año de FIN → diferencia de +1
    # Esta vez SÍ es +1 porque estamos usando datos del inicio de la MISMA temporada
    df_resultado['season_join_key'] = df_resultado['season'] + 1

    # Estandarizar nombres
    df_resultado = estandarizar_nombres(df_resultado, 'team_name')

    df_resultado = df_resultado.rename(columns={
        'team_name': 'equipo',
        'value': 'valor_plantilla_inicio'
    })

    return df_resultado[['equipo', 'season_join_key', 'valor_plantilla_inicio']]