"""
Funciones de apoyo para el feature engineering (notebook 03) del TFM.

Parte de `data/processed/mrv_eda.parquet` (salida del EDA, notebook 02) y prepara el dataset para
la modelización (notebook 04 en adelante). Reutiliza `src/eda_utils.py` para mantener el mismo
criterio de depuración en toda la tubería.

Las decisiones de diseño están documentadas en detalle en el notebook 03; aquí solo el código.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from eda_utils import atipicosAmissing, ImputacionCuant

RANDOM_STATE = 1234567  # semilla fija de todo el proyecto

# ---------------------------------------------------------------------------------------------
# 0. Fiabilidad del target tras la imputación independiente de CO2 y combustible en el EDA
# ---------------------------------------------------------------------------------------------

# Variables núcleo y cota física, réplica exacta de la sección 1 y 6 del notebook 02 (necesaria
# para recuperar, a partir del consolidado crudo, qué filas tenían CO2/combustible missing o
# atípico ANTES de la imputación -- esa información ya no está en mrv_eda.parquet, que se entrega
# totalmente imputado).
_ID_VARS_EDA = ["reporting_year", "ship_imo_number", "ship_name", "company_name"]
_NUCLEO_NUM_EDA = [
    "total_co2_emissions_m_tonnes",
    "total_fuel_consumption_m_tonnes",
    "time_spent_at_sea_hours",
]
_HORAS_MAX_FISICO = 366 * 24


def _pre_imputacion_co2_fuel(consolidado_full: pd.DataFrame) -> pd.DataFrame:
    """
    Réplica exacta del tratamiento de atípicos del notebook 02 (selección de variables núcleo,
    cota física de horas, cota física del ratio CO2/combustible; el criterio estadístico
    `atipicosAmissing` NO se aplica al CO2 ni al combustible, ver sección 6b del notebook 02),
    deteniéndose justo antes de la imputación para recuperar el estado real "missing/no missing"
    de CO2 y combustible.

    El CO2 no se imputa en el notebook 02, así que este cruce sirve sobre todo de verificación
    cruzada: confirma desde el
    dato crudo qué filas quedaron sin target fiable y por qué.

    Devuelve un DataFrame con `id_vars` + `co2_pre`/`fuel_pre` (NaN donde el EDA los habría
    imputado; valor real donde no).
    """
    datos = consolidado_full[_ID_VARS_EDA + _NUCLEO_NUM_EDA].copy()

    # Cota física 1: horas en mar por encima de las que tiene el año (no afecta a co2/fuel, se
    # replica solo por fidelidad exacta al notebook 02 -- atipicosAmissing se llama en el mismo
    # orden sobre los 3 núcleo numéricos).
    datos.loc[datos["time_spent_at_sea_hours"] > _HORAS_MAX_FISICO, "time_spent_at_sea_hours"] = np.nan

    # Cota física 2: ratio CO2/combustible fuera de [2,5, 3,3] -> ambas a missing (no se sabe cuál
    # de las dos está mal).
    ratio = datos["total_co2_emissions_m_tonnes"] / datos["total_fuel_consumption_m_tonnes"]
    fuera_de_rango = (datos["total_fuel_consumption_m_tonnes"] > 0) & ((ratio < 2.5) | (ratio > 3.3))
    datos.loc[fuera_de_rango, ["total_co2_emissions_m_tonnes", "total_fuel_consumption_m_tonnes"]] = np.nan

    # Criterio estadístico, igual que el notebook 02 (sección 6b): se aplica
    # SOLO a las variables sin cota física propia. El CO2 y el combustible quedan fuera a
    # propósito -- el criterio es de escala y confunde "buque grande" con "dato erróneo", y sus
    # errores reales ya los captura la cota física del ratio CO2/combustible de arriba.
    for c in _NUCLEO_NUM_EDA:
        if c in ("total_co2_emissions_m_tonnes", "total_fuel_consumption_m_tonnes"):
            continue
        serie, _n = atipicosAmissing(datos[c])
        datos[c] = serie

    # Solo la clave de cruce (año, IMO) + los dos valores pre-imputación: ship_name/company_name
    # ya están en el df que se va a cruzar, no hace falta duplicarlos aquí.
    pre = datos[
        ["reporting_year", "ship_imo_number", "total_co2_emissions_m_tonnes", "total_fuel_consumption_m_tonnes"]
    ].copy()
    return pre.rename(
        columns={
            "total_co2_emissions_m_tonnes": "co2_pre",
            "total_fuel_consumption_m_tonnes": "fuel_pre",
        }
    )


def reconstruir_target_fiable(df: pd.DataFrame, consolidado_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Excluir sin más las filas con el ratio CO2/combustible roto tras la imputación independiente
    del EDA trataría solo el síntoma. Aquí se ataca la causa: se recupera, desde
    `mrv_consolidado.parquet`, qué filas tenían CO2 y/o combustible missing/atípico ANTES de que
    el EDA los imputara de forma independiente, y se distingue:

      - **Ninguno de los dos faltaba**: valor original real, no se toca.
      - **Solo uno de los dos faltaba**: el otro SÍ se conoce -> se reconstruye el que falta a
        partir del que se conoce, usando el factor de emisión físico (mediana del ratio
        CO2/combustible sobre las filas sin tratar, ~3,137 t/t) en vez de un sorteo aleatorio
        independiente. Deterministic, sin depender de ninguna semilla.
      - **Faltaban los dos a la vez**: no hay nada de lo que partir para reconstruir (es la
        inmensa mayoría de los casos -- CO2 y combustible faltan juntos en el ~96% de las
        veces). Estas filas no son fiables como target y se excluyen de los datasets de
        modelización.

    Con este criterio, el nº de filas "fiables" es ligeramente MENOR que con un simple chequeo del
    ratio -- 3.300 filas tenían ambos valores desconocidos, más que las 3.181 que fallaban el
    chequeo, porque algunas imputaciones aleatorias del EDA, aunque fabricadas, caían por azar
    dentro de [2,5, 3,3]. El objetivo no es maximizar el nº de filas, es maximizar la fiabilidad:
    el target de cada fila que se usa para entrenar o evaluar el modelo es, o bien el dato real
    reportado, o bien una reconstrucción físicamente consistente -- nunca un valor fabricado sin
    relación con el resto de la fila.

    **Cuántas filas acaba reconstruyendo, medido sobre el dataset final: cero.**
    `target_reconstruido` es `False` en las 106.521 filas de `mrv_features_*_ml.parquet`. CO2 y
    combustible faltan siempre juntos, así que la rama del "solo falta uno" nunca llega a
    dispararse y el mecanismo funciona como red de seguridad, no como tratamiento.

    Es decir: **ninguna fila con la que se entrena o se evalúa el modelo tiene el target
    fabricado**. Es dato reportado por el armador y verificado, en el 100%
    de los casos, o no está.

    Devuelve [dataframe_actualizado, resumen] donde `resumen` es un dict con los recuentos de cada
    categoría y el factor de emisión usado, para dejar constancia en el notebook.
    """
    consolidado = pd.read_parquet(consolidado_path)
    consolidado_full = consolidado[consolidado["report_coverage"] == "Full"].reset_index(drop=True)
    pre = _pre_imputacion_co2_fuel(consolidado_full)

    # Únicas por (año, IMO) dentro de 'Full' -- verificado: 0 duplicados -- así que este cruce es una clave segura, no depende del orden de filas.
    df = df.merge(pre, on=["reporting_year", "ship_imo_number"], how="left", validate="one_to_one")

    falta_co2 = df["co2_pre"].isna()
    falta_fuel = df["fuel_pre"].isna()
    completo = ~falta_co2 & ~falta_fuel
    solo_falta_co2 = falta_co2 & ~falta_fuel
    solo_falta_fuel = falta_fuel & ~falta_co2
    faltan_ambos = falta_co2 & falta_fuel

    factor_emision = (df.loc[completo, "co2_pre"] / df.loc[completo, "fuel_pre"]).median()

    df["target_reconstruido"] = solo_falta_co2 | solo_falta_fuel
    df.loc[solo_falta_co2, "total_co2_emissions_m_tonnes"] = df.loc[solo_falta_co2, "fuel_pre"] * factor_emision
    df.loc[solo_falta_fuel, "total_fuel_consumption_m_tonnes"] = df.loc[solo_falta_fuel, "co2_pre"] / factor_emision

    resumen = {
        "factor_emision_mediana": float(factor_emision),
        "n_completo": int(completo.sum()),
        "n_reconstruido": int((solo_falta_co2 | solo_falta_fuel).sum()),
        "n_solo_falta_co2": int(solo_falta_co2.sum()),
        "n_solo_falta_fuel": int(solo_falta_fuel.sum()),
        "n_no_fiable": int(faltan_ambos.sum()),
    }

    df_fiable = df.loc[~faltan_ambos].drop(columns=["co2_pre", "fuel_pre"]).copy()
    return df_fiable, resumen


# ---------------------------------------------------------------------------------------------
# 1. Taxonomía de ship_type
# ---------------------------------------------------------------------------------------------

# Subcategorías introducidas por EMSA a partir de 2023-2024 (ver el cuaderno 01, sección 10)
_SUBCATEGORIAS_NUEVAS = {
    "Passenger ship (Cruise Passenger ship)": "Passenger ship",
    "Other ship types (Offshore)": "Other ship types",
}


def agrupar_ship_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el tratamiento híbrido decidido para `ship_type`:
      - `ship_type_agrupado`: subcategorías nuevas fusionadas con su categoría padre, para
        mantener una taxonomía estable en toda la serie 2018-2025 (evita que el modelo confunda
        "categoría rara" con "año reciente", dado que casi no hay datos de estas subcategorías
        antes de 2023-2024).
      - `es_subcategoria_nueva`: flag binario (0/1) que preserva la información de que esa fila
        pertenecía a una subcategoría introducida recientemente, para poder analizar su efecto
        propio en la interpretabilidad (notebook 07) sin distorsionar la variable categórica principal.
    """
    df = df.copy()
    df["es_subcategoria_nueva"] = df["ship_type"].isin(_SUBCATEGORIAS_NUEVAS).astype(int)
    df["ship_type_agrupado"] = df["ship_type"].replace(_SUBCATEGORIAS_NUEVAS)
    return df


# ---------------------------------------------------------------------------------------------
# 2. Atípicos en te_valor
# ---------------------------------------------------------------------------------------------


def tratar_atipicos_te_valor(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    `te_valor` (índice de eficiencia técnica EIV/EEDI/EEXI, en gCO2/t·nm, comparable entre
    métodos: se ha verificado que la unidad es la misma independientemente de `te_metodo`) no
    recibió tratamiento de atípicos en el EDA (notebook 02): solo se imputaron sus missings
    directos. Su percentil 99.9 es ~261 pero su máximo es 208.390 -- unos 800x mayor -- señal
    clara de atípicos estadísticos no tratados todavía.

    Se aplica aquí el mismo criterio ya validado (`atipicosAmissing`, sección de atípicos del
    notebook 02) y se imputa después con la mediana, igual que el resto de numéricas núcleo.
    No existe para esta variable una cota física evidente (a diferencia de horas en mar o del
    ratio CO2/combustible), así que se usa solo el criterio estadístico.
    """
    df = df.copy()
    serie_tratada, n_atipicos = atipicosAmissing(df["te_valor"])
    df["te_valor"] = serie_tratada
    df["te_valor"] = ImputacionCuant(df["te_valor"], "mediana")
    return df, n_atipicos


# ---------------------------------------------------------------------------------------------
# 3. Datasets de modelización: principal (sin combustible) vs control (con combustible)
# ---------------------------------------------------------------------------------------------

ID_COLS = ["reporting_year", "ship_imo_number", "ship_name", "company_name"]
TARGET_COL = "total_co2_emissions_m_tonnes"

# Predictores del modelo principal: variables operativas/estructurales, sin el combustible
# (ver notebook 03, sección de la casi-tautología CO2 = fuel x factor_emision).
PREDICTORES_PRINCIPAL = [
    "time_spent_at_sea_hours",
    "ship_type_agrupado",
    "es_subcategoria_nueva",
    "ice_class",
    "te_metodo",
    "te_valor",
    "prop_missings",
]
# El modelo de control añade el combustible, para usarlo como cota superior de referencia,
# no como modelo principal de negocio.
PREDICTORES_CONTROL = PREDICTORES_PRINCIPAL + ["total_fuel_consumption_m_tonnes"]
# Nivel intermedio: añade las dos variables operativas reconstruidas (distancia navegada y
# velocidad media anual, ver `reconstruir_distancia_velocidad`). No usa el combustible, así que no
# es tautológico; es el conjunto que necesita el simulador de escenarios (notebook 08).
PREDICTORES_OPERACIONAL = PREDICTORES_PRINCIPAL + ["distancia_nm", "velocidad_nudos"]

# Capacidad de carga reconstruida (ver `reconstruir_capacidad`): la medida de tamaño del buque que
# el MRV no publica. Se guardan las cinco unidades del Reglamento 2016/1928 por separado -- no son
# magnitudes comparables entre sí, y meterlas en una sola columna obligaría a mezclar pasajeros con
# toneladas -- más una columna consolidada por prioridad.
CAPACIDAD_COLS = [
    "capacidad_estimada",
    "capacidad_mass",
    "capacidad_volume",
    "capacidad_dwt",
    "capacidad_pax",
    "capacidad_freight",
]
# Nivel `tamano`: el principal más la capacidad de diseño estimada. Es el nivel que mide, por
# diferencia contra `principal`, cuánto costaba no tener el tamaño del buque.
PREDICTORES_TAMANO = PREDICTORES_PRINCIPAL + CAPACIDAD_COLS
# Nivel `operacional_tamano`: añade además la capacidad efectivamente movida cada año, que es una
# variable operativa (lo que el buque hizo) y no un atributo del buque (lo que el buque es). La
# separación entre las dos no es cosmética: `capacidad_estimada` está disponible antes de que el
# buque navegue, `capacidad_utilizada` solo a posteriori.
PREDICTORES_OPERACIONAL_TAMANO = PREDICTORES_OPERACIONAL + CAPACIDAD_COLS + ["capacidad_utilizada"]


def construir_datasets_modelizacion(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Devuelve un dict con dos versiones "raw" (categóricas sin codificar, pensadas para poder
    reutilizarse tal cual en el notebook de deep learning con capas de embeddings) y dos
    versiones "ml" (con codificación one-hot de las categóricas, en el mismo estilo que
    `crear_data_modelo` de `src/FuncionesMineria.py`, listas para regresión/GLM/Random
    Forest/Gradient Boosting del notebook 04):
      - 'principal_raw'   / 'principal_ml'  : variables de buque y tiempo de operación.
      - 'operacional_raw' / 'operacional_ml': añade distancia navegada y velocidad media.
      - 'tamano_raw'      / 'tamano_ml'     : añade la capacidad de carga reconstruida.
      - 'operacional_tamano_raw' / '..._ml' : las dos anteriores juntas, más la capacidad movida.
      - 'control_raw'     / 'control_ml'    : añade `total_fuel_consumption_m_tonnes` (tautológico).
    Todas incluyen las columnas identificadoras (`ID_COLS`), el target, la columna `es_train`
    (ver `dividir_train_test`) y `target_reconstruido` (ver `reconstruir_target_fiable`) para que
    cada notebook posterior pueda reconstruir la misma partición y saber qué filas tienen el
    target físicamente reconstruido en vez de reportado directamente, sin volver a calcular nada.
    """
    columnas_base = ID_COLS + [TARGET_COL, "es_train", "target_reconstruido"]

    principal_raw = df[columnas_base + PREDICTORES_PRINCIPAL].copy()
    operacional_raw = df[columnas_base + PREDICTORES_OPERACIONAL].copy()
    control_raw = df[columnas_base + PREDICTORES_CONTROL].copy()
    tamano_raw = df[columnas_base + PREDICTORES_TAMANO].copy()
    operacional_tamano_raw = df[columnas_base + PREDICTORES_OPERACIONAL_TAMANO].copy()

    # prop_missings se trata como categórica (pocos valores distintos), igual que en el
    # notebook 02 de EDA -- no como numérica ordinal, para no
    # asumir un efecto lineal en la proporción de missing que no se ha verificado.
    categoricas = ["ship_type_agrupado", "ice_class", "te_metodo", "prop_missings"]
    principal_ml = pd.get_dummies(principal_raw, columns=categoricas, drop_first=True)
    operacional_ml = pd.get_dummies(operacional_raw, columns=categoricas, drop_first=True)
    control_ml = pd.get_dummies(control_raw, columns=categoricas, drop_first=True)
    tamano_ml = pd.get_dummies(tamano_raw, columns=categoricas, drop_first=True)
    operacional_tamano_ml = pd.get_dummies(operacional_tamano_raw, columns=categoricas, drop_first=True)

    return {
        "principal_raw": principal_raw,
        "principal_ml": principal_ml,
        "tamano_raw": tamano_raw,
        "tamano_ml": tamano_ml,
        "operacional_raw": operacional_raw,
        "operacional_ml": operacional_ml,
        "operacional_tamano_raw": operacional_tamano_raw,
        "operacional_tamano_ml": operacional_tamano_ml,
        "control_raw": control_raw,
        "control_ml": control_ml,
    }


# ---------------------------------------------------------------------------------------------
# 3b. Variables operativas reconstruidas: distancia navegada y velocidad media
# ---------------------------------------------------------------------------------------------

_RATIO_CO2_DIST = "annual_average_co2_emissions_per_distance_kg_co2_n_mile"
_RATIO_FUEL_DIST = "annual_average_fuel_consumption_per_distance_kg_n_mile"

# EMSA renombró estas dos columnas a partir del informe de 2024, quitándoles el prefijo
# "annual_average_". La taxonomía de tipos de buque ya había cambiado antes entre ediciones (ver
# agrupar_ship_type), así que el patrón no es nuevo en esta fuente.
#
# No contemplarlo produce un error SILENCIOSO y caro: el ratio sale NaN en el 100% de las filas de
# 2024 y 2025, la distancia y la velocidad quedan vacías para esos dos años -- 31.072 filas, el 29%
# del dataset y 297 Mt de CO2 -- y el nivel operacional (el que usa el simulador del notebook 08) se
# queda sin los dos años más recientes sin que ninguna excepción lo avise.
_RATIO_CO2_DIST_2024 = "co2_emissions_per_distance_kg_co2_n_mile"
_RATIO_FUEL_DIST_2024 = "fuel_consumption_per_distance_kg_n_mile"

# Rango plausible de velocidad media anual para un buque mercante, en nudos. Fuera de ahí el
# cociente distancia/horas no describe una operación real (típicamente, buques con muy pocas horas
# en mar reportadas frente a la distancia declarada).
_VELOCIDAD_MIN, _VELOCIDAD_MAX = 3.0, 30.0
_HORAS_MIN_PARA_VELOCIDAD = 24.0


def _combinar_convenciones(base: pd.DataFrame, nombre_antiguo: str, nombre_2024: str) -> pd.Series:
    """
    Une las dos convenciones de nombre de una misma columna del MRV en una sola serie.

    Falla en voz alta si alguna fila trae valor en las dos a la vez: eso significaría que EMSA
    publica ambas y que la suposición de exclusividad -- sobre la que se apoya el `combine_first` --
    ha dejado de ser cierta.
    """
    if nombre_antiguo not in base.columns:
        return base[nombre_2024]
    if nombre_2024 not in base.columns:
        return base[nombre_antiguo]

    solapan = base[nombre_antiguo].notna() & base[nombre_2024].notna()
    if solapan.any():
        raise ValueError(
            f"{int(solapan.sum())} filas traen '{nombre_antiguo}' y '{nombre_2024}' a la vez: "
            "las dos convenciones de nombre ya no son excluyentes, revisa la fuente."
        )
    return base[nombre_antiguo].combine_first(base[nombre_2024])


def reconstruir_distancia_velocidad(df: pd.DataFrame, consolidado_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Reconstruye la distancia navegada (millas náuticas) y la velocidad media anual (nudos), que el
    MRV no publica directamente pero sí de forma implícita en dos intensidades que sí publica.

    Hay dos vías aritméticas independientes para la distancia:

        distancia = CO2 total x 1000 / (kg CO2 por milla)
        distancia = combustible total x 1000 / (kg combustible por milla)

    **Se usa la del combustible**, que no toca la variable objetivo en ningún momento: así la
    variable derivada no depende del target ni siquiera aritméticamente. La vía del CO2 se calcula
    igualmente, pero solo para validarla contra la primera -- coinciden con una discrepancia
    relativa mediana del orden de 10^-5, lo que confirma que no se está estimando nada: se está
    recuperando una magnitud que el armador reportó.

    La velocidad media solo se calcula para buques con al menos `_HORAS_MIN_PARA_VELOCIDAD` horas
    en mar, y se descarta fuera del rango plausible [3, 30] nudos.

    Aviso: la cobertura es del ~68% y no es uniforme por tipo de buque (los
    cruceros y las subcategorías nuevas de EMSA son los peor cubiertos). El valor ausente se deja
    como NaN a propósito, para que cada modelo lo trate como categoría propia en vez de imputarlo.

    Devuelve [dataframe_con_las_dos_columnas_nuevas, resumen].
    """
    consolidado = pd.read_parquet(consolidado_path)
    consolidado = consolidado[consolidado["report_coverage"] == "Full"]
    cols = ["reporting_year", "ship_imo_number", "total_co2_emissions_m_tonnes",
            "total_fuel_consumption_m_tonnes",
            _RATIO_CO2_DIST, _RATIO_FUEL_DIST, _RATIO_CO2_DIST_2024, _RATIO_FUEL_DIST_2024]
    cols = [c for c in cols if c in consolidado.columns]
    base = consolidado[cols].drop_duplicates(subset=["reporting_year", "ship_imo_number"], keep="last").copy()

    # Las dos convenciones de nombre son excluyentes por año (hasta 2023 la antigua, desde 2024 la
    # nueva), así que combinarlas no puede sobrescribir nada. Se comprueba en vez de suponerlo.
    ratio_fuel = _combinar_convenciones(base, _RATIO_FUEL_DIST, _RATIO_FUEL_DIST_2024)
    ratio_co2 = _combinar_convenciones(base, _RATIO_CO2_DIST, _RATIO_CO2_DIST_2024)

    dist_fuel = base["total_fuel_consumption_m_tonnes"] * 1000 / ratio_fuel.replace(0, np.nan)
    dist_co2 = base["total_co2_emissions_m_tonnes"] * 1000 / ratio_co2.replace(0, np.nan)

    ambas = dist_fuel.notna() & dist_co2.notna() & (dist_fuel > 0)
    discrepancia = ((dist_co2 - dist_fuel).abs() / dist_fuel)[ambas]

    base["distancia_nm"] = dist_fuel
    base = base[["reporting_year", "ship_imo_number", "distancia_nm"]]

    df = df.merge(base, on=["reporting_year", "ship_imo_number"], how="left", validate="one_to_one")

    horas = df["time_spent_at_sea_hours"]
    velocidad = np.where(horas >= _HORAS_MIN_PARA_VELOCIDAD, df["distancia_nm"] / horas.replace(0, np.nan), np.nan)
    velocidad = pd.Series(velocidad, index=df.index)
    df["velocidad_nudos"] = velocidad.where(velocidad.between(_VELOCIDAD_MIN, _VELOCIDAD_MAX))

    resumen = {
        "n_con_ambas_vias": int(ambas.sum()),
        "discrepancia_mediana_pct": float(discrepancia.median() * 100),
        "discrepancia_p99_pct": float(discrepancia.quantile(0.99) * 100),
        "pct_discrepancia_menor_1pct": float((discrepancia < 0.01).mean() * 100),
        "cobertura_distancia_pct": float(df["distancia_nm"].notna().mean() * 100),
        "cobertura_velocidad_pct": float(df["velocidad_nudos"].notna().mean() * 100),
        "velocidad_mediana_nudos": float(df["velocidad_nudos"].median()),
        "cobertura_velocidad_por_anio_pct": (
            df.groupby("reporting_year")["velocidad_nudos"]
            .apply(lambda s: 100 * s.notna().mean())
            .round(1)
            .to_dict()
        ),
    }
    return df, resumen



# ---------------------------------------------------------------------------------------------
# 3c. Capacidad de carga reconstruida: el tamaño del buque que el MRV no publica
# ---------------------------------------------------------------------------------------------

# El registro público del MRV no publica ninguna medida de tamaño del buque (arqueo bruto, porte,
# plazas de pasaje). El informe del JRC sobre este mismo portal (JRC128870) señala esa ausencia
# como el límite de lo que se puede analizar con estos datos, y la literatura que trabaja con el
# dataset la resuelve comprando una base comercial (Clarksons) y cruzando por número IMO.
#
# No hace falta. El MRV sí publica, para cada buque-año, la intensidad por milla navegada Y la
# intensidad por trabajo de transporte, y el denominador de la segunda ES la magnitud de capacidad
# que se busca. Dividiendo una entre otra, la magnitud se despeja:
#
#     capacidad = combustible_por_distancia [kg/milla] x 1000 / combustible_por_trabajo [g/(u·milla)]
#
# Comprobación de unidades: (g/milla) / (g/(u·milla)) = u.
#
# Igual que en `reconstruir_distancia_velocidad`, **se usa la vía del combustible**: la vía del CO2
# se calcula solo para validar. Y hay una diferencia importante a favor de esta reconstrucción: la
# distancia se despeja del combustible TOTAL, mientras que la capacidad sale del cociente de dos
# intensidades, donde el consumo total se cancela. La capacidad reconstruida no arrastra por tanto
# ninguna información sobre el nivel de consumo del buque, y por eso puede entrar en el nivel
# `principal` sin acercarlo a la tautología del nivel `control`.
#
# El Reglamento de Ejecución (UE) 2016/1928 asigna a cada tipo de buque un parámetro de carga
# distinto, así que hay cinco denominadores posibles y NO son intercambiables entre sí: toneladas,
# metros cúbicos, porte transportado, pasajeros y toneladas de carga rodada. Se reconstruyen los
# cinco por separado, cada uno en su propia columna, y además una columna consolidada que elige el
# primero disponible por orden de prioridad.

# unidad -> (nombre hasta 2023, nombre desde 2024) de la intensidad por trabajo de transporte.
# EMSA volvió a quitar el prefijo "annual_average_" en el informe de 2024, exactamente igual que
# hizo con la intensidad por distancia -- lo que, sin tratarlo, vaciaría el 29% del dataset. Por eso
# todas se leen con `_combinar_convenciones`, que falla en voz alta si dejaran de ser excluyentes.
_TRABAJO_FUEL = {
    "mass": ("annual_average_fuel_consumption_per_transport_work_mass_g_m_tonnes_n_miles",
             "fuel_consumption_per_transport_work_mass_g_m_tonnes_n_miles"),
    "volume": ("annual_average_fuel_consumption_per_transport_work_volume_g_m3_n_miles",
               "fuel_consumption_per_transport_work_volume_g_m3_n_miles"),
    "dwt": ("annual_average_fuel_consumption_per_transport_work_dwt_g_dwt_carried_n_miles",
            "fuel_consumption_per_transport_work_dwt_g_dwt_carried_n_miles"),
    "pax": ("annual_average_fuel_consumption_per_transport_work_pax_g_pax_n_miles",
            "fuel_consumption_per_transport_work_pax_g_pax_n_miles"),
    "freight": ("annual_average_fuel_consumption_per_transport_work_freight_g_m_tonnes_n_miles",
                "fuel_consumption_per_transport_work_freight_g_m_tonnes_n_miles"),
}
_TRABAJO_CO2 = {
    "mass": ("annual_average_co2_emissions_per_transport_work_mass_g_co2_m_tonnes_n_miles",
             "co2_emissions_per_transport_work_mass_g_co2_m_tonnes_n_miles"),
    "volume": ("annual_average_co2_emissions_per_transport_work_volume_g_co2_m3_n_miles",
               "co2_emissions_per_transport_work_volume_g_co2_m3_n_miles"),
    "dwt": ("annual_average_co2_emissions_per_transport_work_dwt_g_co2_dwt_carried_n_miles",
            "co2_emissions_per_transport_work_dwt_g_co2_dwt_carried_n_miles"),
    "pax": ("annual_average_co2_emissions_per_transport_work_pax_g_co2_pax_n_miles",
            "co2_emissions_per_transport_work_pax_g_co2_pax_n_miles"),
    "freight": ("annual_average_co2_emissions_per_transport_work_freight_g_co2_m_tonnes_n_miles",
                "co2_emissions_per_transport_work_freight_g_co2_m_tonnes_n_miles"),
}

# Orden de prioridad para la columna consolidada. Va de la unidad más específica del buque a la más
# genérica: un buque que declara pasajeros es un buque de pasaje, y esa es la medida de tamaño que
# lo describe; las toneladas son el último recurso porque las declara casi toda la flota.
_PRIORIDAD_UNIDAD = ["pax", "volume", "dwt", "freight", "mass"]

# Topes físicos por unidad. No son criterios estadísticos: son los límites de lo que existe a flote.
# El buque de pasaje más grande del mundo no llega a 8.000 plazas, ningún mercante supera las
# ~400.000 t de porte (clase Valemax) y el mayor metanero (clase Q-Max) mueve 266.000 m3. Se deja un
# margen sobre esos máximos para no recortar buques reales. Mismo criterio que la cota de
# `_HORAS_MAX_FISICO` y que el rango [3, 30] nudos de la velocidad: donde hay un límite físico
# conocido se usa el límite físico, no el percentil.
_TOPES_CAPACIDAD = {"pax": 9_000, "mass": 460_000, "dwt": 460_000, "volume": 300_000,
                    "freight": 460_000}

# Depuración intra-buque: la capacidad de un buque no cambia de un año para otro, así que una fila
# que se dispara respecto a la mediana histórica de ESE buque es un error de declaración, no un
# buque distinto. El corte en 2,5x separa limpiamente las dos poblaciones (el percentil 99 del
# cociente valor/mediana es 2,0 y el 99,9 es 12,1) y descarta el 0,6% de las filas.
_FACTOR_ATIPICO_INTRA_BUQUE = 2.5
_MIN_ANIOS_PARA_FILTRO_INTRA = 3

# La capacidad declarada varía dentro del mismo buque según lo cargado que navegue cada año, así que
# la mediana mide la carga típica y no el tamaño. El percentil 95 de la serie del buque se acerca
# mucho más al tamaño de diseño: contrastado contra buques identificables por su nombre en el propio
# dataset, el metanero SHAGRA (clase Q-Max, 266.000 m3) sale en 242.267 m3 y el portacontenedores
# CMA CGM JACQUES SAADE (220.766 t de porte) en 209.777 t.
_CUANTIL_CAPACIDAD = 0.95


def reconstruir_capacidad(df: pd.DataFrame, consolidado_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Reconstruye la capacidad de carga del buque -- la medida de tamaño que el MRV no publica -- a
    partir del cociente entre la intensidad por distancia y la intensidad por trabajo de transporte.

    Añade al dataframe:
      - `capacidad_utilizada`  : capacidad movida ese año concreto (variable OPERATIVA, por fila).
      - `capacidad_estimada`   : percentil 95 de la serie histórica del buque (proxy del tamaño de
                                 DISEÑO, constante por buque, atributo del buque).
      - `capacidad_{unidad}`   : lo mismo, separado por cada una de las cinco unidades del
                                 Reglamento 2016/1928, porque no son magnitudes comparables entre sí.
      - `capacidad_unidad`     : qué unidad declara el buque (categórica, para el análisis).

    Aviso metodológico que conviene no saltarse: lo que el MRV permite despejar no es el porte de
    proyecto sino la carga realmente embarcada -- el documento de trabajo de la Comisión define
    "deadweight carried" como el desplazamiento en el puerto de salida menos el peso en rosca. De
    ahí la distinción entre `capacidad_utilizada` (operativa) y `capacidad_estimada` (agregado por
    buque, proxy del tamaño). El agregado se calcula sobre las filas del propio buque, y como la
    partición train/test está agrupada por `ship_imo_number`, ninguna estimación cruza el split.

    Devuelve [dataframe_con_las_columnas_nuevas, resumen].
    """
    consolidado = pd.read_parquet(consolidado_path)
    consolidado = consolidado[consolidado["report_coverage"] == "Full"]
    base = consolidado.drop_duplicates(subset=["reporting_year", "ship_imo_number"], keep="last").copy()

    ratio_fuel_dist = _combinar_convenciones(base, _RATIO_FUEL_DIST, _RATIO_FUEL_DIST_2024)
    ratio_co2_dist = _combinar_convenciones(base, _RATIO_CO2_DIST, _RATIO_CO2_DIST_2024)

    capacidades, discrepancias = {}, []
    for unidad in _TRABAJO_FUEL:
        trabajo_fuel = _combinar_convenciones(base, *_TRABAJO_FUEL[unidad])
        trabajo_co2 = _combinar_convenciones(base, *_TRABAJO_CO2[unidad])

        via_fuel = ratio_fuel_dist * 1000 / trabajo_fuel.replace(0, np.nan)
        via_co2 = ratio_co2_dist * 1000 / trabajo_co2.replace(0, np.nan)

        ambas = via_fuel.notna() & via_co2.notna() & (via_fuel > 0)
        discrepancias.append(((via_co2 - via_fuel).abs() / via_fuel)[ambas])

        capacidades[unidad] = via_fuel.where(via_fuel.between(0, _TOPES_CAPACIDAD[unidad], inclusive="right"))

    discrepancia = pd.concat(discrepancias)

    # Consolidada por fila: la primera unidad disponible según prioridad.
    consolidada = pd.Series(np.nan, index=base.index)
    unidad_elegida = pd.Series(pd.NA, index=base.index, dtype="object")
    for unidad in _PRIORIDAD_UNIDAD:
        pendiente = consolidada.isna() & capacidades[unidad].notna()
        consolidada[pendiente] = capacidades[unidad][pendiente]
        unidad_elegida[pendiente] = unidad

    base["capacidad_utilizada"] = consolidada
    base["capacidad_unidad"] = unidad_elegida

    def _agregado_por_buque(serie: pd.Series) -> pd.Series:
        """Percentil 95 de la serie del buque, tras descartar los años atípicos de ese mismo buque."""
        tabla = pd.DataFrame({"imo": base["ship_imo_number"], "valor": serie}).dropna()
        if tabla.empty:
            return pd.Series(dtype=float)
        grupo = tabla.groupby("imo")["valor"]
        mediana, n_anios = grupo.transform("median"), grupo.transform("size")
        atipica = (n_anios >= _MIN_ANIOS_PARA_FILTRO_INTRA) & (
            tabla["valor"] / mediana.replace(0, np.nan) > _FACTOR_ATIPICO_INTRA_BUQUE)
        return tabla[~atipica].groupby("imo")["valor"].quantile(_CUANTIL_CAPACIDAD)

    por_buque = {"capacidad_estimada": _agregado_por_buque(consolidada)}
    for unidad in _TRABAJO_FUEL:
        por_buque[f"capacidad_{unidad}"] = _agregado_por_buque(capacidades[unidad])

    base = base[["reporting_year", "ship_imo_number", "capacidad_utilizada", "capacidad_unidad"]]
    df = df.merge(base, on=["reporting_year", "ship_imo_number"], how="left", validate="one_to_one")
    for nombre, serie in por_buque.items():
        df[nombre] = df["ship_imo_number"].map(serie)

    resumen = {
        "n_con_ambas_vias": int(discrepancia.notna().sum()),
        "discrepancia_mediana_pct": float(discrepancia.median() * 100),
        "discrepancia_p99_pct": float(discrepancia.quantile(0.99) * 100),
        "pct_discrepancia_menor_1pct": float((discrepancia < 0.01).mean() * 100),
        "cobertura_capacidad_estimada_pct": float(df["capacidad_estimada"].notna().mean() * 100),
        "cobertura_capacidad_utilizada_pct": float(df["capacidad_utilizada"].notna().mean() * 100),
        "buques_con_capacidad": int(df.loc[df["capacidad_estimada"].notna(), "ship_imo_number"].nunique()),
        "cobertura_por_unidad_pct": {
            u: round(float(df[f"capacidad_{u}"].notna().mean() * 100), 2) for u in _TRABAJO_FUEL},
        "reparto_unidad": df["capacidad_unidad"].value_counts(dropna=False).to_dict(),
    }
    return df, resumen

# ---------------------------------------------------------------------------------------------
# 4. Partición train/test agrupada por buque
# ---------------------------------------------------------------------------------------------


def dividir_train_test(df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """
    Partición train/test agrupada por `ship_imo_number`: el mismo buque nunca aparece a la vez
    en train y en test, aunque tenga varias filas (una por año). Un split aleatorio por fila
    dejaría "fugar" el nivel base de emisiones de un buque de train a test (el modelo podría
    reconocer el buque en vez de generalizar el patrón). Así, la serie temporal completa de cada
    buque queda entera en un único lado del split.
    """
    df = df.copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    idx_train, idx_test = next(splitter.split(df, groups=df["ship_imo_number"]))
    df["es_train"] = False
    df.iloc[idx_train, df.columns.get_loc("es_train")] = True
    return df
