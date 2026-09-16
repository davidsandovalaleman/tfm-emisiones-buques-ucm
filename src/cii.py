"""Calificación CII (Carbon Intensity Indicator) sobre el registro MRV.

El CII es la calificación operativa A-E que la OMI aplica desde 2023 a los buques de 5.000 GT o
más -- exactamente el mismo umbral que el registro MRV, así que la población de este proyecto ES la
población regulada--.

    CII_attained = CO2 anual [g] / (capacidad x distancia navegada [milla])
    CII_ref(2019) = a x capacidad^(-c)                       (MEPC.353(78))
    CII_required(anio) = (1 - Z(anio)/100) x CII_ref          (MEPC.400(83))
    banda A-E = donde cae CII_attained frente a CII_required  (MEPC.354(78))

**Por que este modulo solo es posible en este proyecto**: el denominador del CII es la capacidad del
buque, y el MRV no la publica -- es justamente el dato que el notebook 03 reconstruye despejandolo del
cociente de dos intensidades--. Sin `capacidad_estimada` no hay calificacion posible.

Tres limitaciones estructurales, declaradas aqui y no en una nota al pie:

1. **La capacidad es un proxy por debajo del peso muerto de diseno.** `capacidad_estimada` es el
   percentil 95 de la serie historica de capacidad *movida* por el buque; un buque que nunca navega
   completamente cargado da un proxy inferior a su DWT real. Como el denominador queda corto, el
   CII sale ALTO y la calificacion, PESIMISTA. El sesgo es sistematico dentro de cada tipo, asi que
   afecta a la banda absoluta pero se cancela en gran medida en las comparaciones entre buques del
   mismo tipo. `factor_capacidad` permite calibrarlo cuando exista validacion externa.
2. **La distancia es la del perimetro MRV**, no la global del buque. El CII oficial usa la
   distancia anual mundial; aqui se usa la de los viajes que tocan el EEE. Al ser un cociente, el
   proxy es razonable para buques con trafico mayoritariamente europeo y peor para el resto.
3. **Solo se califican los tipos cuyo CII usa DWT.** Ro-ro, ro-pax, crucero y car carrier usan GT,
   que el MRV no publica y que este proyecto no reconstruye.

Por las tres razones, lo que calcula este modulo es un **CII-proxy**: sirve para ordenar, comparar y
detectar desacuerdos con el modelo, no para sustituir a la calificacion oficial de un buque.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------
# Tablas normativas. Transcritas de las resoluciones; no tocar sin citar la fuente.
# --------------------------------------------------------------------------------------------

# MEPC.353(78), tabla 1: CII_ref = a x capacidad^(-c).
# Cada entrada: (limite superior del tramo de capacidad, a, c). `np.inf` cierra el ultimo tramo.
# `tope`: valor maximo de capacidad que la resolucion permite usar en la formula.
LINEAS_REFERENCIA: dict[str, dict] = {
    "bulk":        {"metrica": "DWT", "tope": 279_000, "tramos": [(np.inf, 4745, 0.622)]},
    "gas":         {"metrica": "DWT", "tope": None,    "tramos": [(65_000, 8104, 0.639),
                                                                  (np.inf, 14405e7, 2.071)]},
    "tanker":      {"metrica": "DWT", "tope": None,    "tramos": [(np.inf, 5247, 0.610)]},
    "container":   {"metrica": "DWT", "tope": None,    "tramos": [(np.inf, 1984, 0.489)]},
    "gencargo":    {"metrica": "DWT", "tope": None,    "tramos": [(20_000, 588, 0.3885),
                                                                  (np.inf, 31948, 0.792)]},
    "reefer":      {"metrica": "DWT", "tope": None,    "tramos": [(np.inf, 4600, 0.557)]},
    "combination": {"metrica": "DWT", "tope": None,    "tramos": [(np.inf, 5119, 0.622)]},
    "lng":         {"metrica": "DWT", "tope": None,    "tramos": [(65_000, 14779e10, 2.673),
                                                                  (100_000, 14479e10, 2.673),
                                                                  (np.inf, 9.827, 0.000)]},
}

# MEPC.354(78), tabla 1: vectores dd ya exponenciados. Fronteras = CII_required x exp(d_i).
#   attained < d1 -> A ; < d2 -> B ; < d3 -> C ; < d4 -> D ; resto -> E
VECTORES_DD: dict[str, tuple[float, float, float, float]] = {
    "bulk":        (0.86, 0.94, 1.06, 1.18),
    "gas_ge65k":   (0.81, 0.91, 1.12, 1.44),
    "gas_lt65k":   (0.85, 0.95, 1.06, 1.25),
    "tanker":      (0.82, 0.93, 1.08, 1.28),
    "container":   (0.83, 0.94, 1.07, 1.19),
    "gencargo":    (0.83, 0.94, 1.06, 1.19),
    "reefer":      (0.78, 0.91, 1.07, 1.20),
    "combination": (0.87, 0.96, 1.06, 1.14),
    "lng_ge100k":  (0.89, 0.98, 1.06, 1.13),
    "lng_lt100k":  (0.78, 0.92, 1.10, 1.37),
}

# MEPC.400(83), abril de 2025: reduccion anual del CII requerido respecto a la linea base de 2019.
FACTORES_REDUCCION: dict[int, float] = {
    2019: 0.0, 2020: 1.0, 2021: 2.0, 2022: 3.0,
    2023: 5.0, 2024: 7.0, 2025: 9.0, 2026: 11.0,
    2027: 13.625, 2028: 16.25, 2029: 18.875, 2030: 21.5,
}

# Tipos del MRV agrupados por este proyecto -> tipo del CII. Los que usan GT quedan fuera
# (ro-ro, ro-pax, crucero, car carrier): el MRV no publica el arqueo bruto.
MAPEO_TIPOS: dict[str, str] = {
    "Bulk carrier": "bulk",
    "Container ship": "container",
    "Oil tanker": "tanker",
    "Chemical tanker": "tanker",
    "General cargo ship": "gencargo",
    "Gas carrier": "gas",
    "LNG carrier": "lng",
    "Refrigerated cargo carrier": "reefer",
    "Combination carrier": "combination",
}

BANDAS = ["A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------------------------
# Calculo
# --------------------------------------------------------------------------------------------

def _tramo(tipo: str, capacidad: float) -> tuple[float, float]:
    """Devuelve (a, c) del tramo de capacidad que corresponde al buque."""
    for limite, a, c in LINEAS_REFERENCIA[tipo]["tramos"]:
        if capacidad < limite:
            return a, c
    raise ValueError(f"tramo no encontrado para {tipo} con capacidad {capacidad}")


def _clave_dd(tipo: str, capacidad: float) -> str:
    if tipo == "gas":
        return "gas_ge65k" if capacidad >= 65_000 else "gas_lt65k"
    if tipo == "lng":
        return "lng_ge100k" if capacidad >= 100_000 else "lng_lt100k"
    return tipo


def cii_referencia(tipo: str, capacidad: float) -> float:
    """CII de referencia de 2019 para un buque: `a x capacidad^(-c)` (MEPC.353(78))."""
    tope = LINEAS_REFERENCIA[tipo]["tope"]
    cap = min(capacidad, tope) if tope else capacidad
    a, c = _tramo(tipo, cap)
    return a * cap ** (-c)


def cii_requerido(tipo: str, capacidad: float, anio: int) -> float:
    """CII exigido en `anio`: la referencia de 2019 menos el factor de reduccion del ano."""
    z = FACTORES_REDUCCION.get(anio)
    if z is None:
        raise ValueError(f"sin factor de reduccion publicado para {anio}")
    return (1 - z / 100) * cii_referencia(tipo, capacidad)


def banda(attained: float, requerido: float, tipo: str, capacidad: float) -> str:
    """Calificacion A-E comparando el CII obtenido con las cuatro fronteras del tipo."""
    d1, d2, d3, d4 = VECTORES_DD[_clave_dd(tipo, capacidad)]
    for umbral, letra in zip((d1, d2, d3, d4), BANDAS[:4]):
        if attained < requerido * umbral:
            return letra
    return "E"


def calcular(
    df: pd.DataFrame,
    factor_capacidad: float | dict[str, float] = 1.0,
    col_co2: str = "total_co2_emissions_m_tonnes",
    col_distancia: str = "distancia_nm",
    col_tipo: str = "ship_type_agrupado",
) -> pd.DataFrame:
    """Calcula el CII-proxy y la banda A-E de cada fila buque-ano que sea calificable.

    `factor_capacidad` multiplica la capacidad reconstruida antes de entrar en la formula. Es el
    punto de calibracion frente a una validacion externa del peso muerto: 1,0 significa "uso el
    proxy tal cual" y cualquier valor mayor corrige el sesgo pesimista descrito en la cabecera del
    modulo. Acepta un escalar o un diccionario por tipo del CII.
    """
    d = df.copy()
    d["cii_tipo"] = d[col_tipo].map(MAPEO_TIPOS)
    d["cii_capacidad"] = d["capacidad_dwt"].fillna(d["capacidad_mass"])
    d["cii_capacidad_fuente"] = np.where(
        d["capacidad_dwt"].notna(), "dwt_declarado",
        np.where(d["capacidad_mass"].notna(), "masa_carga", None))

    if isinstance(factor_capacidad, dict):
        d["cii_capacidad"] *= d["cii_tipo"].map(factor_capacidad).fillna(1.0)
    else:
        d["cii_capacidad"] *= factor_capacidad

    calificable = (
        d["cii_tipo"].notna()
        & d["cii_capacidad"].notna() & (d["cii_capacidad"] > 0)
        & d[col_distancia].notna() & (d[col_distancia] > 0)
        & d[col_co2].notna() & (d[col_co2] > 0)
        & d["reporting_year"].isin(FACTORES_REDUCCION)
    )
    d["cii_calificable"] = calificable

    # g CO2 / (unidad de capacidad x milla). El CO2 del MRV viene en toneladas metricas.
    d.loc[calificable, "cii_attained"] = (
        d.loc[calificable, col_co2] * 1e6
        / (d.loc[calificable, "cii_capacidad"] * d.loc[calificable, col_distancia])
    )

    sub = d.loc[calificable]
    d.loc[calificable, "cii_referencia"] = [
        cii_referencia(t, c) for t, c in zip(sub["cii_tipo"], sub["cii_capacidad"])]
    d.loc[calificable, "cii_requerido"] = [
        cii_requerido(t, c, int(y))
        for t, c, y in zip(sub["cii_tipo"], sub["cii_capacidad"], sub["reporting_year"])]
    sub = d.loc[calificable]
    d.loc[calificable, "cii_banda"] = [
        banda(at, req, t, c) for at, req, t, c
        in zip(sub["cii_attained"], sub["cii_requerido"], sub["cii_tipo"], sub["cii_capacidad"])]
    # Cociente sobre el exigido: <1 cumple con holgura, >1 no cumple. Comparable entre tipos.
    d.loc[calificable, "cii_ratio"] = d["cii_attained"] / d["cii_requerido"]
    return d


# --------------------------------------------------------------------------------------------
# Calibracion del proxy de capacidad
# --------------------------------------------------------------------------------------------

# Tipos cuya calificacion no es fiable con datos MRV, y por que.
#  - `lng`  : los metaneros declaran su trabajo de transporte en volumen (m3), no en masa, asi que
#             apenas hay capacidad en unidades de peso muerto; ademas su linea de referencia tiene
#             c = 2,673, tan empinada que cualquier error de capacidad se amplifica al cubo.
#  - `combination`: 72 filas en todo el registro. Es ruido.
TIPOS_NO_FIABLES = {"lng", "combination"}


def calibrar(df: pd.DataFrame, anio_base: int = 2019, min_filas: int = 100) -> dict[str, float]:
    """Estima, por tipo, cuanto infraestima la capacidad reconstruida el peso muerto de diseno.

    Las lineas de referencia de MEPC.353(78) se ajustaron sobre la flota de 2019, asi que **el
    buque mediano de 2019 deberia caer sobre su linea de referencia**: mediana de
    `attained/requerido` igual a 1. Si sale por encima es porque el denominador -- la capacidad --
    va corto, y cuanto va corto se despeja de la propia formula:

        attained / requerido  proporcional a  capacidad^(c-1)
        =>  k = mediana(ratio) ^ (1 / (1 - c))

    El inverso de `k` es la **utilizacion media del peso muerto** de ese tipo de buque, que es una
    magnitud fisica contrastable: sale ~77% en graneleros, ~74% en petroleros (viajes en lastre) y
    ~57% en portacontenedores (que agotan el volumen antes que el peso). Que esas tres cifras
    salgan en el orden correcto, de una formula normativa que no sabe nada de este proyecto, es una
    validacion indirecta de la reconstruccion de la capacidad.

    Devuelve `{tipo: k}` para los tipos con muestra suficiente y fiable.
    """
    d = calcular(df)
    base = d[d["cii_calificable"] & (d["reporting_year"] == anio_base)]
    factores: dict[str, float] = {}
    for tipo, g in base.groupby("cii_tipo"):
        if tipo in TIPOS_NO_FIABLES or len(g) < min_filas:
            continue
        _, c = _tramo(tipo, g["cii_capacidad"].median())
        if c >= 1:
            continue
        factores[tipo] = float(g["cii_ratio"].median() ** (1 / (1 - c)))
    return factores


def marcar_poblacion_regulada(df: pd.DataFrame) -> pd.Series:
    """Marca las filas cuyo buque estaba ya en el MRV antes de la ampliacion de ambito de 2025.

    El Reglamento (UE) 2023/957 amplio el MRV desde el 1 de enero de 2025 a los buques de carga
    general de 400 a 5.000 GT y a los buques offshore de 400 GT o mas. En este dataset eso se ve
    como **3.274 buques que aparecen por primera vez en 2025** (el 19,3% de las filas del ano y el
    9,5% de sus emisiones), con una capacidad mediana de 3.622 frente a 30.652 de los veteranos.

    Dos consecuencias, y ninguna es cosmetica:

    1. **El CII no les aplica**: la regla 28 del anexo VI de MARPOL cubre buques de 5.000 GT o mas.
       Calificarlos mezcla dos poblaciones distintas.
    2. **El ejercicio 2025 no es comparable con los anteriores** en ninguna serie por buque. Toda
       comparacion interanual debe hacerse sobre esta poblacion o declarar la ruptura.
    """
    veteranos = set(df.loc[df["reporting_year"] <= 2024, "ship_imo_number"])
    return df["ship_imo_number"].isin(veteranos)
