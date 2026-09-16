"""Coste de cumplimiento del EU ETS marítimo sobre el registro MRV.

Desde 2024 el transporte marítimo entra en el régimen de comercio de derechos de emisión de la UE.
Este módulo traduce las emisiones del registro a **euros de derechos**, que es la unidad en la que
el sector decide, y engancha esa traducción al simulador de escenarios (notebook 08).

Las cuatro reglas del régimen, y dónde sale cada una del dato:

1. **Ámbito geográfico**: 100% de los viajes entre puertos del EEE y de las emisiones en puerto, y
   **50%** de los viajes con un extremo fuera. No hay que reconstruirlo: desde el ejercicio 2024 el
   propio MRV publica la columna `co2_emissions_to_be_reported_under_directive_2003_87_ec`, que es
   la emisión dentro del ámbito ya calculada y **verificada**. `reconstruir_ambito` reproduce la
   regla a mano para los ejercicios anteriores y como control de la columna declarada.
2. **Introducción gradual**, sobre el año de emisión: se entrega el **40%** de lo emitido en 2024,
   el **70%** de 2025 y el **100%** desde 2026.
3. **Gases**: solo CO2 en 2024 y 2025; desde **2026** entran también CH4 y N2O, así que la base
   pasa a ser el CO2 equivalente.
4. **Buques**: carga y pasaje de 5.000 GT o más desde 2024; los buques offshore no entran hasta
   2027. Es un subconjunto del MRV, que desde 2025 llega a los 400 GT.

**El precio no es un dato del proyecto**: es una cotización de mercado que cambia cada día. Por eso
entra como parámetro explícito con su fecha y su fuente, nunca escondido en una constante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Cotización del derecho de emisión (EUA). Es un parámetro de mercado, no un resultado: se cita con
# fecha y fuente, y cualquier cifra de coste publicada debe llevarlas al lado.
PRECIO_EUA_POR_DEFECTO = 82.42          # EUR/t CO2eq
PRECIO_EUA_FECHA = "2026-08-27"
PRECIO_EUA_FUENTE = "EU Carbon Permits, cierre del 27/08/2026"

# Porcentaje de las emisiones verificadas de cada AÑO DE EMISIÓN que hay que cubrir con derechos.
FACTOR_ENTREGA: dict[int, float] = {2024: 0.40, 2025: 0.70}
FACTOR_ENTREGA_PLENO = 1.00             # desde 2026

# Año a partir del cual CH4 y N2O entran en el régimen y la base pasa a CO2 equivalente.
ANIO_CO2EQ = 2026

COL_TOTAL = "total_co2_emissions_m_tonnes"
COL_ETS_CO2 = "co2_emissions_to_be_reported_under_directive_2003_87_ec_m_tonnes"
COL_ETS_CO2EQ = "co2eq_emissions_to_be_reported_under_directive_2003_87_ec_m_tonnes"
COL_INTRA = "co2_emissions_from_all_voyages_between_ports_under_a_ms_jurisdiction_m_tonnes"
COL_SALIDA = "co2_emissions_from_all_voyages_which_departed_from_ports_under_a_ms_jurisdiction_m_tonnes"
COL_LLEGADA = "co2_emissions_from_all_voyages_to_ports_under_a_ms_jurisdiction_m_tonnes"
COL_PUERTO = "co2_emissions_which_occurred_within_ports_under_a_ms_jurisdiction_m_tonnes"
COL_ATRAQUE = "co2_emissions_which_occurred_within_ports_under_a_ms_jurisdiction_at_berth_m_tonnes"


def factor_entrega(anio: int) -> float:
    """Fracción de las emisiones de `anio` que hay que cubrir con derechos."""
    return FACTOR_ENTREGA.get(int(anio), FACTOR_ENTREGA_PLENO)


def base_gases(anio: int) -> str:
    """Qué gas define la base imponible del año: solo CO2 hasta 2025, CO2 equivalente desde 2026."""
    return "co2eq" if int(anio) >= ANIO_CO2EQ else "co2"


def reconstruir_ambito(df: pd.DataFrame) -> pd.Series:
    """Aplica la regla 100%/50% a mano, sin usar la columna declarada.

    Sirve para dos cosas: estimar el ámbito de los ejercicios 2018-2023, en los que la columna aún
    no existía, y **controlar** la columna declarada en 2024-2025. Sobre 2025 reproduce el valor
    declarado exactamente en tres de cada cuatro buques; el resto se explica por derogaciones
    (regiones ultraperiféricas, islas pequeñas, contratos de servicio público) y por errores de
    reporte que `auditar` señala.
    """
    puerto = df[COL_PUERTO] if COL_PUERTO in df else pd.Series(np.nan, index=df.index)
    puerto = puerto.fillna(df[COL_ATRAQUE]) if COL_ATRAQUE in df else puerto
    return (df[COL_INTRA].fillna(0)
            + puerto.fillna(0)
            + 0.5 * (df[COL_SALIDA].fillna(0) + df[COL_LLEGADA].fillna(0)))


def emisiones_en_ambito(df: pd.DataFrame, anio_regimen: int | None = None) -> pd.Series:
    """Emisiones dentro del ámbito del ETS, en toneladas.

    Usa la columna declarada cuando existe y la reconstrucción cuando no. `anio_regimen` decide si
    la base es CO2 o CO2 equivalente; si se omite, se toma el año de cada fila (es decir, el régimen
    que de verdad le aplicaba).
    """
    anio = pd.Series(anio_regimen, index=df.index) if anio_regimen is not None else df["reporting_year"]
    usa_eq = anio.astype(int) >= ANIO_CO2EQ

    declarada = pd.Series(np.nan, index=df.index, dtype="float64")
    if COL_ETS_CO2 in df:
        declarada = df[COL_ETS_CO2].astype("float64")
    if COL_ETS_CO2EQ in df:
        declarada = declarada.mask(usa_eq & df[COL_ETS_CO2EQ].notna(), df[COL_ETS_CO2EQ])
    return declarada.fillna(reconstruir_ambito(df))


# Tolerancia con la que se acepta un factor de ámbito ligeramente por encima de 1. Es la misma que
# usa `auditar` para decidir si el ámbito declarado supera de verdad al total, y por la misma razón:
# EMSA publica las emisiones redondeadas, así que un buque cuya operación es íntegramente intra-EEE
# --factor real 1,000-- sale del cociente en 1,0000000001 o en 1,0008 según cómo caiga el redondeo.
TOLERANCIA_AMBITO = 1.001


def factor_ambito(df: pd.DataFrame) -> pd.Series:
    """Fracción de las emisiones totales del buque que cae dentro del ámbito del ETS.

    Es la pieza que conecta el régimen con el simulador: el modelo predice la emisión **total** de
    un buque, y solo esta fracción de un ahorro se convierte en derechos que no hay que comprar.
    En 2025 su mediana es 0,56 y su agregado 0,585, no 1: usar el ahorro total como si fuera
    ahorro de derechos sobrestimaria el resultado en un 71%.

    **Tolerancia de redondeo.** En el ejercicio 2025 hay **70 buques** cuyo cociente cae entre
    1,0000000001 y 1,0008 por el redondeo de la fuente: operan solo dentro del EEE y su factor real
    es 1,000. Un filtro `f <= 1` estricto los descartaría y, al imputarles la mediana de la flota
    (0,552), a un buque con el 100% de sus emisiones en ámbito se le aplicaría el 55%. Por eso se
    acepta hasta `TOLERANCIA_AMBITO` y se recorta a 1,0, que es el valor físicamente correcto; por
    encima de esa tolerancia el resultado es `NaN`, porque entonces sí es una incoherencia del dato
    y la señala `auditar`.
    """
    f = emisiones_en_ambito(df) / df[COL_TOTAL]
    valido = np.isfinite(f) & (f >= 0) & (f <= TOLERANCIA_AMBITO)
    return f.where(valido).clip(upper=1.0)


def uplift_co2eq(df: pd.DataFrame) -> float:
    """Cuánto sube la base imponible al entrar CH4 y N2O, medido sobre los totales del propio año.

    **Por qué hace falta y no basta la columna declarada.** Desde 2026 la base del ETS es el CO2
    equivalente, y el MRV publica `co2eq_emissions_to_be_reported_under_directive_2003_87_ec`. En
    el ejercicio **2024** esa columna funciona: va un +2,43% por encima de la de CO2 y solo coincide
    con ella en el 31,5% de las filas. En el ejercicio **2025 está mal publicada**: coincide con la
    de CO2 en el **100%** de las filas, es decir, se rellenó copiando el CO2, mientras que el total
    de CO2eq del mismo fichero sí va un +3,07% por encima del total de CO2. La columna de ámbito en
    CO2eq de 2025, por tanto, **no puede usarse**.

    La salida limpia es no usarla: el *uplift* se mide sobre los totales, que sí son coherentes, y
    se aplica al ámbito de CO2. Es una hipótesis explícita -- que la mezcla de gases dentro del
    ámbito es la misma que fuera de él-- y en 2024, donde se puede comprobar, se cumple con holgura
    (+2,43% dentro frente a +2,74% en el total).
    """
    total_co2 = df[COL_TOTAL].sum()
    total_eq = df["total_co2eq_emissions_m_tonnes"].sum()
    if not total_co2 or not np.isfinite(total_eq) or total_eq <= 0:
        return 1.0
    return float(total_eq / total_co2)


def base_desde_ambito(emisiones_ambito_t, anio_regimen: int, uplift: float = 1.0):
    """Base imponible a partir de una emisión **ya medida dentro del ámbito**.

    Existe para que la regla «desde 2026 la base es CO2 equivalente» viva en un solo sitio. La usan
    `base_imponible` (que mide el ámbito sobre el registro) y la API (que parte de un ahorro ya
    multiplicado por el factor de ámbito), de modo que ninguna de las dos vuelve a escribir el
    condicional del año a mano.
    """
    return emisiones_ambito_t * (uplift if int(anio_regimen) >= ANIO_CO2EQ else 1.0)


def base_imponible(df: pd.DataFrame, anio_regimen: int) -> pd.Series:
    """Toneladas sobre las que se calculan los derechos bajo el régimen de `anio_regimen`.

    Hasta 2025 es el CO2 dentro del ámbito. Desde 2026 se le aplica el *uplift* de CO2 equivalente
    de `uplift_co2eq`, en vez de la columna declarada, por el defecto de publicación que esa función
    documenta.

    El `anio_regimen=2025` de la llamada interna no es un descuido: fuerza a leer la columna de
    ámbito en **CO2**, que es la única fiable, y el recargo de CO2eq se aplica después con
    `base_desde_ambito`.
    """
    ambito_co2 = emisiones_en_ambito(df, anio_regimen=2025)
    if int(anio_regimen) < ANIO_CO2EQ:
        return ambito_co2
    return base_desde_ambito(ambito_co2, anio_regimen, uplift_co2eq(df))


def coste(
    emisiones_ambito_t: pd.Series | np.ndarray | float,
    anio_regimen: int,
    precio: float = PRECIO_EUA_POR_DEFECTO,
) -> pd.Series | np.ndarray | float:
    """Coste en euros de los derechos: emisiones en ámbito x factor de entrega del año x precio."""
    return emisiones_ambito_t * factor_entrega(anio_regimen) * precio


def factura(df: pd.DataFrame, anio_regimen: int,
            precio: float = PRECIO_EUA_POR_DEFECTO) -> pd.Series:
    """Euros de derechos que le corresponden a **cada fila** bajo el régimen de `anio_regimen`.

    Es `base_imponible` x `factor_entrega` x precio, y existe para que no haya dos caminos hacia la
    misma cifra: multiplicar la emisión total por `factor_ambito` sin el recargo de CO2eq daría
    7.003 millones de euros donde la base correcta da 7.462. **Toda cifra de factura del proyecto
    sale de aquí**;
    lo único que puede diferir entre dos agregados es la población sobre la que se suman, y eso se
    declara en `reports/12_ets_conciliacion.csv`.
    """
    return base_imponible(df, anio_regimen) * factor_entrega(anio_regimen) * precio


def auditar(df: pd.DataFrame) -> pd.DataFrame:
    """Marca incoherencias aritméticas del propio registro alrededor del ámbito ETS.

    No son sospechas: son identidades que el dato declarado debe cumplir y no cumple.

      * `ets_mayor_que_total` : la emisión dentro del ámbito supera a la emisión total del buque.
      * `componentes_descuadran` : los cuatro desgloses geograficos no suman el total (tolerancia 1%).
    """
    fuera = pd.DataFrame(index=df.index)
    if COL_ETS_CO2 in df:
        fuera["ets_mayor_que_total"] = df[COL_ETS_CO2] > df[COL_TOTAL] * 1.001
    componentes = (df[COL_INTRA].fillna(0) + df[COL_SALIDA].fillna(0)
                   + df[COL_LLEGADA].fillna(0) + df[COL_ATRAQUE].fillna(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        desajuste = (componentes - df[COL_TOTAL]) / df[COL_TOTAL]
    fuera["desajuste_componentes"] = desajuste
    fuera["componentes_descuadran"] = desajuste.abs() > 0.01
    return fuera


def solo_informes_completos(df: pd.DataFrame) -> pd.DataFrame:
    """Se queda con los informes anuales completos, como hace `feature_engineering`.

    Desde 2024 el MRV publica también informes **parciales** (art. 11(2)): un buque que cambia de
    compañía a mitad de año declara dos tramos. Sumar completos y parciales duplica emisiones -- en
    2025 son 1.512 filas parciales sobre 18.472--, así que toda cifra agregada de este módulo se
    calcula sobre 'Full', que es la misma población con la que se entrenó el modelo.
    """
    if "report_coverage" not in df:
        return df
    return df[df["report_coverage"] == "Full"]
