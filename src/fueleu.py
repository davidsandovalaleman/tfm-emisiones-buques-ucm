"""FuelEU Maritime: intensidad GEI del combustible, balance de cumplimiento y penalizacion.

Tercera pata regulatoria del trabajo, despues del CII (`src/cii.py`) y del EU ETS (`src/ets.py`).
FuelEU no grava las emisiones: grava **la intensidad de carbono de la energia consumida**, medida de
pozo a estela (*well-to-wake*), y multa la desviacion.

    GHGIE_objetivo(anio) = 91,16 x (1 - reduccion(anio))
    balance [gCO2eq]     = (GHGIE_objetivo - GHGIE_real) x energia_en_ambito
    penalizacion [EUR]   = |deficit| / (GHGIE_real x 41.000) x 2.400

donde 41.000 MJ es el contenido energetico de una tonelada de VLSFO y 2.400 EUR su precio de
referencia: la multa es, literalmente, *el combustible equivalente que habria hecho falta ahorrar*.

--------------------------------------------------------------------------------------------------
EL PROBLEMA, Y COMO SE RESUELVE CON DATOS PUBLICOS
--------------------------------------------------------------------------------------------------

Para calcular la intensidad hace falta la **energia** consumida en megajulios, y el MRV publica el
combustible en **toneladas**, sin decir de que tipo. Sin el tipo no hay poder calorifico, y sin
poder calorifico no hay energia.

Se resuelve con el mismo movimiento que el notebook 03 usa para recuperar la capacidad: **dividir dos
magnitudes publicadas para recuperar una que no lo esta**. El cociente CO2/combustible es la huella
dactilar del combustible quemado, porque cada uno tiene su factor de emision fijo:

    fuelóleo pesado 3,114 · gasoleo marino 3,206 · GNL 2,750   (t CO2 / t combustible)

Y el dato lo confirma solo, sin ayuda: los metaneros del registro dan una mediana de **2,750**
exacta, y los buques de pasaje y offshore **3,206** exacta. No hay que suponer el combustible: esta
escrito en el cociente.

De ahi sale la mezcla de cada buque, su poder calorifico y su factor de pozo a tanque, y con la
emision de CO2 equivalente que el propio MRV declara, la intensidad completa.

--------------------------------------------------------------------------------------------------
LA VALIDACION QUE HACE CREIBLE TODO LO DEMAS
--------------------------------------------------------------------------------------------------

El reglamento fija su valor de referencia en **91,16 gCO2eq/MJ**, que es la media de la flota
mundial en 2020. Esta reconstruccion, hecha desde cero con datos publicos y factores por defecto,
da una **mediana de 91,24 en 2025 y 91,36 en 2024**: un 0,2% de diferencia frente al numero que el
legislador calculo por su cuenta con datos que no son estos. Es la mejor prueba de que el metodo
funciona, y no ha costado comprar un solo dato.

--------------------------------------------------------------------------------------------------
LO QUE ESTA ESTIMACION NO PUEDE SABER, Y POR TANTO SOBRESTIMA
--------------------------------------------------------------------------------------------------

1. **Los combustibles renovables no se ven.** Un buque que queme biodiesel certificado tiene un
   factor de pozo a tanque muchisimo mas bajo, pero su cociente CO2/combustible es casi identico al
   del gasoleo fosil. Aqui contaria como fosil. Por eso la penalizacion calculada es una **cota
   superior**, y su signo de error se conoce.
2. **No se modela la agrupacion (*pooling*) ni el arrastre de saldos (*banking*)**, que son los dos
   mecanismos con los que el sector evita la mayor parte de la multa. Lo que se calcula es *la
   penalizacion si cada buque liquidara solo*.
3. **No se modela la energia de tierra (OPS) ni los factores de recompensa** por viento o por
   combustibles RFNBO.
4. **El ambito energetico se aproxima con el del ETS** (misma regla 100%/50%, pero medida sobre CO2
   en vez de sobre energia).
5. **La mezcla es de dos componentes.** El metanol (1,375) y el GLP quedan fuera del modelo y se
   marcan como no fiables.

Con esas cinco salvedades escritas, la cifra sirve para lo que sirve: ordenar buques, comparar
flotas y dimensionar el orden de magnitud del problema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------
# Tablas del reglamento (UE) 2023/1805
# --------------------------------------------------------------------------------------------

VALOR_REFERENCIA = 91.16          # gCO2eq/MJ, media de la flota mundial en 2020
ENERGIA_VLSFO_MJ_POR_TONELADA = 41_000
PRECIO_PENALIZACION_EUR = 2_400   # EUR por tonelada de VLSFO equivalente de deficit

# Reduccion exigida sobre el valor de referencia, por periodo.
REDUCCIONES = {2025: 0.02, 2030: 0.06, 2035: 0.145, 2040: 0.31, 2045: 0.62, 2050: 0.80}

# Combustibles de referencia: factor de emision de CO2 (t/t), poder calorifico inferior (MJ/g) y
# factor de pozo a tanque (gCO2eq/MJ). Anexo II del reglamento.
COMBUSTIBLES = {
    "fueloleo_pesado": {"cf": 3.114, "lcv": 0.0405, "wtt": 13.5},
    "gasoleo_marino":  {"cf": 3.206, "lcv": 0.0427, "wtt": 14.4},
    "gnl":             {"cf": 2.750, "lcv": 0.0491, "wtt": 18.5},
}

COL_FUEL = "total_fuel_consumption_m_tonnes"
COL_CO2 = "total_co2_emissions_m_tonnes"
COL_CO2EQ = "total_co2eq_emissions_m_tonnes"


def reduccion(anio: int) -> float:
    """Reduccion exigida en `anio`: la del ultimo periodo que ya ha empezado."""
    aplicables = [a for a in REDUCCIONES if a <= int(anio)]
    if not aplicables:
        return 0.0
    return REDUCCIONES[max(aplicables)]


def objetivo(anio: int) -> float:
    """Intensidad GEI maxima permitida en `anio`, en gCO2eq/MJ."""
    return VALOR_REFERENCIA * (1 - reduccion(anio))


# --------------------------------------------------------------------------------------------
# Reconstruccion del combustible
# --------------------------------------------------------------------------------------------

def mezcla_de_combustible(df: pd.DataFrame) -> pd.DataFrame:
    """Deduce la mezcla de combustible de cada buque a partir del cociente CO2/combustible.

    Dos componentes, elegidos segun a que lado del fueloleo pesado caiga el cociente: por encima,
    mezcla de fueloleo y gasoleo marino; por debajo, de fueloleo y GNL. Devuelve el poder calorifico
    y el factor de pozo a tanque de la mezcla.

    El factor de pozo a tanque se pondera **por energia**, no por masa, porque esta expresado por
    megajulio. Ponderarlo por masa introduciria un sesgo del orden del 1% en las mezclas con GNL,
    que es justo el orden de magnitud del margen de cumplimiento.
    """
    pesado, gasoleo, gnl = (COMBUSTIBLES[k] for k in ("fueloleo_pesado", "gasoleo_marino", "gnl"))

    with np.errstate(divide="ignore", invalid="ignore"):
        cociente = df[COL_CO2] / df[COL_FUEL]
    cociente = cociente.where(np.isfinite(cociente))

    por_encima = cociente >= pesado["cf"]
    frac_gasoleo = ((cociente - pesado["cf"]) / (gasoleo["cf"] - pesado["cf"])).clip(0, 1)
    frac_gnl = ((pesado["cf"] - cociente) / (pesado["cf"] - gnl["cf"])).clip(0, 1)

    def _mezclar(fraccion: pd.Series, otro: dict) -> tuple[np.ndarray, np.ndarray]:
        lcv = (1 - fraccion) * pesado["lcv"] + fraccion * otro["lcv"]
        energia_pesado = (1 - fraccion) * pesado["lcv"]
        energia_otro = fraccion * otro["lcv"]
        wtt = (energia_pesado * pesado["wtt"] + energia_otro * otro["wtt"]) / (energia_pesado + energia_otro)
        return lcv.to_numpy(), wtt.to_numpy()

    lcv_arriba, wtt_arriba = _mezclar(frac_gasoleo, gasoleo)
    lcv_abajo, wtt_abajo = _mezclar(frac_gnl, gnl)

    return pd.DataFrame(
        {
            "cociente_co2_combustible": cociente,
            "frac_gasoleo": frac_gasoleo.where(por_encima, 0.0),
            "frac_gnl": frac_gnl.where(~por_encima, 0.0),
            "lcv_MJ_por_g": np.where(por_encima, lcv_arriba, lcv_abajo),
            "wtt_g_por_MJ": np.where(por_encima, wtt_arriba, wtt_abajo),
            # Fuera del rango de los tres combustibles modelados: metanol, GLP o dato erroneo.
            "mezcla_fiable": cociente.between(gnl["cf"] * 0.98, gasoleo["cf"] * 1.02),
        },
        index=df.index,
    )


# --------------------------------------------------------------------------------------------
# Intensidad, balance y penalizacion
# --------------------------------------------------------------------------------------------

def calcular(
    df: pd.DataFrame,
    anio_regimen: int | None = None,
    factor_ambito: pd.Series | np.ndarray | float = 1.0,
) -> pd.DataFrame:
    """Intensidad GEI, balance de cumplimiento y penalizacion de cada fila buque-ano.

    `factor_ambito` es la fraccion de la energia que cae dentro del ambito de FuelEU. Se le pasa
    normalmente el factor del ETS (`ets.factor_ambito`), porque la regla geografica es la misma.
    """
    salida = mezcla_de_combustible(df)
    anio = int(anio_regimen) if anio_regimen is not None else None

    # Energia consumida: toneladas -> gramos -> megajulios.
    salida["energia_MJ"] = df[COL_FUEL] * 1e6 * salida["lcv_MJ_por_g"]

    # Intensidad de tanque a estela, medida con el CO2 equivalente que declara el propio buque.
    # Antes de 2024 el MRV no publicaba CH4 ni N2O, asi que se cae al CO2 y se marca.
    co2eq = df[COL_CO2EQ] if COL_CO2EQ in df else pd.Series(np.nan, index=df.index)
    salida["co2eq_disponible"] = co2eq.notna() & (co2eq > 0)
    co2eq = co2eq.where(salida["co2eq_disponible"], df[COL_CO2])
    with np.errstate(divide="ignore", invalid="ignore"):
        salida["ghgie_ttw"] = co2eq * 1e6 / salida["energia_MJ"]
    salida["ghgie_wtw"] = salida["ghgie_ttw"] + salida["wtt_g_por_MJ"]

    anios = pd.Series(anio, index=df.index) if anio is not None else df["reporting_year"].astype(int)
    salida["anio_regimen"] = anios
    salida["objetivo"] = anios.map(objetivo)

    energia_ambito = salida["energia_MJ"] * factor_ambito
    salida["energia_en_ambito_MJ"] = energia_ambito
    salida["balance_gco2eq"] = (salida["objetivo"] - salida["ghgie_wtw"]) * energia_ambito
    # Cumplir se decide sobre la INTENSIDAD, no sobre el signo del balance. No es lo mismo: un buque
    # con energia cero dentro del ambito tiene balance cero, y leerlo como "cumple" inflaria la tasa
    # de cumplimiento del 4% real al 10%.
    salida["cumple"] = salida["ghgie_wtw"] <= salida["objetivo"]

    deficit = (-salida["balance_gco2eq"]).clip(lower=0)
    salida["deficit_t_vlsfo_eq"] = deficit / (salida["ghgie_wtw"] * ENERGIA_VLSFO_MJ_POR_TONELADA)
    salida["penalizacion_eur"] = salida["deficit_t_vlsfo_eq"] * PRECIO_PENALIZACION_EUR

    salida["calculable"] = (
        np.isfinite(salida["ghgie_wtw"]) & (salida["energia_MJ"] > 0) & salida["mezcla_fiable"]
    )
    return salida
