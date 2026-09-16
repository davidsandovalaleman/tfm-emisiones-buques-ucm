"""
Productivizacion: API HTTP del simulador de emisiones de CO2 de buques.

Este modulo NO calcula nada por su cuenta. Es una envoltura HTTP sobre `src/simulador.py`, el mismo
modulo que ejecuta el notebook 08, de modo que la cifra que devuelve un endpoint y la que aparece
en la memoria salen literalmente de la misma linea de codigo. Cualquier discrepancia entre la API
y el notebook seria un fallo de esta capa, nunca una diferencia de metodo.

--------------------------------------------------------------------------------------------------
QUE EXPONE Y POR QUE
--------------------------------------------------------------------------------------------------

    GET  /                      panel web: el simulador con sliders y graficos
    GET  /salud                 estado del servicio y que modelos ha conseguido cargar
    GET  /metadatos             metricas de test, valores admitidos y limites del modelo
    POST /predecir              un buque: cuanto CO2 emite con esas caracteristicas
    POST /simular/buque         un buque: efecto de uno o varios escenarios operativos
    POST /simular/flota         una flota: ahorro agregado de un escenario, con banda
    POST /simular/focalizado    misma reduccion de horas, repartida o concentrada: cuanto cambia
    GET  /buques/buscar         localiza un buque por nombre o IMO, para el panel
    GET  /ets/parametros        reglas vigentes del EU ETS y precio de referencia del derecho

Los dos primeros son de servicio. `/predecir` y `/simular/buque` responden a la pregunta del
armador ("este barco, este ano, que emite y que pasaria si..."). `/simular/flota` y
`/simular/focalizado` responden a la pregunta del regulador ("que politica rinde mas"), y el
segundo es el resultado de cabecera del TFM: a igualdad de horas de navegacion evitadas,
concentrarlas en los cuatro tipos de mayor intensidad duplica el ahorro.

--------------------------------------------------------------------------------------------------
CUATRO DECISIONES DE DISENO
--------------------------------------------------------------------------------------------------

1. **Los modelos se cargan una sola vez, al arrancar** (`lifespan`), no en cada peticion. Cargar el
   XGBoost y la red por peticion multiplicaria por mil el tiempo de respuesta. Como efecto
   colateral util: si falta un artefacto, el servicio no arranca en vez de fallar en produccion a
   la primera llamada.

2. **El ahorro se mide predicho contra predicho**, igual que en el notebook 08: nunca contra el
   dato observado. Lo garantiza `simulador.simular()`, no esta capa.

3. **El coste en euros nunca se calcula sobre el ahorro total.** El modelo predice la emision
   completa de un buque, pero el EU ETS solo cubre el 100% de los viajes entre puertos del EEE y
   las emisiones en puerto, y el 50% de los viajes con un extremo fuera: por eso cada respuesta con
   coste trae la fraccion dentro del ambito -- la mediana de la flota es 0,55--, el factor de
   entrega del ano y el precio del derecho con su fecha. Multiplicar toneladas por euros sin ese
   desglose sobrestimaria el resultado en un 70%.

4. **Ninguna respuesta sale sin su aviso de fiabilidad.** El notebook 04 midio que el error relativo
   mediano del modelo es del 26,1% en el quintil de buques que menos emiten frente al 12-13% del
   resto. Devolver "4.812 t" a secas para un buque pequeno seria tecnicamente correcto y
   practicamente enganoso, asi que cada prediccion individual viaja con el error tipico de su
   tramo. El agregado de flota es fiable; el buque pequeno individual, no.

--------------------------------------------------------------------------------------------------
ARRANQUE
--------------------------------------------------------------------------------------------------

    conda activate tfm                      # entorno del proyecto, Python 3.11.15
    pip install -r requirements.txt
    python -m uvicorn src.api:app --reload  # desde la RAIZ del repositorio
    http://127.0.0.1:8000/docs              # documentacion interactiva (OpenAPI)

Con `python -m uvicorn` y no con `uvicorn` a secas: el ejecutable suelto usa el primer Python del
PATH, que rara vez es el del entorno del proyecto. El sintoma de equivocarse es un
`ModuleNotFoundError: No module named 'fastapi'` que parece un fallo del codigo y es de interprete.

`/docs` la genera FastAPI a partir de los modelos Pydantic de este fichero: es documentacion que no
puede desincronizarse del codigo porque se deriva de el.
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

# El simulador vive en la misma carpeta. Anadirla al path permite arrancar tanto con
# `uvicorn src.api:app` desde la raiz como con `uvicorn api:app` desde dentro de src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cii  # noqa: E402
import economia  # noqa: E402
import ets  # noqa: E402
import fueleu  # noqa: E402
import simulador as sim  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "models"
DIR_DATOS = RAIZ / "data" / "processed"
DIR_WEB = RAIZ / "web"
DIR_INFORMES = RAIZ / "reports"

VERSION = "1.0.0"

# Categorias de referencia del one-hot (las que NO tienen columna propia porque se usaron como
# base en el notebook 03). Se admiten como entrada valida: significan "todas las dummies a 0".
TIPO_REFERENCIA = "Bulk carrier"
HIELO_REFERENCIA = "IA"
METODO_REFERENCIA = "Desconocido"

# Estado del servicio. Se rellena en el arranque y de ahi no se vuelve a tocar.
ESTADO: dict = {}


# ==================================================================================================
# Arranque
# ==================================================================================================


def _factores_ambito_ets(raw: pd.DataFrame) -> dict:
    """
    Fraccion de las emisiones de cada buque-ano que cae dentro del ambito del EU ETS.

    Es la pieza que traduce toneladas a euros sin mentir. El modelo predice la emision TOTAL de un
    buque, pero el regimen solo cubre el 100% de los viajes intra-EEE y las emisiones en puerto, y
    el 50% de los viajes con un extremo fuera. Esa fraccion tiene mediana 0,55: dar por bueno el
    ahorro total como si fuera ahorro de derechos sobrestimaria el resultado en un 70%.

    No se reconstruye la regla: desde el ejercicio 2024 el propio MRV publica la emision dentro del
    ambito, ya verificada. `ets.factor_ambito` la usa y cae a la reconstruccion 100/50 solo si falta.

    Se calcula una vez al arrancar, alineada fila a fila con `raw`, porque las peticiones de flota
    la indexan con la misma mascara que las predicciones.
    """
    consolidado = ets.solo_informes_completos(pd.read_parquet(DIR_DATOS / "mrv_consolidado.parquet"))
    consolidado = consolidado.drop_duplicates(["reporting_year", "ship_imo_number"]).copy()
    consolidado["_f"] = ets.factor_ambito(consolidado)

    indice = pd.MultiIndex.from_arrays([raw["reporting_year"], raw["ship_imo_number"]])
    tabla = consolidado.set_index(["reporting_year", "ship_imo_number"])["_f"]
    factor = pd.Series(tabla.reindex(indice).to_numpy(), index=raw.index)
    mediana = float(factor.median())

    # FuelEU: la penalizacion anual de cada buque bajo cada uno de los seis periodos del
    # reglamento. Son seis columnas de 106.521 numeros, asi que se precalculan enteras al arrancar
    # en vez de rehacerlas en cada peticion.
    ambito = consolidado["_f"].fillna(consolidado["_f"].median())
    penalizaciones: dict[int, np.ndarray] = {}
    for periodo in sorted(fueleu.REDUCCIONES):
        calculo = fueleu.calcular(consolidado, anio_regimen=periodo, factor_ambito=ambito)
        serie = calculo["penalizacion_eur"].where(calculo["calculable"])
        serie.index = consolidado.set_index(["reporting_year", "ship_imo_number"]).index
        penalizaciones[periodo] = np.nan_to_num(serie.reindex(indice).to_numpy(dtype="float64"))
    referencia = fueleu.calcular(consolidado, anio_regimen=2025, factor_ambito=ambito)
    intensidad = referencia["ghgie_wtw"].where(referencia["calculable"])
    intensidad.index = consolidado.set_index(["reporting_year", "ship_imo_number"]).index
    cobertura_fueleu = float(referencia["calculable"].mean())
    # La mediana de referencia se mide SOLO sobre el ultimo ejercicio: antes de 2024 el MRV no
    # publicaba CH4 ni N2O, asi que aquellos anos salen artificialmente bajos y mezclarlos
    # ensuciaria justo la cifra que valida el modulo.
    ultimo_ejercicio = int(consolidado["reporting_year"].max())
    del_ultimo = referencia.loc[
        referencia["calculable"] & (consolidado["reporting_year"] == ultimo_ejercicio), "ghgie_wtw"
    ]
    mediana_intensidad = float(del_ultimo.median())

    # El uplift de CO2eq se mide sobre el ultimo ejercicio disponible y sobre los TOTALES, no sobre
    # la columna de ambito en CO2eq: en el ejercicio 2025 esa columna se publico copiada de la de
    # CO2 (coincide en el 100% de las filas), asi que es inservible. Ver `ets.uplift_co2eq`.
    ultimo = int(consolidado["reporting_year"].max())
    return {
        "por_fila": factor.fillna(mediana).to_numpy(dtype="float64"),
        "cobertura": float(factor.notna().mean()),
        "mediana": mediana,
        "uplift_co2eq": float(ets.uplift_co2eq(consolidado[consolidado["reporting_year"] == ultimo])),
        "anio_uplift": ultimo,
        "fueleu": {
            "penalizacion_por_periodo": penalizaciones,
            "intensidad_por_fila": intensidad.reindex(indice).to_numpy(dtype="float64"),
            "cobertura": cobertura_fueleu,
            "mediana_intensidad": mediana_intensidad,
            "anio_mediana": ultimo_ejercicio,
        },
    }


def _cargar_economia() -> dict | None:
    """
    Carga los artefactos de la capa economica, si estan calculados.

    La API **no recalcula** el analisis: eso son tres ajustes de XGBoost y no cabe en un arranque de
    servicio. Lee los informes que escribe `scripts/analisis_economia.py`, igual que el panel lee la
    API. Si no estan, devuelve `None` y los endpoints responden un 503 explicativo en vez de un
    error raro: es la misma degradacion declarada que se usa con la red neuronal cuando falta
    TensorFlow.
    """
    ficha = DIR_INFORMES / "18_pasaporte_buques.csv"
    if not ficha.exists():
        return None
    buques = pd.read_csv(ficha, dtype={"ship_imo_number": str})
    return {
        "buques": buques.set_index("ship_imo_number", drop=False),
        "ahorro": pd.read_csv(DIR_INFORMES / "14_ahorro_actividad_constante.csv"),
        "persistencia": pd.read_csv(DIR_INFORMES / "14_persistencia_residuo.csv").iloc[0].to_dict(),
        "sensibilidad": pd.read_csv(DIR_INFORMES / "14_sensibilidad_precios.csv"),
        "palancas": pd.read_csv(DIR_INFORMES / "15_palancas_eur_por_dia.csv"),
        "navieras": pd.read_csv(DIR_INFORMES / "16_navieras_exposicion.csv"),
        "pooling": pd.read_csv(DIR_INFORMES / "16_fueleu_pooling.csv").iloc[0].to_dict(),
        "bandas": pd.read_csv(DIR_INFORMES / "17_cii_trayectoria_bandas.csv", index_col=0),
    }


def _cargar_estado() -> dict:
    """
    Carga modelos y datos una sola vez y precalcula lo que toda respuesta necesita.

    Lo unico no evidente que hace aqui es la tabla de fiabilidad: se calcula sobre la particion de
    TEST (`es_train == False`), que son buques que ningun modelo vio entrenando, porque un error
    tipico medido sobre el train seria optimista y el aviso perderia todo su valor.
    """
    inicio = time.perf_counter()

    modelos = sim.cargar_modelos(str(DIR_MODELOS))
    ml = pd.read_parquet(DIR_DATOS / "mrv_features_operacional_tamano_ml.parquet")
    raw = pd.read_parquet(DIR_DATOS / "mrv_features_operacional_tamano_raw.parquet")

    # Las familias y la de referencia las decide el propio simulador segun el nivel cargado: en el
    # nivel operacional+tamano no existe 'mezcla' (ver simulador.familias_disponibles).
    familias = list(modelos["familias"])

    # Fiabilidad por tramo de emision, medida solo en test.
    es_test = ~ml["es_train"].to_numpy()
    ml_test = ml.loc[es_test]
    familia_ref = modelos["familia_principal"]
    y_real = ml_test[sim.COL_TARGET].to_numpy(dtype="float64")
    y_pred = sim.predecir(modelos, ml_test, familia_ref)
    fiabilidad = sim.fiabilidad_por_tamano(y_real, y_pred)
    # Cortes de los quintiles, para poder situar una prediccion nueva en su tramo.
    cortes = np.quantile(y_real, [0.2, 0.4, 0.6, 0.8])

    # Valor tipico de eficiencia tecnica por tipo de buque, para cuando el usuario no lo aporta.
    te_mediana_por_tipo = raw.groupby("ship_type_agrupado", observed=True)["te_valor"].median().to_dict()

    return {
        "modelos": modelos,
        "ml": ml,
        "raw": raw,
        "columnas": list(modelos["columnas"]),
        "familias": familias,
        "familia_por_defecto": familia_ref,
        "fiabilidad": fiabilidad,
        "cortes_quintiles": cortes,
        "te_mediana_por_tipo": te_mediana_por_tipo,
        "tipos_validos": sorted(raw["ship_type_agrupado"].dropna().unique().tolist()),
        "hielo_validos": sorted(raw["ice_class"].dropna().unique().tolist()),
        "metodos_validos": sorted(raw["te_metodo"].dropna().unique().tolist()),
        "anios": sorted(int(a) for a in raw["reporting_year"].unique()),
        "ets": _factores_ambito_ets(raw),
        "economia": _cargar_economia(),
        "base_cache": {},  # familia -> predicciones de todo el dataset, calculadas al pedirlas
        "segundos_arranque": time.perf_counter() - inicio,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    ESTADO.update(_cargar_estado())
    yield
    ESTADO.clear()


app = FastAPI(
    title="Estela - API de emisiones y cumplimiento de la flota europea",
    version=VERSION,
    description=(
        "Productivizacion del TFM. Envoltura HTTP sobre `src/simulador.py`: predice las "
        "emisiones de un buque y simula escenarios operativos de reduccion sobre la flota del EEE, "
        "a partir de los datos publicos THETIS-MRV (2018-2025)."
    ),
    lifespan=lifespan,
)


# ==================================================================================================
# Utilidades internas
# ==================================================================================================


def _predicciones_base(familia: str) -> np.ndarray:
    """Predicciones del caso sin cambios sobre TODO el dataset, calculadas una vez y reutilizadas."""
    cache = ESTADO["base_cache"]
    if familia not in cache:
        cache[familia] = sim.predecir(ESTADO["modelos"], ESTADO["ml"], familia)
    return cache[familia]


def _familias_a_usar(familia: str | None, banda: bool = True) -> tuple[str, ...]:
    """
    Familias que hay que evaluar para responder.

    Con `banda=True` (lo normal) se evaluan las tres, porque la banda por desacuerdo entre modelos
    es como el TFM reporta sus cifras (restriccion 3 de `src/simulador.py`) y no tendria sentido
    dejarla fuera de la respuesta por defecto.

    Con `banda=False` se evalua solo la pedida. No es un atajo cosmetico: sobre la flota de un
    ejercicio, calcular las tres familias cuesta ~1,8 s y calcular una sola ~0,2 s, porque la red
    neuronal acaba ejecutandose dos veces (una como `red` y otra dentro de `mezcla`). Un consumidor
    que esta explorando —el panel mientras se arrastra un deslizador— no necesita la banda en cada
    fotograma; la pide al soltar. La cifra central es identica en los dos casos.
    """
    familia = familia if familia is not None else ESTADO["familia_por_defecto"]
    if familia not in ESTADO["familias"]:
        raise HTTPException(
            status_code=503,
            detail=f"La familia '{familia}' no esta disponible: solo se cargo {ESTADO['familias']}.",
        )
    if banda and len(ESTADO["familias"]) > 1:
        return tuple(ESTADO["familias"])
    return (familia,)


def _banda(por_familia: dict, clave: str) -> dict | None:
    """Banda por desacuerdo entre familias. No es un intervalo de confianza y no se llama asi."""
    referencia = ESTADO["familia_por_defecto"]
    extremos = [por_familia[f][clave] for f in ("xgb", "red") if f in por_familia]
    if referencia not in por_familia or len(extremos) < 2:
        # Sin las dos familias por separado no hay desacuerdo que medir, y una banda de un solo
        # punto seria una banda falsa. Se devuelve None y la respuesta lo dice con un hueco, no
        # con un intervalo inventado.
        return None
    return {
        "minimo": float(min(extremos)),
        "central": float(por_familia[referencia][clave]),
        "maximo": float(max(extremos)),
        "que_mide": (
            "desacuerdo entre XGBoost y la red neuronal; no es un intervalo de confianza. "
            "En el nivel operacional+tamano la red esta entrenada sobre menos informacion, asi "
            "que la banda mide tambien la sensibilidad al nivel de informacion"
        ),
    }


def _aviso_fiabilidad(co2_t: float) -> dict:
    """Error relativo tipico del tramo en el que cae esta prediccion, medido en test."""
    quintil = int(np.searchsorted(ESTADO["cortes_quintiles"], co2_t, side="right")) + 1
    tabla = ESTADO["fiabilidad"]
    fila = tabla.loc[quintil] if quintil in tabla.index else tabla.iloc[-1]
    error = float(fila["error_relativo_mediano_pct"])
    return {
        "quintil_de_emision": quintil,
        "error_relativo_mediano_pct": round(error, 1),
        "aviso": (
            "Buque de emisión baja: el modelo es proporcionalmente menos preciso en este tramo. "
            "Úselo para comparar escenarios, no como medición del buque."
            if quintil == 1
            else "Precisión típica del modelo en este tramo de emisión."
        ),
    }


def _fila_desde_entrada(buque: "BuqueEntrada") -> tuple[pd.DataFrame, list[str]]:
    """
    Traduce las caracteristicas legibles de un buque a la fila de 35 columnas que espera el modelo.

    Devuelve tambien la lista de avisos: cualquier valor que la API haya tenido que rellenar por su
    cuenta sale aqui, para que ninguna imputacion viaje escondida dentro de la prediccion.
    """
    avisos: list[str] = []
    fila = {c: 0.0 for c in ESTADO["columnas"]}

    fila["time_spent_at_sea_hours"] = float(buque.horas_mar)
    fila["es_subcategoria_nueva"] = 1.0 if buque.es_subcategoria_nueva else 0.0

    if buque.velocidad_nudos is None:
        fila["velocidad_nudos"] = np.nan
        avisos.append(
            "Sin velocidad media: XGBoost la trata como ausente de forma nativa y la red la imputa "
            "por la mediana del entrenamiento. La palanca de velocidad no es utilizable en este caso."
        )
    else:
        fila["velocidad_nudos"] = float(buque.velocidad_nudos)

    if buque.te_valor is None:
        mediana = ESTADO["te_mediana_por_tipo"].get(buque.tipo_buque)
        if mediana is None or not np.isfinite(mediana):
            raise HTTPException(status_code=422, detail=f"Tipo de buque sin mediana de eficiencia: {buque.tipo_buque}")
        fila["te_valor"] = float(mediana)
        avisos.append(
            f"Sin índice de eficiencia técnica: se usa la mediana de '{buque.tipo_buque}' "
            f"({mediana:.2f}). Aportar el EEDI/EIV real mejora la predicción."
        )
    else:
        fila["te_valor"] = float(buque.te_valor)

    for valor, prefijo, referencia, validos in (
        (buque.tipo_buque, "ship_type_agrupado_", TIPO_REFERENCIA, ESTADO["tipos_validos"]),
        (buque.ice_class, "ice_class_", HIELO_REFERENCIA, ESTADO["hielo_validos"]),
        (buque.te_metodo, "te_metodo_", METODO_REFERENCIA, ESTADO["metodos_validos"]),
    ):
        if valor not in validos:
            raise HTTPException(status_code=422, detail=f"Valor no admitido: {valor!r}. Admitidos: {validos}")
        columna = f"{prefijo}{valor}"
        if columna in fila:
            fila[columna] = 1.0
        elif valor != referencia:  # pragma: no cover - imposible con la validacion de arriba
            raise HTTPException(status_code=422, detail=f"Sin columna para {valor!r}")

    if buque.prop_missings in ("0.25", "0.5"):
        fila[f"prop_missings_{buque.prop_missings}"] = 1.0

    return pd.DataFrame([fila], columns=ESTADO["columnas"]), avisos


def _coste_ets(
    ahorro_por_fila: np.ndarray,
    factor_ambito: np.ndarray,
    anio_regimen: int,
    precio: float | None,
) -> dict:
    """
    Valor en derechos de emision de un ahorro de CO2, con las tres reglas del regimen aplicadas.

    Devuelve siempre el desglose completo -- cuanto del ahorro cae dentro del ambito, que fraccion
    hay que entregar ese ano y a que precio-- para que la cifra en euros nunca viaje sola. El
    precio es una cotizacion de mercado, no un resultado del trabajo: va con su fecha y su fuente.
    """
    precio = float(precio) if precio is not None else ets.PRECIO_EUA_POR_DEFECTO
    total = float(np.nansum(ahorro_por_fila))
    en_ambito = float(np.nansum(np.asarray(ahorro_por_fila, dtype="float64") * factor_ambito))
    # La regla «desde 2026 la base es CO2 equivalente» vive en `ets.base_desde_ambito`, no aquí: es
    # el mismo condicional que aplica `ets.base_imponible` sobre el registro, y la formula de la
    # factura vive en un unico sitio.
    base = ets.base_desde_ambito(en_ambito, anio_regimen, ESTADO["ets"]["uplift_co2eq"])
    entrega = ets.factor_entrega(anio_regimen)

    return {
        "anio_regimen": int(anio_regimen),
        "factor_entrega": entrega,
        "base_gases": ets.base_gases(anio_regimen),
        "precio_eua_eur_t": round(precio, 2),
        "precio_referencia": {"fecha": ets.PRECIO_EUA_FECHA, "fuente": ets.PRECIO_EUA_FUENTE},
        # Se dan las mismas cifras en toneladas y en Mt a proposito: la respuesta sirve tanto para
        # una flota de 17.000 buques como para uno solo, y redondear a Mt un ahorro individual
        # convertiria 988.000 EUR en "1,0 millones", que es precision fingida.
        "ahorro_total_t": round(total, 1),
        "ahorro_en_ambito_t": round(en_ambito, 1),
        "derechos_evitados_t": round(base * entrega, 1),
        "ahorro_eur": round(float(ets.coste(base, anio_regimen, precio)), 2),
        "ahorro_total_Mt": round(total / 1e6, 4),
        "ahorro_en_ambito_Mt": round(en_ambito / 1e6, 4),
        "pct_del_ahorro_dentro_del_ambito": round(100.0 * en_ambito / total, 1) if total else None,
        "derechos_evitados_Mt": round(base * entrega / 1e6, 4),
        "ahorro_millones_eur": round(float(ets.coste(base, anio_regimen, precio)) / 1e6, 1),
        "nota": (
            "Solo la parte del ahorro dentro del ambito del ETS se convierte en derechos: 100% de "
            "los viajes entre puertos del EEE y de las emisiones en puerto, 50% de los viajes con "
            "un extremo fuera. Desde 2026 la base incluye ademas CH4 y N2O."
        ),
    }


def _periodo_fueleu(anio_regimen: int) -> int:
    """Periodo de FuelEU vigente en `anio_regimen`: la reduccion cambia cada cinco anos."""
    aplicables = [a for a in fueleu.REDUCCIONES if a <= int(anio_regimen)]
    return max(aplicables) if aplicables else min(fueleu.REDUCCIONES)


def _penalizacion_fueleu(mascara: np.ndarray | None, anio_regimen: int) -> np.ndarray:
    """Penalizacion anual de FuelEU de las filas seleccionadas, bajo el periodo de `anio_regimen`."""
    tabla = ESTADO["ets"]["fueleu"]["penalizacion_por_periodo"][_periodo_fueleu(anio_regimen)]
    return tabla if mascara is None else tabla[mascara]


def _coste_fueleu(
    ahorro_por_fila: np.ndarray,
    base_por_fila: np.ndarray,
    penalizacion: np.ndarray,
    anio_regimen: int,
) -> dict:
    """
    Penalizacion de FuelEU que deja de pagarse al aplicar un escenario.

    FuelEU no grava emisiones sino **intensidad**, asi que navegar menos no mejora la calificacion
    de un buque: baja la energia consumida dentro del ambito y con ella la multa, en la misma
    proporcion. Esa proporcionalidad es lo que se aplica aqui -- la energia cae como el combustible
    y el combustible como el CO2, porque la mezcla no cambia--, y es tambien el motivo por el que
    el ahorro en FuelEU es mucho menor que el del ETS a igualdad de toneladas evitadas.
    """
    periodo = _periodo_fueleu(anio_regimen)
    base = np.asarray(base_por_fila, dtype="float64")
    penalizacion = np.asarray(penalizacion, dtype="float64")
    if penalizacion.size != base.size:
        # Un buque descrito a mano no esta en el registro, asi que no tiene penalizacion que
        # reducir. Se devuelve cero, no una estimacion inventada a partir de la mediana.
        penalizacion = np.zeros_like(base)
    with np.errstate(divide="ignore", invalid="ignore"):
        proporcion = np.where(base > 0, np.asarray(ahorro_por_fila, dtype="float64") / base, 0.0)
    evitada = float(np.nansum(penalizacion * proporcion))

    return {
        "periodo": periodo,
        "objetivo_gco2eq_MJ": round(fueleu.objetivo(anio_regimen), 3),
        "penalizacion_actual_eur": round(float(np.nansum(penalizacion)), 2),
        "penalizacion_evitada_eur": round(evitada, 2),
        "penalizacion_evitada_millones_eur": round(evitada / 1e6, 1),
        "nota": (
            "FuelEU grava la intensidad de carbono de la energia, no las emisiones. Navegar menos "
            "reduce la multa en proporcion a la energia ahorrada, pero no mejora la intensidad del "
            "buque: eso solo se consigue cambiando de combustible. La cifra es una cota superior, "
            "porque los combustibles renovables no son identificables en el registro publico y no "
            "se modelan la agrupacion ni el arrastre de saldos."
        ),
    }


def _seleccionar_flota(anio: int | None, tipos: list[str] | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Subconjunto de la flota sobre el que se simula, mas la mascara para indexar la cache base."""
    mascara = np.ones(len(ESTADO["ml"]), dtype=bool)
    if anio is not None:
        if anio not in ESTADO["anios"]:
            raise HTTPException(status_code=422, detail=f"Año no disponible: {anio}. Hay datos de {ESTADO['anios']}.")
        mascara &= (ESTADO["ml"]["reporting_year"] == anio).to_numpy()
    if tipos:
        desconocidos = [t for t in tipos if t not in ESTADO["tipos_validos"]]
        if desconocidos:
            raise HTTPException(status_code=422, detail=f"Tipos no admitidos: {desconocidos}")
        mascara &= ESTADO["raw"]["ship_type_agrupado"].isin(tipos).to_numpy()
    if not mascara.any():
        raise HTTPException(status_code=404, detail="Ningún buque cumple ese filtro.")
    return ESTADO["ml"].loc[mascara], mascara


# ==================================================================================================
# Contratos de entrada y salida
# ==================================================================================================


class BuqueEntrada(BaseModel):
    """Caracteristicas de un buque-ano. Solo el tipo y las horas son obligatorios."""

    tipo_buque: str = Field(..., description="Tipo agrupado THETIS-MRV, p.ej. 'Container ship'", examples=["LNG carrier"])
    horas_mar: float = Field(..., ge=0, le=sim.HORAS_MAX_FISICO, description="Horas de navegacion en el ano")
    velocidad_nudos: float | None = Field(None, gt=0, le=40, description="Velocidad media anual; opcional")
    te_valor: float | None = Field(None, ge=0, description="Indice de eficiencia tecnica (EEDI/EIV); opcional")
    te_metodo: str = Field(METODO_REFERENCIA, description="Metodo del indice: EEDI, EEXI, EIV, Not Applicable")
    ice_class: str = Field("Sin clase de hielo", description="Clase de hielo del buque")
    es_subcategoria_nueva: bool = Field(False, description="Buque que entra por la ampliacion del Reglamento 2023/957")
    prop_missings: Literal["0.0", "0.25", "0.5"] = Field("0.0", description="Tramo de datos ausentes del registro")


class PeticionPrediccion(BaseModel):
    buque: BuqueEntrada
    familia: Literal["xgb", "red", "mezcla"] | None = None


class Escenario(BaseModel):
    """Un escenario operativo. Las horas NO se fijan: se derivan de distancia = horas x velocidad."""

    delta_distancia: float = Field(0.0, gt=-1, le=1, description="-0.10 = recorrer un 10% menos de millas")
    delta_velocidad: float = Field(0.0, gt=-1, le=1, description="-0.10 = navegar un 10% mas despacio")
    etiqueta: str | None = Field(None, description="Nombre libre para identificar el escenario en la respuesta")


class PeticionBuque(BaseModel):
    """Un buque identificado por IMO+ano (dato real) o descrito a mano. Exactamente uno de los dos."""

    imo: str | None = Field(None, description="Numero IMO de un buque presente en THETIS-MRV")
    anio: int | None = Field(None, description="Ano de reporte, obligatorio junto a 'imo'")
    buque: BuqueEntrada | None = None
    escenarios: list[Escenario] = Field(..., min_length=1, max_length=20)
    familia: Literal["xgb", "red", "mezcla"] | None = None
    anio_regimen: int = Field(
        2026, ge=2024, le=2050,
        description="Regimen del ETS aplicado: 2024 entrega el 40% de lo emitido, 2025 el 70%, y desde 2026 el 100% e incluye CH4 y N2O",
    )
    precio_eua: float | None = Field(
        None, gt=0, le=1000,
        description="Precio del derecho de emision en EUR/t. Si se omite se usa la cotizacion de referencia del modulo",
    )


    @model_validator(mode="after")
    def _uno_u_otro(self):
        por_imo = self.imo is not None
        if por_imo == (self.buque is not None):
            raise ValueError("Indica 'imo' + 'anio' (buque real) o 'buque' (características), pero no ambos.")
        if por_imo and self.anio is None:
            raise ValueError("Con 'imo' hay que indicar también 'anio'.")
        return self


class PeticionFlota(BaseModel):
    anio: int | None = Field(2025, description="Ano de la flota simulada; null = los ocho ejercicios")
    tipos_buque: list[str] | None = Field(None, description="Restringe la flota a estos tipos")
    delta_distancia: float = Field(0.0, gt=-1, le=1)
    delta_velocidad: float = Field(0.0, gt=-1, le=1)
    desglose_por_tipo: bool = True
    bootstrap: bool = Field(False, description="Anade el intervalo del 95% por remuestreo de buques (mas lento)")
    banda: bool = Field(
        True,
        description=(
            "Calcula la banda por desacuerdo entre familias. Ponerlo a false evalua solo la familia "
            "pedida y responde ~9 veces mas rapido, con la misma cifra central; util para exploracion "
            "interactiva."
        ),
    )
    familia: Literal["xgb", "red", "mezcla"] | None = None
    anio_regimen: int = Field(
        2026, ge=2024, le=2050,
        description="Regimen del ETS aplicado: 2024 entrega el 40% de lo emitido, 2025 el 70%, y desde 2026 el 100% e incluye CH4 y N2O",
    )
    precio_eua: float | None = Field(
        None, gt=0, le=1000,
        description="Precio del derecho de emision en EUR/t. Si se omite se usa la cotizacion de referencia del modulo",
    )



class PeticionFocalizada(BaseModel):
    anio: int = 2025
    pct_horas: float | None = Field(0.05, gt=0, lt=1, description="Fraccion de horas de flota a evitar")
    horas_a_evitar: float | None = Field(None, gt=0, description="Alternativa absoluta a 'pct_horas'")
    tipos_objetivo: list[str] | None = Field(None, description="Por defecto, los cuatro de mayor intensidad")
    familia: Literal["xgb", "red", "mezcla"] | None = None
    anio_regimen: int = Field(
        2026, ge=2024, le=2050,
        description="Regimen del ETS aplicado: 2024 entrega el 40% de lo emitido, 2025 el 70%, y desde 2026 el 100% e incluye CH4 y N2O",
    )
    precio_eua: float | None = Field(
        None, gt=0, le=1000,
        description="Precio del derecho de emision en EUR/t. Si se omite se usa la cotizacion de referencia del modulo",
    )



# ==================================================================================================
# Endpoints de servicio
# ==================================================================================================


@app.get("/salud", tags=["servicio"], summary="Estado del servicio")
def salud() -> dict:
    if not ESTADO:
        raise HTTPException(status_code=503, detail="El servicio aun no ha terminado de arrancar.")
    return {
        "estado": "ok",
        "version": VERSION,
        "familias_disponibles": ESTADO["familias"],
        "red_neuronal_cargada": ESTADO["modelos"].get("red") is not None,
        # Si la red no está, se dice POR QUÉ. Un despliegue que responde sin banda tiene que poder
        # distinguir «la imagen no lleva TensorFlow, y es a propósito» de «el .keras no carga».
        "red_neuronal_motivo": ESTADO["modelos"].get("red_no_cargada"),
        "filas_cargadas": int(len(ESTADO["ml"])),
        "anios": ESTADO["anios"],
        "segundos_de_arranque": round(ESTADO["segundos_arranque"], 2),
    }


@app.get("/metadatos", tags=["servicio"], summary="Metricas, valores admitidos y limites del modelo")
def metadatos() -> dict:
    return {
        "modelo": {
            "nivel_de_informacion": (
                "operacional + tamano (buque, tiempo, velocidad media y capacidad de carga "
                "reconstruida)"
            ),
            "familia_de_referencia": ESTADO["familia_por_defecto"],
            "familias_disponibles": ESTADO["familias"],
            "red_neuronal_motivo": ESTADO["modelos"].get("red_no_cargada"),
            "metricas_test_xgboost": ESTADO["modelos"].get("metricas_xgb_test", {}),
            "r2_test_publicado": {
                "operacional_tamano": {"xgb": 0.9484},
                "operacional": {"xgb": 0.9164, "red": 0.9156, "mezcla": 0.9215},
            },
            "entrenado_con": "THETIS-MRV 2018-2025, partición 80/20 agrupada por buque",
        },
        "valores_admitidos": {
            "tipo_buque": ESTADO["tipos_validos"],
            "ice_class": ESTADO["hielo_validos"],
            "te_metodo": ESTADO["metodos_validos"],
            "prop_missings": ["0.0", "0.25", "0.5"],
            "anios": ESTADO["anios"],
        },
        "limites": {
            "horas_max_fisico": sim.HORAS_MAX_FISICO,
            "tipos_alta_intensidad": sim.TIPOS_ALTA_INTENSIDAD,
        },
        "fiabilidad_por_quintil_de_emision": ESTADO["fiabilidad"].reset_index().to_dict(orient="records"),
        "advertencias": [
            "El ahorro de un escenario se calcula predicho contra predicho, nunca contra el observado.",
            "El nivel absoluto de la línea base va un 2,7% alto sobre 2025; los ahorros, al ser "
            "diferencias, no se ven afectados.",
            "La palanca de velocidad está infravalorada: 'velocidad_nudos' es una media anual que "
            "mezcla navegación con tiempo parado. Su elasticidad es una cota inferior.",
            "El MRV público no incluye el tamaño del buque, que es la variable que más falta.",
        ],
        "fuente_de_datos": "THETIS-MRV (EMSA), Reglamento (UE) 2015/757. Reutilización citando la fuente.",
    }


# ==================================================================================================
# Endpoints de un buque
# ==================================================================================================


@app.get("/ets/parametros", tags=["servicio"], summary="Reglas vigentes del EU ETS maritimo")
def parametros_ets() -> dict:
    """
    Reglas del regimen y precio de referencia del derecho de emision.

    Existe para que el panel -- y cualquier otro consumidor-- no tenga que llevar las reglas
    duplicadas: se piden aqui y se muestran. El precio es una cotizacion de mercado que cambia cada
    dia, asi que viaja siempre con su fecha y su fuente, y cualquier peticion puede sustituirlo.
    """
    return {
        "precio_eua_eur_t": ets.PRECIO_EUA_POR_DEFECTO,
        "precio_fecha": ets.PRECIO_EUA_FECHA,
        "precio_fuente": ets.PRECIO_EUA_FUENTE,
        "factor_entrega_por_anio": {
            **{str(a): f for a, f in sorted(ets.FACTOR_ENTREGA.items())},
            "2026_en_adelante": ets.FACTOR_ENTREGA_PLENO,
        },
        "anio_desde_el_que_entran_ch4_y_n2o": ets.ANIO_CO2EQ,
        "uplift_co2eq_aplicado": round(ESTADO["ets"]["uplift_co2eq"], 4),
        "anio_del_uplift": ESTADO["ets"]["anio_uplift"],
        "ambito": {
            "viajes_entre_puertos_del_eee": 1.0,
            "viajes_con_un_extremo_fuera": 0.5,
            "emisiones_en_puerto": 1.0,
            "fraccion_mediana_observada": round(ESTADO["ets"]["mediana"], 3),
            "cobertura_del_dato_declarado": round(ESTADO["ets"]["cobertura"], 3),
        },
        "buques_cubiertos": "carga y pasaje de 5.000 GT o mas desde 2024; buques offshore desde 2027",
    }


@app.get("/fueleu/parametros", tags=["servicio"], summary="Reglas vigentes de FuelEU Maritime")
def parametros_fueleu() -> dict:
    """
    Reglas del reglamento (UE) 2023/1805 y alcance de la estimacion.

    Devuelve tambien la validacion que hace creible el modulo: la intensidad mediana reconstruida
    frente al valor de referencia que el propio legislador fijo con datos distintos.
    """
    return {
        "valor_de_referencia_gco2eq_MJ": fueleu.VALOR_REFERENCIA,
        "reduccion_por_periodo": {str(a): r for a, r in sorted(fueleu.REDUCCIONES.items())},
        "objetivo_por_periodo": {
            str(a): round(fueleu.objetivo(a), 3) for a in sorted(fueleu.REDUCCIONES)
        },
        "penalizacion_eur_por_tonelada_vlsfo_eq": fueleu.PRECIO_PENALIZACION_EUR,
        "energia_vlsfo_MJ_por_tonelada": fueleu.ENERGIA_VLSFO_MJ_POR_TONELADA,
        "buques_cubiertos": "buques de 5.000 GT o mas que tocan puertos del EEE, desde 2025",
        "intensidad_mediana_reconstruida": round(ESTADO["ets"]["fueleu"]["mediana_intensidad"], 3),
        "anio_de_la_mediana": ESTADO["ets"]["fueleu"]["anio_mediana"],
        "cobertura_del_calculo": round(ESTADO["ets"]["fueleu"]["cobertura"], 3),
        "limitaciones": [
            "El tipo de combustible no se declara: se deduce del cociente CO2/combustible con una "
            "mezcla de dos componentes (fueloleo, gasoleo marino, GNL).",
            "Los combustibles renovables certificados no son distinguibles, asi que su factor de "
            "pozo a tanque se sobrestima y la penalizacion calculada es una cota superior.",
            "No se modelan la agrupacion (pooling) ni el arrastre de saldos (banking), que es como "
            "el sector evita la mayor parte de la multa.",
            "El ambito energetico se aproxima con el del ETS, que sigue la misma regla 100%/50%.",
        ],
    }


@app.post("/predecir", tags=["un buque"], summary="Emisiones anuales de CO2 de un buque")
def predecir(peticion: PeticionPrediccion) -> dict:
    """
    Predice las toneladas de CO2 que emite un buque-ano con las caracteristicas indicadas.

    Devuelve siempre la banda entre familias de modelo y el error tipico del tramo de emision, para
    que la cifra no se lea como una medicion exacta.
    """
    fila, avisos = _fila_desde_entrada(peticion.buque)
    familia_pedida = peticion.familia or ESTADO["familia_por_defecto"]
    familias = _familias_a_usar(peticion.familia)

    por_familia = {f: float(sim.predecir(ESTADO["modelos"], fila, f)[0]) for f in familias}
    central = por_familia.get(familia_pedida, por_familia[familias[0]])

    banda = _banda({f: {"v": v} for f, v in por_familia.items()}, "v")
    if banda is not None:
        banda = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in banda.items()}

    return {
        "co2_predicho_t": round(central, 1),
        "familia": familia_pedida,
        "por_familia": {f: round(v, 1) for f, v in por_familia.items()},
        "banda": banda,
        "fiabilidad": _aviso_fiabilidad(central),
        "avisos": avisos,
    }


@app.post("/simular/buque", tags=["un buque"], summary="Efecto de escenarios operativos sobre un buque")
def simular_buque(peticion: PeticionBuque) -> dict:
    """
    Aplica uno o varios escenarios a un solo buque y devuelve el ahorro de cada uno.

    El buque puede darse de dos formas: por IMO y ano (se recupera su fila real de THETIS-MRV, que
    es el caso demostrativo) o describiendo sus caracteristicas a mano.
    """
    avisos: list[str] = []

    if peticion.imo is not None:
        seleccion = (
            (ESTADO["raw"]["ship_imo_number"].astype(str) == str(peticion.imo))
            & (ESTADO["raw"]["reporting_year"] == peticion.anio)
        ).to_numpy()
        if not seleccion.any():
            raise HTTPException(status_code=404, detail=f"No hay registro para el IMO {peticion.imo} en {peticion.anio}.")
        fila = ESTADO["ml"].loc[seleccion]
        fila_raw = ESTADO["raw"].loc[seleccion]
        factor_ambito = ESTADO["ets"]["por_fila"][seleccion][:1]
        mascara_fila = np.flatnonzero(seleccion)[:1]
        identificacion = {
            "imo": str(fila_raw["ship_imo_number"].iloc[0]),
            "nombre": str(fila_raw["ship_name"].iloc[0]),
            "tipo": str(fila_raw["ship_type_agrupado"].iloc[0]),
            "anio": int(fila_raw["reporting_year"].iloc[0]),
            "co2_observado_t": round(float(fila_raw[sim.COL_TARGET].iloc[0]), 1),
            "visto_en_entrenamiento": bool(fila_raw["es_train"].iloc[0]),
        }
    else:
        fila, avisos = _fila_desde_entrada(peticion.buque)
        identificacion = {"tipo": peticion.buque.tipo_buque, "co2_observado_t": None}
        # Un buque descrito a mano no tiene historico del que sacar su ambito ETS, asi que se le
        # aplica la mediana de la flota y se dice, en vez de dar una cifra en euros como si fuera
        # suya.
        factor_ambito = np.array([ESTADO["ets"]["mediana"]])
        # Sin IMO tampoco hay penalizacion de FuelEU que reducir: se devuelve cero, no una invencion.
        mascara_fila = np.array([], dtype=int)
        avisos.append(
            "El coste ETS usa la fracción mediana de la flota dentro del ámbito "
            f"({ESTADO['ets']['mediana']:.2f}) porque este buque no se ha identificado por IMO."
        )

    familia = peticion.familia or ESTADO["familia_por_defecto"]
    _familias_a_usar(familia)
    base = float(sim.predecir(ESTADO["modelos"], fila, familia)[0])

    resultados = []
    for escenario in peticion.escenarios:
        modificada, av = sim.aplicar_escenario(fila, escenario.delta_distancia, escenario.delta_velocidad)
        nueva = float(sim.predecir(ESTADO["modelos"], modificada, familia)[0])
        resultados.append(
            {
                "etiqueta": escenario.etiqueta
                or f"distancia {escenario.delta_distancia:+.0%}, velocidad {escenario.delta_velocidad:+.0%}",
                "delta_distancia": escenario.delta_distancia,
                "delta_velocidad": escenario.delta_velocidad,
                "horas_resultantes": round(float(modificada[sim.COL_HORAS].iloc[0]), 1),
                "velocidad_resultante_nudos": (
                    None
                    if not np.isfinite(modificada[sim.COL_VELOCIDAD].iloc[0])
                    else round(float(modificada[sim.COL_VELOCIDAD].iloc[0]), 2)
                ),
                "co2_escenario_t": round(nueva, 1),
                "ahorro_t": round(base - nueva, 1),
                "ahorro_pct": round(100.0 * (base - nueva) / base, 2) if base else None,
                "horas_recortadas_al_maximo_fisico": av["horas_recortadas_al_maximo"],
                "coste_ets": _coste_ets(
                    np.array([base - nueva]), factor_ambito, peticion.anio_regimen, peticion.precio_eua
                ),
                "coste_fueleu": _coste_fueleu(
                    np.array([base - nueva]), np.array([base]),
                    _penalizacion_fueleu(mascara_fila, peticion.anio_regimen), peticion.anio_regimen
                ),
            }
        )

    return {
        "buque": identificacion,
        "co2_base_predicho_t": round(base, 1),
        "familia": familia,
        "escenarios": resultados,
        "fiabilidad": _aviso_fiabilidad(base),
        "avisos": avisos,
    }


# ==================================================================================================
# Endpoints de flota
# ==================================================================================================


@app.post("/simular/flota", tags=["flota"], summary="Ahorro agregado de un escenario sobre la flota")
def simular_flota(peticion: PeticionFlota) -> dict:
    """
    Aplica un escenario a toda una flota (opcionalmente filtrada por ano y tipo) y devuelve el
    ahorro agregado con su banda de incertidumbre.

    La elasticidad implicita que se devuelve es el numero interpretable: cuanto baja el CO2 por
    cada punto porcentual de millas recortado. No es 1,00 porque hay emisiones que no dependen de
    las millas navegadas, y el modelo lo aprendio del dato.
    """
    flota, mascara = _seleccionar_flota(peticion.anio, peticion.tipos_buque)
    familia_pedida = peticion.familia or ESTADO["familia_por_defecto"]
    familias = _familias_a_usar(peticion.familia, peticion.banda)

    base_cache = {f: _predicciones_base(f)[mascara] for f in familias}
    resultado = sim.simular(
        ESTADO["modelos"],
        flota,
        delta_distancia=peticion.delta_distancia,
        delta_velocidad=peticion.delta_velocidad,
        familias=familias,
        prediccion_base=base_cache,
    )

    familia = familia_pedida if familia_pedida in familias else familias[0]
    metricas = resultado["por_familia"][familia]

    salida: dict = {
        "flota": {
            "anio": peticion.anio,
            "tipos_buque": peticion.tipos_buque or "todos",
            "n_buques_anio": int(len(flota)),
        },
        "escenario": {
            "delta_distancia": peticion.delta_distancia,
            "delta_velocidad": peticion.delta_velocidad,
        },
        "co2_base_Mt": round(metricas["co2_base_t"] / 1e6, 3),
        "co2_escenario_Mt": round(metricas["co2_escenario_t"] / 1e6, 3),
        "ahorro_Mt": round(metricas["ahorro_t"] / 1e6, 3),
        "ahorro_pct": round(metricas["ahorro_pct"], 3),
        "familia": familia,
        "banda_Mt": None,
        "horas_evitadas": round(
            resultado["avisos"]["horas_totales_base"] - resultado["avisos"]["horas_totales_escenario"], 0
        ),
        "avisos_de_calculo": resultado["avisos"],
    }

    ahorro_fila_ets = (
        resultado["predicciones"][familia]["base"] - resultado["predicciones"][familia]["escenario"]
    )
    if peticion.delta_distancia or peticion.delta_velocidad:
        salida["coste_ets"] = _coste_ets(
            ahorro_fila_ets,
            ESTADO["ets"]["por_fila"][mascara],
            peticion.anio_regimen,
            peticion.precio_eua,
        )
        salida["coste_fueleu"] = _coste_fueleu(
            ahorro_fila_ets,
            resultado["predicciones"][familia]["base"],
            _penalizacion_fueleu(mascara, peticion.anio_regimen),
            peticion.anio_regimen,
        )
        salida["cumplimiento_evitado_millones_eur"] = round(
            salida["coste_ets"]["ahorro_millones_eur"]
            + salida["coste_fueleu"]["penalizacion_evitada_millones_eur"],
            1,
        )

    banda = _banda(resultado["por_familia"], "ahorro_t")
    if banda is not None:
        salida["banda_Mt"] = {
            "minimo": round(banda["minimo"] / 1e6, 3),
            "central": round(banda["central"] / 1e6, 3),
            "maximo": round(banda["maximo"] / 1e6, 3),
            "que_mide": banda["que_mide"],
        }

    if peticion.delta_distancia:
        salida["elasticidad_implicita"] = round(
            metricas["ahorro_pct"] / (100.0 * abs(peticion.delta_distancia)), 3
        )

    if peticion.bootstrap:
        ahorro_fila = (
            resultado["predicciones"][familia]["base"] - resultado["predicciones"][familia]["escenario"]
        )
        bajo, alto = sim.bootstrap_flota(ahorro_fila, ESTADO["raw"].loc[mascara, "ship_imo_number"])
        salida["intervalo_95_bootstrap_Mt"] = {
            "minimo": round(bajo / 1e6, 3),
            "maximo": round(alto / 1e6, 3),
            "que_mide": "incertidumbre de muestreo, remuestreando buques completos",
        }

    if peticion.desglose_por_tipo:
        tabla = sim.agregar_por_tipo(resultado, ESTADO["raw"].loc[mascara, "ship_type_agrupado"], familia)
        salida["por_tipo"] = [
            {
                "tipo": indice,
                "n": int(fila["n"]),
                "co2_base_Mt": round(fila["co2_base_t"] / 1e6, 3),
                "ahorro_Mt": round(fila["ahorro_t"] / 1e6, 4),
                "ahorro_pct": round(fila["ahorro_pct"], 2),
            }
            for indice, fila in tabla.iterrows()
        ]

    return salida


@app.post("/simular/focalizado", tags=["flota"], summary="Repartir el esfuerzo o concentrarlo: cuanto cambia")
def simular_focalizado(peticion: PeticionFocalizada) -> dict:
    """
    Compara dos politicas que cuestan **exactamente las mismas horas de navegacion evitadas**:
    repartirlas por igual entre toda la flota, o concentrarlas en los tipos de mayor intensidad.

    Es el resultado de cabecera del TFM. La comparacion se hace a esfuerzo constante justamente
    para que la ventaja no venga de recortar mas, sino de recortar mejor.
    """
    flota, mascara = _seleccionar_flota(peticion.anio, None)
    familia = peticion.familia or ESTADO["familia_por_defecto"]
    _familias_a_usar(familia)

    horas_totales = float(np.nansum(flota[sim.COL_HORAS].to_numpy(dtype="float64")))
    if peticion.horas_a_evitar is not None:
        horas = float(peticion.horas_a_evitar)
    else:
        horas = horas_totales * float(peticion.pct_horas)

    try:
        resultado = sim.escenario_focalizado(
            ESTADO["modelos"],
            flota,
            ESTADO["raw"].loc[mascara, "ship_type_agrupado"],
            horas_a_evitar=horas,
            tipos_objetivo=peticion.tipos_objetivo,
            familia=familia,
            prediccion_base={familia: _predicciones_base(familia)[mascara]},
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    factor_ambito = ESTADO["ets"]["por_fila"][mascara]

    def _coste_de(clave: str) -> dict:
        predicciones = resultado[clave]["predicciones"][familia]
        return _coste_ets(
            predicciones["base"] - predicciones["escenario"],
            factor_ambito,
            peticion.anio_regimen,
            peticion.precio_eua,
        )

    def _fueleu_de(clave: str) -> dict:
        predicciones = resultado[clave]["predicciones"][familia]
        return _coste_fueleu(
            predicciones["base"] - predicciones["escenario"],
            predicciones["base"],
            _penalizacion_fueleu(mascara, peticion.anio_regimen),
            peticion.anio_regimen,
        )

    coste_uniforme = _coste_de("resultado_uniforme")
    coste_focalizado = _coste_de("resultado_focalizado")
    fueleu_uniforme = _fueleu_de("resultado_uniforme")
    fueleu_focalizado = _fueleu_de("resultado_focalizado")

    return {
        "anio": peticion.anio,
        "familia": familia,
        "esfuerzo": {
            "horas_a_evitar": round(resultado["horas_a_evitar"], 0),
            "pct_de_las_horas_de_flota": round(100.0 * resultado["horas_a_evitar"] / horas_totales, 2),
            "horas_totales_flota": round(resultado["horas_totales_flota"], 0),
            "horas_de_los_tipos_objetivo": round(resultado["horas_totales_objetivo"], 0),
        },
        "tipos_objetivo": resultado["tipos_objetivo"],
        "politica_uniforme": {
            "recorte_pct": round(resultado["recorte_uniforme_pct"], 2),
            "ahorro_Mt": round(resultado["ahorro_uniforme_t"] / 1e6, 3),
            "coste_ets": coste_uniforme,
            "coste_fueleu": fueleu_uniforme,
        },
        "politica_focalizada": {
            "recorte_pct_en_los_tipos_objetivo": round(resultado["recorte_focalizado_pct"], 2),
            "ahorro_Mt": round(resultado["ahorro_focalizado_t"] / 1e6, 3),
            "coste_ets": coste_focalizado,
            "coste_fueleu": fueleu_focalizado,
        },
        "ventaja_de_focalizar": {
            "Mt_adicionales": round(resultado["ventaja_focalizado_t"] / 1e6, 3),
            "pct": round(resultado["ventaja_focalizado_pct"], 1),
            "millones_eur_adicionales": round(
                coste_focalizado["ahorro_millones_eur"] - coste_uniforme["ahorro_millones_eur"], 1
            ),
        },
        "aviso_extrapolacion": (
            "El recorte que exige la política focalizada supera el 25% de las millas de esos tipos. "
            "Es una extrapolación muy por encima de cualquier variación observada en el histórico: "
            "úsela para comparar políticas, no como previsión."
            if abs(resultado["recorte_focalizado_pct"]) > 25
            else None
        ),
        "lectura": (
            "Con las mismas horas de navegación evitadas, concentrar el esfuerzo en los tipos de "
            "mayor intensidad por hora ahorra "
            f"{resultado['ventaja_focalizado_pct']:.1f}% más CO₂ que repartirlo por igual, "
            f"lo que bajo el régimen de {peticion.anio_regimen} equivale a "
            f"{coste_focalizado['ahorro_millones_eur'] - coste_uniforme['ahorro_millones_eur']:,.0f} "
            "millones de euros más de derechos de emisión que no hay que comprar."
        ),
    }


# ==================================================================================================
# Panel web y busqueda
# ==================================================================================================


@app.get("/", include_in_schema=False)
def panel():
    """
    Panel web del simulador: `web/index.html`.

    Se sirve desde la misma aplicacion que la API a proposito. No es un cliente aparte con su
    propio calculo: es HTML plano que llama por `fetch` a los mismos endpoints publicos que
    usaria cualquier otro consumidor, asi que lo que se ve en pantalla no puede divergir de lo
    que responde el servicio. Sin build, sin dependencias de terceros y sin red: funciona igual
    sin conexion a internet.
    """
    fichero = DIR_WEB / "index.html"
    if not fichero.exists():  # pragma: no cover
        raise HTTPException(status_code=404, detail="Falta web/index.html en el repositorio.")
    # Sin cabecera de cache el navegador reutiliza la copia que ya tiene, asi que una version nueva
    # del panel puede no llegar a verse nunca: el fichero cambia en disco y la pantalla no. Es un
    # fallo silencioso -- nada falla, todo responde 200, y lo que se lee es viejo--. El panel es un unico fichero
    # de 100 KB servido en local: no cachearlo no cuesta nada.
    return FileResponse(
        fichero,
        media_type="text/html",
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/buques/buscar", tags=["un buque"], summary="Localiza un buque por nombre o numero IMO")
def buscar_buques(
    q: str = Query(..., min_length=2, description="Parte del nombre del buque, o su numero IMO"),
    anio: int | None = Query(None, description="Restringe la busqueda a un ejercicio"),
    limite: int = Query(10, ge=1, le=50),
) -> dict:
    """
    Busqueda sobre los buques de THETIS-MRV, ordenada por emision descendente.

    Existe para que el panel pueda partir de un nombre real en vez
    de exigir que el usuario se sepa un numero IMO de memoria. Devuelve tambien si el buque estuvo
    en el entrenamiento, que es el dato que convierte un ejemplo cualquiera en un ejemplo
    defendible.
    """
    raw = ESTADO["raw"]
    texto = q.strip().lower()

    coincide = raw["ship_name"].str.lower().str.contains(texto, na=False, regex=False)
    coincide |= raw["ship_imo_number"].astype(str).str.startswith(texto)
    if anio is not None:
        coincide &= raw["reporting_year"] == anio

    encontrados = raw.loc[coincide].nlargest(limite, sim.COL_TARGET)
    return {
        "consulta": q,
        "n": int(len(encontrados)),
        "buques": [
            {
                "imo": str(fila["ship_imo_number"]),
                "nombre": str(fila["ship_name"]),
                "tipo": str(fila["ship_type_agrupado"]),
                "anio": int(fila["reporting_year"]),
                "co2_observado_t": round(float(fila[sim.COL_TARGET]), 1),
                "visto_en_entrenamiento": bool(fila["es_train"]),
            }
            for _, fila in encontrados.iterrows()
        ],
    }


# ==================================================================================================
# Capa economica
# ==================================================================================================
#
# Cinco endpoints que traducen las toneladas a euros y a decisiones. Ninguno recalcula nada: leen
# los informes 14-18 que escribe `scripts/analisis_economia.py`, salvo el escenario del pasaporte,
# que se simula en vivo con el mismo `src/simulador.py` que usa todo lo demas.


def _economia_o_503() -> dict:
    """Devuelve el estado de la capa economica o falla con un 503 que explica que hay que hacer."""
    estado = ESTADO.get("economia")
    if estado is None:
        raise HTTPException(
            status_code=503,
            detail=("La capa economica no esta calculada. Ejecuta `python scripts/analisis_economia.py` "
                    "para generar los informes 14-18 y reinicia el servicio."),
        )
    return estado


def _redondear(valor, decimales: int = 1):
    """None si el valor no es finito; asi el JSON nunca lleva NaN, que no es JSON valido."""
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return None
    return round(float(valor), decimales)


@app.get("/economia/parametros", tags=["servicio"], summary="Precios y ratios de la capa economica")
def parametros_economia() -> dict:
    """
    Los parametros de precio, con su banda y su procedencia.

    El precio del combustible es lo que convierte toneladas en euros y **no es un dato del
    proyecto**: es un supuesto. Por eso viaja aqui con su banda y su nota, igual que el derecho de
    emision viaja con su fecha y su fuente. La conclusion que sostiene el trabajo -- el combustible
    vale varias veces mas que el derecho -- se cumple en todo el rango, y eso se puede comprobar
    pidiendo `/economia/ahorro-actividad-constante`.
    """
    return {
        "precio_bunker_eur_t": economia.PRECIO_BUNKER_EUR_POR_T,
        "precio_bunker_banda": list(economia.PRECIO_BUNKER_BANDA),
        "precio_bunker_nota": economia.PRECIO_BUNKER_NOTA,
        "precio_eua_eur_t": ets.PRECIO_EUA_POR_DEFECTO,
        "precio_eua_fecha": ets.PRECIO_EUA_FECHA,
        "ratio_co2_combustible": economia.RATIO_CO2_FUEL,
        "toneladas_de_fuel_por_tonelada_de_co2": round(1 / economia.RATIO_CO2_FUEL, 4),
        "capa_calculada": ESTADO.get("economia") is not None,
    }


@app.get("/economia/ahorro-actividad-constante", tags=["flota"],
         summary="Cuanto se puede ahorrar sin recortar una sola milla")
def ahorro_actividad_constante(
    frontera: Literal["mediana", "mejor_cuartil"] = Query("mediana"),
    precio_bunker: float | None = Query(None, ge=100, le=1500,
                                        description="EUR/t; por defecto, el parametro del modulo"),
) -> dict:
    """
    El ahorro que se consigue **operando como los pares**, no navegando menos.

    Responde a la objecion natural al escenario del simulador: recortar un 10% de las millas es
    transportar un 10% menos. Aqui las millas, las horas y la velocidad de cada buque se quedan
    como estan; lo unico que cambia es que los buques peores de cada grupo -- mismo tipo y mismo
    quintil de tamano -- pasen a emitir como la mediana (o como el mejor cuartil) de sus pares.

    Se devuelven **dos versiones a proposito**. La `ingenua` usa el residuo del modelo tal cual, y
    su numero es en parte mecanico: el procedimiento produce ahorro incluso sobre ruido. La
    `persistente` retiene solo la parte del residuo que se repite ano tras ano en el mismo buque, y
    es la que se publica. La diferencia entre las dos se muestra explicitamente.
    """
    est = _economia_o_503()
    precio = precio_bunker if precio_bunker is not None else economia.PRECIO_BUNKER_EUR_POR_T
    salida = {}
    for residuo in ("ingenuo", "persistente"):
        fila = est["ahorro"][(est["ahorro"].frontera == frontera)
                             & (est["ahorro"].residuo == residuo)].iloc[0]
        combustible = float(fila.combustible_t) * precio
        salida[residuo] = {
            "ahorro_t": _redondear(fila.ahorro_t),
            "ahorro_mt": _redondear(fila.ahorro_t / 1e6, 3),
            "pct_emisiones_flota": _redondear(fila.pct_emisiones, 2),
            "buques_afectados": int(fila.buques),
            "combustible_evitado_t": _redondear(fila.combustible_t),
            "valor_combustible_eur": _redondear(combustible),
            "valor_ets_eur": _redondear(fila.ets_eur),
            "valor_fueleu_eur": _redondear(fila.fueleu_eur),
            "valor_total_eur": _redondear(combustible + fila.ets_eur + fila.fueleu_eur),
            "eur_por_t_cumplimiento": _redondear(fila.eur_por_t_cumplimiento, 2),
            "eur_por_t_combustible": _redondear(fila.combustible_t * precio / fila.ahorro_t, 2),
        }
    persistencia = est["persistencia"]
    return {
        "frontera": frontera,
        "precio_bunker_eur_t": precio,
        "precio_eua_eur_t": ets.PRECIO_EUA_POR_DEFECTO,
        "resultado": salida,
        "recorte_por_persistencia_pct": _redondear(
            100 * (1 - salida["persistente"]["ahorro_t"] / salida["ingenuo"]["ahorro_t"]), 1),
        "persistencia_del_residuo": {
            "correlacion_con_el_anio_anterior": _redondear(persistencia["corr_lag1_pearson"], 3),
            "correlacion_spearman": _redondear(persistencia["corr_lag1_spearman"], 3),
            "icc": _redondear(persistencia["icc"], 3),
            "pares_buque_anio": int(persistencia["pares_consecutivos"]),
        },
        "aviso": ("Cota superior de lo alcanzable, no dinero garantizado: parte del hueco hasta la "
                  "frontera es heterogeneidad no medida (rutas, meteorologia, espera en puerto, "
                  "carga real) y no ineficiencia."),
    }


@app.get("/economia/palancas", tags=["flota"], summary="Cuanto vale un dia de mar evitado, por tipo")
def palancas_economicas(
    palanca: Literal["millas_-10pct", "velocidad_-10pct"] = Query("millas_-10pct"),
) -> dict:
    """
    El precio de equilibrio de cada palanca operativa, por tipo de buque.

    En vez de suponer cuanto gana un buque al dia -- que es el dato que no se tiene-- se calcula
    **cuanto vale el dia de mar evitado**, y quien decide lo compara con lo que el ya sabe que gana.
    Por debajo de esa cifra, recortar millas sale a cuenta sin que nadie pague por el CO2.

    En la palanca de velocidad el signo se invierte y hay que leerlo al reves: navegar mas despacio
    **anade** dias de mar, asi que la cifra es lo maximo que puede costar cada dia extra para que la
    palanca siga compensando.
    """
    est = _economia_o_503()
    d = est["palancas"][est["palancas"].palanca == palanca].copy()
    d["eur_por_dia_mar"] = d.valor_eur / (d.horas_evitadas / 24)
    total_dias = float(d.horas_evitadas.sum()) / 24
    return {
        "palanca": palanca,
        "anade_dias_de_mar": bool(total_dias < 0),
        "ahorro_mt": _redondear(d.ahorro_t.sum() / 1e6, 3),
        "dias_de_mar": _redondear(total_dias),
        "valor_total_eur": _redondear(d.valor_eur.sum()),
        "eur_por_dia_mar_flota": _redondear(d.valor_eur.sum() / total_dias if total_dias else np.nan),
        "por_tipo": [
            {
                "tipo": fila.tipo,
                "ahorro_t": _redondear(fila.ahorro_t),
                "eur_por_dia_mar": _redondear(fila.eur_por_dia_mar),
            }
            for fila in d.sort_values("eur_por_dia_mar", ascending=False).itertuples()
        ],
    }


@app.get("/economia/navieras", tags=["flota"], summary="La factura de cumplimiento, por naviera")
def navieras(limite: int = Query(20, ge=1, le=200)) -> dict:
    """
    Quien paga los 8.237 M EUR, con que concentracion, y cuanto puede compensar por dentro.

    El agregado por naviera sale del numero IMO de compania que publica el propio MRV. Identifica a
    la **entidad gestora**, no al grupo empresarial, asi que un grupo grande aparece repartido en
    varias filas; se declara y no se corrige a mano.
    """
    est = _economia_o_503()
    n = est["navieras"]
    total = float(n.factura_eur.sum())
    acumulada = n.cuota_acumulada_pct
    pool = est["pooling"]
    return {
        "navieras": int(len(n)),
        "factura_total_eur": _redondear(total),
        "navieras_que_pagan_la_mitad": int((acumulada < 50).sum()) + 1,
        "cuota_top10_pct": _redondear(acumulada.iloc[9], 1),
        "cuota_top20_pct": _redondear(acumulada.iloc[19], 1),
        "navieras_de_un_solo_buque": int((n.buques == 1).sum()),
        "pooling_fueleu": {
            "companias_con_deficit": int(pool["companias_con_deficit"]),
            "pct_con_algun_buque_excedentario": _redondear(pool["pct_companias_con_excedente"], 1),
            "pct_deficit_compensable_por_dentro": _redondear(pool["pct_deficit_neteable"], 1),
            "penalizacion_sin_pooling_eur": _redondear(pool["penalizacion_sin_pooling_eur"]),
            "penalizacion_con_pooling_eur": _redondear(pool["penalizacion_con_pooling_eur"]),
        },
        "ranking": [
            {
                "nombre": str(fila.nombre),
                "buques": int(fila.buques),
                "factura_eur": _redondear(fila.factura_eur),
                "cuota_pct": _redondear(fila.cuota_pct, 2),
                "ahorro_alcanzable_t": _redondear(fila.ahorro_t),
            }
            for fila in n.head(limite).itertuples()
        ],
        "aviso": ("El IMO de compania identifica a la entidad gestora, no al grupo: MSC, por ejemplo, "
                  "aparece en varias filas."),
    }


@app.get("/buques/{imo}/pasaporte", tags=["un buque"],
         summary="Pasaporte de cumplimiento: banda, factura, deficit y que hacer")
def pasaporte(
    imo: str,
    delta_distancia: float = Query(-0.10, gt=-1.0, le=1.0),
    delta_velocidad: float = Query(0.0, gt=-1.0, le=1.0),
) -> dict:
    """
    Todo lo que el trabajo sabe de un buque, en la unidad en la que su armador decide.

    Cinco bloques: **prediccion** (y su contraste con lo declarado), **CII** (banda de hoy,
    trayectoria hasta 2030 y el ano en que cae a D/E si sigue operando igual), **ETS** (emision
    dentro de ambito y factura al precio del dia), **FuelEU** (intensidad, si cumple y cuanto le
    costaria) y **palancas** (lo que ya podria ahorrar operando como sus pares, y lo que le hace un
    escenario operativo a las cuatro cosas anteriores).

    El escenario se simula en vivo con `src/simulador.py`; el resto viene de los informes. La banda
    del escenario se recalcula con la formula de la OMI sobre el CO2 y la distancia nuevos, que es
    justo el efecto que un armador quiere ver antes de decidir.
    """
    est = _economia_o_503()
    buques = est["buques"]
    imo = str(imo).strip()
    if imo not in buques.index:
        raise HTTPException(status_code=404,
                            detail=f"El buque {imo!r} no esta en el ejercicio 2025 del registro MRV.")
    b = buques.loc[imo]
    if isinstance(b, pd.DataFrame):
        b = b.iloc[0]

    anios = [a for a in range(2025, 2031)]
    trayectoria = {str(a): (None if pd.isna(b.get(f"banda_{a}")) else str(b[f"banda_{a}"])) for a in anios}
    # Un CII vacio no es un fallo: hay tipos que la formula de la OMI califica sobre arqueo bruto,
    # que el MRV no publica. Decirlo aqui evita que el panel ensene un hueco sin explicacion.
    motivo_sin_cii = None
    if trayectoria["2025"] is None:
        motivo_sin_cii = (
            "El CII de este tipo de buque (ro-ro, ro-pax, crucero y car carrier) se calcula sobre "
            "arqueo bruto, que el registro MRV no publica."
            if str(b.ship_type_agrupado) in {"Ro-ro ship", "Ro-pax ship", "Passenger ship", "Vehicle carrier"}
            else "Sin capacidad en unidades de peso muerto o sin distancia declarada: no es calificable.")

    # --- escenario, simulado en vivo sobre la fila real del buque
    ml = ESTADO["ml"]
    fila = ml[(ml["ship_imo_number"].astype(str) == imo) & (ml["reporting_year"] == 2025)]
    escenario = None
    if len(fila) and (delta_distancia or delta_velocidad):
        modelos = ESTADO["modelos"]
        base = float(sim.predecir(modelos, fila)[0])
        cambiada, avisos = sim.aplicar_escenario(fila, delta_distancia, delta_velocidad)
        nueva = float(sim.predecir(modelos, cambiada)[0])
        ahorro = base - nueva
        factor = float(b.factor_ambito) if pd.notna(b.factor_ambito) else 0.566
        combustible = ahorro / economia.RATIO_CO2_FUEL
        banda_escenario = None
        if pd.notna(b.get("cii_tipo")) and pd.notna(b.get("cii_capacidad")) and pd.notna(b.get("distancia_nm")):
            distancia = float(b.distancia_nm) * (1 + delta_distancia)
            if distancia > 0:
                attained = nueva * 1e6 / (float(b.cii_capacidad) * distancia)
                requerido = cii.cii_requerido(str(b.cii_tipo), float(b.cii_capacidad), 2026)
                banda_escenario = cii.banda(attained, requerido, str(b.cii_tipo), float(b.cii_capacidad))
        escenario = {
            "delta_distancia": delta_distancia,
            "delta_velocidad": delta_velocidad,
            "co2_base_t": _redondear(base),
            "co2_escenario_t": _redondear(nueva),
            "ahorro_t": _redondear(ahorro),
            "horas_evitadas": _redondear(avisos["horas_totales_base"] - avisos["horas_totales_escenario"]),
            "ahorro_ets_eur": _redondear(ahorro * factor * ets.PRECIO_EUA_POR_DEFECTO),
            "ahorro_combustible_eur": _redondear(combustible * economia.PRECIO_BUNKER_EUR_POR_T),
            "banda_cii_2026_con_escenario": banda_escenario,
            "banda_cii_2026_sin_escenario": trayectoria.get("2026"),
        }

    return {
        "imo": imo,
        "nombre": str(b.ship_name),
        "compania": str(b.company_name),
        "tipo": str(b.ship_type_agrupado),
        "anio": int(b.reporting_year),
        "capacidad_estimada_t": _redondear(b.capacidad_estimada),
        "prediccion": {
            "co2_declarado_t": _redondear(b[sim.COL_TARGET]),
            "co2_predicho_t": _redondear(b.pred_honesta),
            "prediccion_fuera_de_muestra": str(b.origen) == "test",
            "residuo_relativo_pct": _redondear(100 * b.residuo_rel, 1),
            "residuo_persistente_pct": _redondear(100 * b.residuo_persistente, 1),
            "grupo_de_comparacion": str(b.grupo),
        },
        "cii": {
            "banda_2025": trayectoria.get("2025"),
            "trayectoria": trayectoria,
            "anio_de_caida_a_de": None if pd.isna(b.anio_caida) else int(b.anio_caida),
            "attained": _redondear(b.cii_attained, 3),
            "requerido_2025": _redondear(b.cii_requerido, 3),
            "no_calificable_porque": motivo_sin_cii,
        },
        "ets": {
            "fraccion_dentro_de_ambito": _redondear(b.factor_ambito, 3),
            "factura_2026_eur": _redondear(b.ets_2026_eur),
        },
        "fueleu": {
            "intensidad_wtw_g_mj": _redondear(b.ghgie_wtw, 2),
            "objetivo_2025_g_mj": _redondear(b.objetivo_fueleu, 2),
            "cumple": None if pd.isna(b.cumple_fueleu) else bool(b.cumple_fueleu),
            "penalizacion_eur": _redondear(b.penalizacion_fueleu_eur),
        },
        "palancas": {
            "ahorro_a_actividad_constante_t": _redondear(b.ahorro_actividad_constante_t),
            "ahorro_a_actividad_constante_eur": _redondear(b.ahorro_eur),
            "escenario": escenario,
        },
    }


# ==================================================================================================
# Capa de gestion de flota
# ==================================================================================================
#
# Tres endpoints que cambian el sujeto de la pregunta. Todo lo anterior responde "que le pasa a la
# flota europea" o "que le pasa a este buque"; un gestor de flota no manda sobre ninguna de las dos
# cosas: manda sobre **sus** buques. Estos tres endpoints recortan el registro por compania y
# devuelven la misma informacion que ya calculan las capas regulatoria y economica, agregada a esa
# unidad y ordenada por lo unico que importa cuando hay que decidir manana: cuanto se puede ahorrar
# en cada buque y cual esta a punto de caer de banda.
#
# No calculan nada nuevo. `/navieras/buscar` y `/navieras/{cid}/flota` leen el pasaporte que escribe
# `scripts/analisis_economia.py`; `/navieras/{cid}/simular` llama al mismo `src/simulador.py` que el
# cuaderno 08, con una mascara de filas en vez de la flota entera. Por eso la suma de los escenarios
# de todas las navieras reproduce el escenario de flota: es literalmente la misma funcion.


# Identificador que `scripts/analisis_economia.py` asigna a las companias sin IMO propio.
PREFIJO_SIN_ID = "SIN_ID:"


def _cid_serie(serie: pd.Series) -> pd.Series:
    """
    Numero IMO de compania en forma canonica: solo digitos, sin ceros a la izquierda.

    Los dos informes que hay que cruzar lo guardan de forma distinta, y es un desajuste que pasaria
    inadvertido: `16_navieras_exposicion`
    lo trae como texto de siete caracteres con ceros por delante ('0750415') y
    `18_pasaporte_buques` como numero en coma flotante (750415.0). Comparados tal cual, ninguna de
    las dos representaciones casa con la otra: la naviera existiria en el ranking y su flota
    saldria vacia, sin que nada fallara. Se normaliza aqui, en un unico sitio, para que el fallo no
    pueda reaparecer en el siguiente endpoint que filtre por compania.
    """
    return (
        serie.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"^0+(?=\d)", "", regex=True)
    )


def _naviera_o_404(cid: str) -> tuple[dict, pd.DataFrame, str]:
    """Estado economico, filas del pasaporte de esa compania y su nombre publicado."""
    est = _economia_o_503()
    buques = est["buques"]
    cid = str(cid).strip()
    if cid.startswith(PREFIJO_SIN_ID):
        # Tres companias del registro (ocho buques) no declaran IMO de compania, y el informe 16 las
        # identifica por su nombre con este prefijo. Sin esta rama existirian en el buscador y
        # devolverian un 404 al abrirlas, que es el peor de los dos fallos posibles.
        nombre_buscado = cid[len(PREFIJO_SIN_ID):].strip().lower()
        seleccion = buques.loc[
            buques["company_imo_number"].isna()
            & (buques["company_name"].astype(str).str.strip().str.lower() == nombre_buscado)
        ]
    else:
        seleccion = buques.loc[_cid_serie(buques["company_imo_number"]) == cid]
    if not len(seleccion):
        raise HTTPException(
            status_code=404,
            detail=f"No hay ninguna compania con IMO {cid!r} en el ejercicio 2025 del registro MRV.",
        )
    # El nombre normalizado vive en el informe 16, que es el que agrega por compania; el del
    # pasaporte es el que declara cada buque y puede variar de fila a fila dentro de la misma.
    navieras = est["navieras"]
    fila = navieras.loc[_cid_serie(navieras["_cid"]) == cid]
    nombre = str(fila["nombre"].iloc[0]) if len(fila) else str(seleccion["company_name"].iloc[0])
    return est, seleccion, nombre


def _riesgo_cii(banda: str | None, anio_caida) -> tuple[str, int]:
    """
    Traduce la calificacion CII a una etiqueta accionable y a un orden.

    La banda por si sola no dice que hacer: un buque en C que cae a D el ano que viene es un
    problema mas urgente que uno en D que lleva ahi desde 2023 y ya tiene plan. Por eso el
    semaforo mezcla la banda de hoy con el ano de caida, que es justo la informacion que el
    armador no tiene en ningun sitio publico.
    """
    if banda in ("D", "E"):
        return "incumple", 0
    if anio_caida is not None and int(anio_caida) <= 2027:
        return "cae pronto", 1
    if anio_caida is not None:
        return "cae en " + str(int(anio_caida)), 2
    if banda in ("A", "B", "C"):
        return "cumple", 3
    return "no calificable", 4


@app.get("/navieras/buscar", tags=["naviera"], summary="Localiza una naviera por nombre o IMO de compania")
def buscar_navieras(
    q: str | None = Query(None, min_length=2, description="Parte del nombre de la naviera, o su IMO de compania"),
    limite: int = Query(12, ge=1, le=50),
) -> dict:
    """
    Busqueda sobre las companias del registro, ordenada por factura de cumplimiento descendente.

    Sin `q` devuelve las mayores, que es lo que el panel necesita para no arrancar con la pantalla
    vacia: un panel que exige escribir algo antes de ensenar nada obliga a saber que escribir.
    """
    est = _economia_o_503()
    n = est["navieras"].copy()
    n["cid_txt"] = _cid_serie(n["_cid"])

    if q:
        texto = q.strip().lower()
        coincide = n["nombre"].astype(str).str.lower().str.contains(texto, na=False, regex=False)
        coincide |= n["cid_txt"].str.startswith(texto)
        n = n.loc[coincide]

    total = float(est["navieras"]["factura_eur"].sum())
    return {
        "consulta": q,
        "n": int(len(n)),
        "factura_total_flota_eur": _redondear(total),
        "navieras": [
            {
                "cid": str(fila.cid_txt),
                "nombre": str(fila.nombre),
                "buques": int(fila.buques),
                "co2_t": _redondear(fila.co2_t),
                "factura_eur": _redondear(fila.factura_eur),
                "cuota_pct": _redondear(fila.cuota_pct, 3),
                "ahorro_alcanzable_t": _redondear(fila.ahorro_t),
            }
            for fila in n.head(limite).itertuples()
        ],
    }


@app.get("/navieras/{cid}/flota", tags=["naviera"], summary="La cartera de buques de una naviera, priorizada")
def flota_naviera(
    cid: str,
    orden: Literal["ahorro", "factura", "riesgo", "co2"] = Query("ahorro"),
    limite: int = Query(60, ge=1, le=500),
) -> dict:
    """
    Todos los buques de una compania con lo que el trabajo sabe de cada uno, ordenados por donde
    hay mas que ganar.

    El orden por defecto **no** es el tamano ni la factura: es el ahorro alcanzable a actividad
    constante, es decir, lo que cada buque dejaria de gastar operando como la mediana de sus pares
    sin recortar una sola milla. Es el unico de los cuatro criterios que responde a "por donde
    empiezo", y por eso es el que sale primero sin que haya que elegirlo.
    """
    est, seleccion, nombre = _naviera_o_404(cid)

    filas = []
    for b in seleccion.itertuples():
        banda = None if pd.isna(getattr(b, "banda_2025", np.nan)) else str(b.banda_2025)
        caida = None if pd.isna(b.anio_caida) else int(b.anio_caida)
        etiqueta, orden_riesgo = _riesgo_cii(banda, caida)
        trayectoria = {
            str(a): (None if pd.isna(getattr(b, f"banda_{a}", np.nan)) else str(getattr(b, f"banda_{a}")))
            for a in range(2025, 2031)
        }
        filas.append(
            {
                "imo": str(b.ship_imo_number),
                "nombre": str(b.ship_name),
                "tipo": str(b.ship_type_agrupado),
                "co2_declarado_t": _redondear(b.total_co2_emissions_m_tonnes),
                "co2_predicho_t": _redondear(b.pred_honesta),
                "horas_mar": _redondear(b.time_spent_at_sea_hours),
                "capacidad_estimada_t": _redondear(b.capacidad_estimada),
                "banda_cii_2025": banda,
                "trayectoria_cii": trayectoria,
                "anio_de_caida_a_de": caida,
                "riesgo": etiqueta,
                "_orden_riesgo": orden_riesgo,
                "factura_ets_2026_eur": _redondear(b.ets_2026_eur),
                "penalizacion_fueleu_eur": _redondear(b.penalizacion_fueleu_eur),
                "cumple_fueleu": None if pd.isna(b.cumple_fueleu) else bool(b.cumple_fueleu),
                "ahorro_alcanzable_t": _redondear(b.ahorro_actividad_constante_t),
                "ahorro_alcanzable_eur": _redondear(b.ahorro_eur),
                "grupo_de_comparacion": str(b.grupo),
            }
        )

    claves = {
        "ahorro": lambda f: -(f["ahorro_alcanzable_eur"] or 0.0),
        "factura": lambda f: -((f["factura_ets_2026_eur"] or 0.0) + (f["penalizacion_fueleu_eur"] or 0.0)),
        "co2": lambda f: -(f["co2_declarado_t"] or 0.0),
        "riesgo": lambda f: (f["_orden_riesgo"], -(f["co2_declarado_t"] or 0.0)),
    }
    filas.sort(key=claves[orden])

    ets_total = float(seleccion["ets_2026_eur"].fillna(0).sum())
    fueleu_total = float(seleccion["penalizacion_fueleu_eur"].fillna(0).sum())
    en_riesgo = [f for f in filas if f["_orden_riesgo"] <= 1]
    calificables = [f for f in filas if f["banda_cii_2025"]]

    for f in filas:
        f.pop("_orden_riesgo")

    return {
        "cid": str(cid),
        "nombre": nombre,
        "anio": 2025,
        "resumen": {
            "buques": int(len(seleccion)),
            "co2_declarado_t": _redondear(seleccion["total_co2_emissions_m_tonnes"].sum()),
            "horas_mar": _redondear(seleccion["time_spent_at_sea_hours"].sum()),
            "factura_ets_2026_eur": _redondear(ets_total),
            "penalizacion_fueleu_eur": _redondear(fueleu_total),
            "factura_total_eur": _redondear(ets_total + fueleu_total),
            "ahorro_alcanzable_t": _redondear(seleccion["ahorro_actividad_constante_t"].sum()),
            "ahorro_alcanzable_eur": _redondear(seleccion["ahorro_eur"].sum()),
            "buques_en_riesgo_cii": int(len(en_riesgo)),
            "buques_calificables_cii": int(len(calificables)),
            "buques_que_incumplen_fueleu": int((seleccion["cumple_fueleu"] == False).sum()),  # noqa: E712
        },
        "buques": filas[:limite],
        "aviso": (
            "El IMO de compania identifica a la entidad gestora, no al grupo empresarial: un grupo "
            "grande aparece repartido en varias fichas. Las cifras son del ejercicio 2025, que es el "
            "ultimo publicado por EMSA."
        ),
    }


class PeticionNaviera(BaseModel):
    """Un escenario operativo aplicado solo a los buques de una compania."""

    delta_distancia: float = Field(-0.10, gt=-1, le=1, description="-0.10 = recorrer un 10% menos de millas")
    delta_velocidad: float = Field(0.0, gt=-1, le=1, description="-0.10 = navegar un 10% mas despacio")
    familia: Literal["xgb", "red", "mezcla"] | None = None
    anio_regimen: int = Field(2026, ge=2024, le=2050)
    precio_eua: float | None = Field(None, gt=0, le=1000)
    limite_buques: int = Field(12, ge=1, le=200, description="Cuantos buques devolver en el detalle")


@app.post("/navieras/{cid}/simular", tags=["naviera"], summary="Efecto de un escenario sobre la flota de una naviera")
def simular_naviera(cid: str, peticion: PeticionNaviera) -> dict:
    """
    Aplica un escenario a los buques de una compania y devuelve el ahorro agregado y el de cada buque.

    Es el mismo `sim.simular()` del cuaderno 08 con una mascara de filas: si se llamara una vez por
    cada compania y se sumaran los resultados saldria exactamente el escenario de flota, porque no
    hay ninguna formula distinta en medio. Por eso el desglose por buque se puede leer como reparto
    del agregado y no como una segunda estimacion.
    """
    est, seleccion, nombre = _naviera_o_404(cid)

    # Los dos lados del cruce se normalizan igual. El pasaporte lee el IMO como texto y el parquet
    # puede traerlo como entero o como flotante segun la version de pandas: sin esta normalizacion,
    # un '.0' de mas dejaria la mascara vacia y toda naviera responderia 404 sin que nada fallara.
    def _imo_txt(serie: pd.Series) -> pd.Series:
        return serie.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    imos = set(_imo_txt(seleccion["ship_imo_number"]))
    raw = ESTADO["raw"]
    mascara = (
        _imo_txt(raw["ship_imo_number"]).isin(imos) & (raw["reporting_year"] == 2025)
    ).to_numpy()
    if not mascara.any():
        raise HTTPException(
            status_code=404,
            detail=f"Los buques de {nombre!r} no tienen fila de 2025 con predictores completos.",
        )

    familia = peticion.familia or ESTADO["familia_por_defecto"]
    _familias_a_usar(familia)
    flota = ESTADO["ml"].loc[mascara]

    resultado = sim.simular(
        ESTADO["modelos"],
        flota,
        delta_distancia=peticion.delta_distancia,
        delta_velocidad=peticion.delta_velocidad,
        familias=(familia,),
        prediccion_base={familia: _predicciones_base(familia)[mascara]},
    )
    metricas = resultado["por_familia"][familia]
    base = resultado["predicciones"][familia]["base"]
    nueva = resultado["predicciones"][familia]["escenario"]
    ahorro_fila = base - nueva

    coste = _coste_ets(ahorro_fila, ESTADO["ets"]["por_fila"][mascara], peticion.anio_regimen, peticion.precio_eua)
    fuel = _coste_fueleu(
        ahorro_fila, base, _penalizacion_fueleu(mascara, peticion.anio_regimen), peticion.anio_regimen
    )

    # El combustible no es un resultado del ETS: se deriva del CO2 con el mismo ratio que usa la
    # capa economica, y su precio es un supuesto declarado, no un dato del registro.
    combustible_t = float(np.nansum(ahorro_fila)) / economia.RATIO_CO2_FUEL
    combustible_eur = combustible_t * economia.PRECIO_BUNKER_EUR_POR_T

    factores = ESTADO["ets"]["por_fila"][mascara]
    precio_eua = peticion.precio_eua or ets.PRECIO_EUA_POR_DEFECTO
    detalle_raw = ESTADO["raw"].loc[mascara]
    pasaporte = est["buques"]
    detalle = []
    for i, (_, fila) in enumerate(detalle_raw.iterrows()):
        imo = str(fila["ship_imo_number"]).strip().removesuffix(".0")
        p = pasaporte.loc[imo] if imo in pasaporte.index else None
        if isinstance(p, pd.DataFrame):
            p = p.iloc[0]
        detalle.append(
            {
                "imo": imo,
                "nombre": str(fila["ship_name"]),
                "tipo": str(fila["ship_type_agrupado"]),
                "co2_base_t": _redondear(float(base[i])),
                "co2_escenario_t": _redondear(float(nueva[i])),
                "ahorro_t": _redondear(float(ahorro_fila[i])),
                "ahorro_pct": _redondear(100.0 * float(ahorro_fila[i]) / float(base[i]), 2) if base[i] else None,
                "ahorro_eur": _redondear(
                    float(ahorro_fila[i]) / economia.RATIO_CO2_FUEL * economia.PRECIO_BUNKER_EUR_POR_T
                    + float(ahorro_fila[i]) * float(factores[i]) * precio_eua
                ),
                "banda_cii_2025": None if p is None or pd.isna(p.get("banda_2025")) else str(p["banda_2025"]),
            }
        )
    detalle.sort(key=lambda d: -(d["ahorro_t"] or 0.0))

    horas_evitadas = (
        resultado["avisos"]["horas_totales_base"] - resultado["avisos"]["horas_totales_escenario"]
    )

    # Reparto por tipo de buque dentro de la compania. Es la misma funcion que usa el endpoint de
    # flota, asi que el desglose de una naviera y el de Europa se leen en la misma escala.
    tabla_tipos = sim.agregar_por_tipo(
        resultado, ESTADO["raw"].loc[mascara, "ship_type_agrupado"], familia
    )
    por_tipo = [
        {
            "tipo": indice,
            "n": int(f["n"]),
            "co2_base_t": _redondear(f["co2_base_t"]),
            "ahorro_t": _redondear(f["ahorro_t"]),
            "ahorro_pct": _redondear(f["ahorro_pct"], 2),
        }
        for indice, f in tabla_tipos.iterrows()
    ]
    por_tipo.sort(key=lambda t: -(t["ahorro_t"] or 0.0))

    return {
        "cid": str(cid),
        "nombre": nombre,
        "familia": familia,
        "escenario": {
            "delta_distancia": peticion.delta_distancia,
            "delta_velocidad": peticion.delta_velocidad,
        },
        "buques_simulados": int(mascara.sum()),
        "co2_base_t": _redondear(metricas["co2_base_t"]),
        "co2_escenario_t": _redondear(metricas["co2_escenario_t"]),
        "ahorro_t": _redondear(metricas["ahorro_t"]),
        "ahorro_pct": _redondear(metricas["ahorro_pct"], 2),
        "horas_evitadas": _redondear(horas_evitadas, 0),
        "dias_de_mar_evitados": _redondear(horas_evitadas / 24, 0),
        "coste_ets": coste,
        "coste_fueleu": fuel,
        "combustible_evitado_t": _redondear(combustible_t),
        "combustible_evitado_eur": _redondear(combustible_eur),
        "ahorro_total_eur": _redondear(
            combustible_eur + coste["ahorro_eur"] + fuel["penalizacion_evitada_eur"]
        ),
        "precio_combustible_eur_t": economia.PRECIO_BUNKER_EUR_POR_T,
        "avisos_de_calculo": resultado["avisos"],
        "por_tipo": por_tipo,
        "buques": detalle[: peticion.limite_buques],
        "nota": (
            "El ahorro se mide predicho contra predicho sobre la misma fila del registro, nunca "
            "contra el dato declarado: asi el error del modelo en cada buque se cancela al restar."
        ),
    }


# ==================================================================================================
# Capa de explicacion y geografia
# ==================================================================================================
#
# El trabajo declara tres objetivos -- predecir, **explicar** y simular--. Estos tres endpoints
# sirven el segundo: la interpretabilidad del notebook 07 y el analisis de pabellon, que de otro
# modo solo estarian en cuadernos y figuras PNG, se ponen en linea para el panel.
#
# Ninguno calcula nada nuevo. Leen los informes que ya escriben los cuadernos 04, 07 y el script de
# pabellon, salvo la serie anual, que se agrega en vivo con las mismas predicciones cacheadas que
# usa todo lo demas. Si un informe falta, el endpoint responde 503 explicando que hay que ejecutar,
# igual que hace la capa economica, en vez de fallar con un error raro.

_CACHE_INFORMES: dict = {}


def _informe(nombre: str) -> pd.DataFrame:
    """Lee un CSV de `reports/` una sola vez. Falla con un 503 que dice que hay que ejecutar."""
    if nombre not in _CACHE_INFORMES:
        ruta = DIR_INFORMES / nombre
        if not ruta.exists():
            raise HTTPException(
                status_code=503,
                detail=(f"Falta el informe {nombre}. Se genera ejecutando el cuaderno o el script "
                        "que lo produce; el resto del servicio no depende de él."),
            )
        _CACHE_INFORMES[nombre] = pd.read_csv(ruta)
    return _CACHE_INFORMES[nombre]


# Codigos ISO-3166 alfa-3 a alfa-2, para poder componer la bandera como emoji en el cliente sin
# cargar ni un solo archivo de imagen. Solo los paises que aparecen en el registro.
ISO3_A_ISO2 = {
    "LBR": "LR", "MLT": "MT", "MHL": "MH", "PAN": "PA", "ITA": "IT", "SGP": "SG", "PRT": "PT",
    "BHS": "BS", "DNK": "DK", "HKG": "HK", "CYP": "CY", "GRC": "GR", "NOR": "NO", "NLD": "NL",
    "GBR": "GB", "DEU": "DE", "ESP": "ES", "FRA": "FR", "SWE": "SE", "FIN": "FI", "BEL": "BE",
    "IRL": "IE", "POL": "PL", "EST": "EE", "LVA": "LV", "LTU": "LT", "HRV": "HR", "SVN": "SI",
    "BGR": "BG", "ROU": "RO", "ISL": "IS", "LUX": "LU", "AUT": "AT", "CZE": "CZ", "HUN": "HU",
    "SVK": "SK", "CHN": "CN", "JPN": "JP", "KOR": "KR", "USA": "US", "RUS": "RU", "TUR": "TR",
    "IND": "IN", "IDN": "ID", "PHL": "PH", "VNM": "VN", "THA": "TH", "MYS": "MY", "TWN": "TW",
    "ARE": "AE", "SAU": "SA", "QAT": "QA", "KWT": "KW", "EGY": "EG", "MAR": "MA", "DZA": "DZ",
    "TUN": "TN", "NGA": "NG", "ZAF": "ZA", "BRA": "BR", "ARG": "AR", "CHL": "CL", "MEX": "MX",
    "CAN": "CA", "AUS": "AU", "NZL": "NZ", "CHE": "CH", "UKR": "UA", "GEO": "GE", "AZE": "AZ",
    "ATG": "AG", "BRB": "BB", "BLZ": "BZ", "BMU": "BM", "CYM": "KY", "COK": "CK", "CUW": "CW",
    "GIB": "GI", "IMN": "IM", "JAM": "JM", "KNA": "KN", "LCA": "LC", "MDA": "MD", "MNG": "MN",
    "PLW": "PW", "SLE": "SL", "TGO": "TG", "TON": "TO", "TUV": "TV", "VUT": "VU", "VCT": "VC",
    "FRO": "FO", "GGY": "GG", "JEY": "JE", "CUB": "CU", "ISR": "IL", "LBN": "LB", "OMN": "OM",
    "SYC": "SC", "MUS": "MU", "PER": "PE", "COL": "CO", "ECU": "EC", "URY": "UY", "VEN": "VE",
}


@app.get("/explicabilidad", tags=["modelo"], summary="Que explica las emisiones, y cuanto vale cada dato")
def explicabilidad() -> dict:
    """
    El segundo objetivo del TFM, servido por HTTP: **por que** emite lo que emite.

    Tres cosas distintas que conviene no mezclar:

    1. **El reparto de la explicacion (SHAP) en los cuatro niveles de informacion.** El resultado
       que ordena el notebook 07 es que el tipo de buque pasa de explicar el 41,6% al 6,6% en cuanto
       entran la velocidad y el tamano: **era un sustituto de algo que no se habia medido**.
    2. **Lo que vale cada pieza de informacion**, en R2. Las dos variables reconstruidas valen entre
       las dos +0,115 de R2 en validacion cruzada; pasar de Random Forest a XGBoost valia +0,0077.
    3. **El numero accionable**: toneladas de CO2 por hora de navegacion, por tipo de buque. Es la
       cifra que convierte el modelo en una decision.
    """
    shap = _informe("07_niveles_shap.csv")
    niveles = _informe("04_niveles_informacion.csv")
    pendientes = _informe("07_pendiente_horas_por_tipo.csv")

    columnas_nivel = [c for c in shap.columns if c != "variable"]
    return {
        "shap": {
            "niveles": columnas_nivel,
            "variables": [
                {
                    "variable": str(f["variable"]),
                    "por_nivel": {c: (None if pd.isna(f[c]) else float(f[c])) for c in columnas_nivel},
                }
                for _, f in shap.iterrows()
            ],
            "que_mide": (
                "porcentaje de la explicación del modelo que se lleva cada variable, medido con "
                "SHAP sobre el conjunto de test"
            ),
            "aviso": (
                "Todo lo que aporta SHAP está contrastado contra otras dos medidas de "
                "importancia: «total_gain» e importancia por permutación."
            ),
            "lectura": (
                "El tipo de buque pasa del 41,6 % al 6,6 % en cuanto entran la velocidad media y "
                "la capacidad de carga. No explicaba las emisiones: hacía de sustituto del tamaño "
                "y de la operación, que no estaban medidos. Una variable categórica dominante debe "
                "mirarse siempre con sospecha."
            ),
        },
        "niveles": [
            {
                "nivel": str(f["nivel"]),
                "n_variables": int(f["n_variables"]),
                "r2_cv": None if pd.isna(f["R2_cv"]) else round(float(f["R2_cv"]), 4),
                "r2_test": round(float(f["R2_test"]), 4),
                "mae_test": round(float(f["MAE_test"]), 1),
            }
            for _, f in niveles.iterrows()
        ],
        "coste_por_hora": [
            {
                "tipo": str(f["tipo"]),
                "n": int(f["n"]),
                "t_co2_por_hora": round(float(f["t_CO2_por_hora_navegada"]), 3),
            }
            for _, f in pendientes.iterrows()
        ],
        "nota_niveles": (
            "El nivel de control incluye el combustible consumido, que es casi el propio CO₂: sirve "
            "de techo de referencia y no como modelo utilizable, porque nadie que quiera predecir "
            "emisiones conoce de antemano el combustible."
        ),
    }


@app.get("/pabellon/mapa", tags=["flota"], summary="Quien paga la factura europea, y desde que pais")
def pabellon_mapa(limite: int = Query(15, ge=1, le=120)) -> dict:
    """
    La unica dimension geografica que el registro publico permite: el pais de registro del buque.

    Sale de `port_of_registry`, una columna que cubre el 96,4% de las emisiones de 2025. El eje principal es factual -- **dentro o fuera del EEE**-- y la
    marca de *registro abierto* no es un juicio propio: es la lista de pabellones de conveniencia
    que publica la Federacion Internacional de los Trabajadores del Transporte.

    Limitacion que hay que decir en voz alta: el puerto de registro **no es exactamente el
    pabellon**. Coinciden en los grandes registros y pueden no coincidir en casos sueltos; para
    ordenar paises por emisiones y por factura la aproximacion es buena, para afirmar el pabellon de
    un buque concreto, no.
    """
    mapa = _informe("19_pabellon_mapa.csv")
    bloques = _informe("19_pabellon_bloques.csv")

    return {
        "paises": [
            {
                "pais": str(f["pais"]),
                "iso3": str(f["iso3"]),
                "iso2": ISO3_A_ISO2.get(str(f["iso3"])),
                "bloque": str(f["bloque"]),
                "buques": int(f["buques"]),
                "co2_Mt": round(float(f["co2_Mt"]), 3),
                "factura_MEUR": round(float(f["factura_MEUR"]), 1),
                "ahorro_Mt": round(float(f["ahorro_Mt"]), 4),
                "pct_registro_abierto": round(float(f["pct_registro_abierto"]), 1),
                "cuota_pct": round(float(f["cuota_pct"]), 3),
            }
            for _, f in mapa.head(limite).iterrows()
        ],
        "paises_totales": int(len(mapa)),
        "bloques": [
            {
                "bloque": str(f["bloque"]),
                "buques": int(f["buques"]),
                "co2_Mt": round(float(f["co2_t"]) / 1e6, 3),
                "factura_MEUR": round(float(f["factura_eur"]) / 1e6, 1),
                "cuota_factura_pct": round(float(f["cuota_factura_pct"]), 2),
            }
            for _, f in bloques.iterrows()
        ],
        "lectura": (
            "Más de la mitad de la factura de cumplimiento europea la pagan buques registrados "
            "fuera del Espacio Económico Europeo. El régimen grava la escala en puerto europeo, no "
            "la nacionalidad del registro, así que el instrumento alcanza a quien opera en Europa "
            "aunque no esté registrado en Europa."
        ),
        "aviso": (
            "El país sale del puerto de registro declarado, no de un registro de pabellones. La "
            "tabla puerto→país es manual y cubre el 96,4 % de las emisiones de 2025; el resto queda "
            "sin identificar y no se reparte."
        ),
    }


@app.get("/flota/serie", tags=["flota"], summary="Ocho ejercicios de la flota europea, observado y predicho")
def serie_flota(familia: str | None = Query(None)) -> dict:
    """
    La serie anual completa: lo que la flota declaro y lo que el modelo predice, ejercicio a ejercicio.

    Sirve para dos cosas a la vez. La primera es enseñar la profundidad del dato -- ocho ejercicios,
    no una foto--. La segunda es mas incomoda y por eso va tambien: **el sesgo agregado de la linea
    base**. El modelo esta entrenado con ocho anos en los que la intensidad media era mayor, asi que
    sobre los ejercicios recientes predice por encima. No afecta a ninguna cifra de ahorro, que son
    diferencias entre dos predicciones del mismo modelo, pero enseñarlo es mas honesto que no hacerlo.
    """
    familia = familia or ESTADO["familia_por_defecto"]
    _familias_a_usar(familia)

    raw = ESTADO["raw"]
    predicho = _predicciones_base(familia)
    tabla = pd.DataFrame(
        {
            "anio": raw["reporting_year"].to_numpy(),
            "observado": raw[sim.COL_TARGET].to_numpy(dtype="float64"),
            "predicho": np.asarray(predicho, dtype="float64"),
            "horas": raw[sim.COL_HORAS].to_numpy(dtype="float64"),
        }
    )
    agregada = tabla.groupby("anio", observed=True).agg(
        buques=("observado", "size"), observado=("observado", "sum"),
        predicho=("predicho", "sum"), horas=("horas", "sum"),
    )

    return {
        "familia": familia,
        "anios": [
            {
                "anio": int(a),
                "buques": int(f["buques"]),
                "co2_observado_Mt": round(float(f["observado"]) / 1e6, 3),
                "co2_predicho_Mt": round(float(f["predicho"]) / 1e6, 3),
                "sesgo_pct": round(100.0 * (f["predicho"] - f["observado"]) / f["observado"], 2),
                "horas_millones": round(float(f["horas"]) / 1e6, 3),
            }
            for a, f in agregada.iterrows()
        ],
        "nota": (
            "El salto de 2024 no es un repunte de la actividad: es la ampliación del Reglamento "
            "2023/957, que mete en el registro tipos de buque que antes no declaraban. Comparar "
            "2023 con 2024 sin decirlo sería comparar dos poblaciones distintas."
        ),
        "nota_sesgo": (
            "El sesgo es la firma de un modelo entrenado con ocho ejercicios en los que la "
            "intensidad media era mayor. No contamina las cifras de ahorro, que se miden predicho "
            "contra predicho sobre la misma fila."
        ),
    }
