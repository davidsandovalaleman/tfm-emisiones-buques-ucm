"""
Motor del simulador de escenarios operativos de reducción de emisiones (notebook 08).

Este módulo es deliberadamente independiente de los notebooks: la API (`src/api.py`) importa estas
mismas funciones, de modo que el simulador que se enseña en el notebook y el que se sirve por HTTP
son literalmente el mismo código. Es la razón de que aquí no haya ni un `print` ni un gráfico.

--------------------------------------------------------------------------------------------------
QUÉ SE PUEDE SIMULAR Y POR QUÉ, EXACTAMENTE ESTO Y NO OTRA COSA
--------------------------------------------------------------------------------------------------

El motor trabaja por defecto sobre el nivel **operacional+tamaño**, que añade la
capacidad de carga reconstruida en el notebook 03: XGBoost llega ahí a R^2 0,9484 en test, por encima
de la mezcla del nivel operacional anterior (0,9215) y con un solo modelo en vez de dos. El nivel
`"operacional"` sigue disponible en `cargar_modelos` para comparar ambos niveles.

Añadir la capacidad **no añade grados de libertad al simulador**: la capacidad de diseño de un buque
no es una palanca de gestión, así que entra como contexto que mejora la predicción, no como variable
que el escenario mueva. El modelo sigue viendo dos variables movibles a corto plazo:
`time_spent_at_sea_hours` y `velocidad_nudos`. La distancia
recorrida NO es una predictora del modelo, pero sí es la magnitud con sentido de negocio, y las tres
están ligadas por una identidad física exacta:

    distancia (millas náuticas) = horas en el mar x velocidad (nudos)

Es decir: solo hay dos grados de libertad. Por eso un escenario aquí se define con dos porcentajes,
`delta_distancia` y `delta_velocidad`, y las horas se DERIVAN en vez de fijarse a mano:

    distancia' = distancia x (1 + delta_distancia)
    velocidad' = velocidad x (1 + delta_velocidad)
    horas'     = horas x (1 + delta_distancia) / (1 + delta_velocidad)

Fijar las tres a la vez permitiría construir escenarios físicamente imposibles (recorrer las mismas
millas, en menos horas, a la misma velocidad), que el modelo aceptaría sin protestar porque solo ve
dos de ellas. Derivar las horas cierra esa puerta.

Las tres restricciones que se derivan de la interpretabilidad (notebook 07) y que este módulo respeta:

1. Los escenarios se construyen sobre DISTANCIA y HORAS, no sobre velocidad "pura". La elasticidad
   medida dentro de cada buque en el notebook 04 es 0,84, no el 3 de la ley cúbica del slow steaming,
   porque `velocidad_nudos` es una media anual que mezcla navegación con tiempo parado. La palanca
   de velocidad se deja disponible, pero el notebook la usa para MEDIR ese límite, no para
   acreditar ahorros.
2. El efecto de una hora navegada depende del tipo de buque (de 3,02 t CO2/h en carga general a
   6,48 t/h en metaneros, notebook 07). `agregar_por_tipo` y `escenario_focalizado` existen para eso.
3. Las cifras se reportan con banda de incertidumbre, no como número puntual: ver
   `banda_familias` (desacuerdo entre las dos familias de modelos) y `bootstrap_flota`
   (incertidumbre de muestreo del agregado).

--------------------------------------------------------------------------------------------------
CÓMO SE MIDE EL AHORRO: PREDICHO CONTRA PREDICHO
--------------------------------------------------------------------------------------------------

El ahorro de un escenario NUNCA se calcula contra las emisiones observadas, sino contra la
predicción del mismo modelo sobre la fila sin modificar:

    ahorro = f(fila_base) - f(fila_escenario)

Comparar contra el dato observado mezclaría el efecto del escenario con el error del modelo en ese
buque (que en el peor caso del test son 78.868 t, ver notebook 07). Al restar dos predicciones del mismo
modelo, el sesgo sistemático del modelo en ese buque se cancela y queda el efecto del cambio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Máximo físico de horas en el mar en un año natural (mismo criterio que feature_engineering.py).
HORAS_MAX_FISICO = 366 * 24

COL_HORAS = "time_spent_at_sea_hours"
COL_VELOCIDAD = "velocidad_nudos"
COL_TARGET = "total_co2_emissions_m_tonnes"

# Pendientes t CO2 por hora navegada estimadas por SHAP en el notebook 07 (reports/07_pendiente_horas_
# _por_tipo.csv). Se usan solo como contraste externo del simulador, no como parte del cálculo.
TIPOS_ALTA_INTENSIDAD = [
    "LNG carrier",
    "Passenger ship",
    "Ro-pax ship",
    "Container ship",
]


# --------------------------------------------------------------------------------------------
# Carga de modelos
# --------------------------------------------------------------------------------------------


NIVELES = ("operacional", "operacional_tamano")

# Familia de referencia de cada nivel: la que produce la cifra que se publica.
FAMILIA_PRINCIPAL = {
    # Nivel 3: la media 50/50 de XGBoost y la red es el mejor modelo del nivel (R^2 0,9215 en test
    # frente a 0,9164 y 0,9156 por separado).
    "operacional": "mezcla",
    # Nivel 4: XGBoost solo llega a R^2 0,9484 en test, por encima de la mezcla del nivel anterior.
    # NO se promedia con la red: la red está entrenada sobre el nivel 3 y promediar un modelo bueno
    # con otro peor entrenado sobre menos información daría una cifra peor que el mejor de los dos.
    # La red se conserva cargada, pero solo como segunda opinión para la banda (ver banda_familias).
    "operacional_tamano": "xgb",
}


def cargar_modelos(dir_models: str = "../models", nivel: str = "operacional_tamano") -> dict:
    """
    Carga los modelos del nivel de información pedido y devuelve todo lo necesario para predecir en
    toneladas: el XGBoost con su lista de columnas, la red con su escalador, y qué familia es la de
    referencia.

    Dos niveles disponibles:

    * `"operacional"` — buque + tiempo + velocidad media. Se conserva como nivel de
      comparación.
    * `"operacional_tamano"` (por defecto) — añade la capacidad de carga reconstruida en el notebook
      03. R^2 0,9484 en test frente a 0,9215 de la mezcla del nivel anterior.

    **La red se carga siempre desde el nivel operacional**, y eso es deliberado: sus columnas son un
    subconjunto de las del nivel operacional+tamaño, así que puede predecir sobre el mismo
    dataframe seleccionando las suyas. En el nivel 4 no entra en la cifra publicada, solo en la
    banda: sirve para responder "¿cuánto se movería el resultado con otro modelo, entrenado además
    sobre otra información?", que es una pregunta de robustez más exigente que comparar dos modelos
    sobre las mismas variables.

    La red se importa aquí dentro y no arriba a propósito: así el módulo se puede usar (y la API
    arrancar) con solo XGBoost si TensorFlow no está instalado.
    """
    import joblib
    import pickle

    if nivel not in NIVELES:
        raise ValueError(f"nivel desconocido: {nivel!r} (usa uno de {NIVELES})")

    sufijo = "" if nivel == "operacional" else "_tamano"
    with open(f"{dir_models}/modelo_xgb_operacional{sufijo}.pickle", "rb") as fichero:
        pack_xgb = pickle.load(fichero)

    modelos = {
        "nivel": nivel,
        "xgb": pack_xgb["modelo"],
        "columnas": list(pack_xgb["columnas_predictoras"]),
        "metricas_xgb_test": pack_xgb.get("metricas_test", {}),
        "red": None,
        "preproceso": None,
        "familia_principal": FAMILIA_PRINCIPAL[nivel],
    }

    # La red se carga en dos intentos separados y el motivo del fallo se GUARDA, no se traga: así
    # «no hay TensorFlow» se distingue de «el .keras no carga», «falta el fichero» o «keras cambió
    # el formato». El motivo llega hasta `/salud` y `/metadatos`, que es donde se mira cuando un
    # despliegue responde sin banda.
    modelos["red_no_cargada"] = None
    try:
        import keras
    except ImportError as exc:  # pragma: no cover - depende del entorno
        modelos["red_no_cargada"] = f"TensorFlow/Keras no está instalado: {exc}"
    else:
        try:
            modelos["red"] = keras.models.load_model(f"{dir_models}/modelo_dl_operacional.keras")
            modelos["preproceso"] = joblib.load(f"{dir_models}/preproceso_dl_operacional.joblib")
        except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - depende del entorno
            modelos["red"] = None
            modelos["preproceso"] = None
            modelos["red_no_cargada"] = (
                f"Keras está instalado pero el modelo no se pudo cargar desde {dir_models}: "
                f"{type(exc).__name__}: {exc}"
            )

    modelos["familias"] = familias_disponibles(modelos)
    if modelos["familia_principal"] not in modelos["familias"]:
        # Sin red no hay mezcla: en el nivel 3 la referencia pasa a ser XGBoost. Es una degradación
        # explícita porque `red_no_cargada` dice por qué, y se sirve en `/salud`.
        modelos["familia_principal"] = "xgb"
    return modelos


def familias_disponibles(modelos: dict) -> tuple[str, ...]:
    """
    Qué familias se pueden pedir a `predecir` con los modelos cargados.

    En el nivel operacional+tamaño **no se ofrece "mezcla"**: promediar el XGBoost de ese nivel con
    una red entrenada sobre menos información daría una predicción peor que la del mejor de los dos.
    """
    if modelos.get("red") is None:
        return ("xgb",)
    if modelos.get("nivel", "operacional") == "operacional_tamano":
        return ("xgb", "red")
    return ("xgb", "red", "mezcla")


def familia_por_defecto(modelos: dict, familia: str | None = None) -> str:
    """Resuelve `None` a la familia de referencia del nivel cargado."""
    return familia if familia is not None else modelos.get("familia_principal", "xgb")


# --------------------------------------------------------------------------------------------
# Predicción
# --------------------------------------------------------------------------------------------


def predecir_xgb(modelos: dict, df_ml: pd.DataFrame) -> np.ndarray:
    """
    Predicción de XGBoost, con el mismo recorte a cero que usan los notebooks 04 y 05.

    Los NaN de `velocidad_nudos` NO se imputan: XGBoost los trata de forma nativa y así se entrenó
    (ver la nota guardada dentro del propio pickle del modelo).
    """
    pred = modelos["xgb"].predict(df_ml[modelos["columnas"]])
    return np.clip(np.asarray(pred, dtype="float64"), 0.0, None)


def predecir_red(modelos: dict, df_ml: pd.DataFrame) -> np.ndarray:
    """
    Predicción de la red densa. Reproduce el preproceso exacto con el que se entrenó: imputación de
    la velocidad por la mediana del train, indicador binario de ausencia, escalado de las 36
    columnas, y desescalado de la salida a toneladas.

    Ese indicador de ausencia no es un adorno: el notebook 05 midió que, sin imputar, la red no lanza
    ninguna excepción -- simplemente se para a las 16 épocas y predice una constante (R^2 0,000).
    """
    if modelos.get("red") is None:
        raise RuntimeError("La red no está cargada (¿falta TensorFlow?). Usa solo XGBoost.")

    pre = modelos["preproceso"]
    x = df_ml[pre["predictoras"]].copy()
    ausente = x[pre["columna_imputada"]].isna().astype("float64")
    x[pre["columna_imputada"]] = x[pre["columna_imputada"]].fillna(pre["mediana_velocidad"])
    x[pre["columna_extra"]] = ausente

    escalado = pre["escalador_x"].transform(x.values).astype("float32")

    # Se llama al modelo directamente en vez de usar `.predict()`. No es una preferencia de estilo:
    # sobre las 16.939 filas del ejercicio 2025, `.predict()` tarda 794 ms y `modelo(x)` 27 ms --29
    # veces mas rapido-- con una diferencia entre las dos salidas de EXACTAMENTE cero. El coste de
    # predecir de la red estaba dominado por el sobrecoste del bucle de `predict` (callbacks, troceado
    # y recoleccion de resultados), no por el calculo. Es el tipo de diferencia que solo aparece
    # cuando el modelo se sirve de verdad (el panel responde a cada deslizador) en vez de evaluarse
    # una vez en un notebook.
    #
    # Se trocea en bloques para no materializar de golpe un lote arbitrariamente grande: `predict`
    # lo hace por dentro y esta via no.
    TAMANO_BLOQUE = 50_000
    if len(escalado) <= TAMANO_BLOQUE:
        salida = np.asarray(modelos["red"](escalado, training=False)).ravel()
    else:
        trozos = [
            np.asarray(modelos["red"](escalado[i : i + TAMANO_BLOQUE], training=False)).ravel()
            for i in range(0, len(escalado), TAMANO_BLOQUE)
        ]
        salida = np.concatenate(trozos)

    return np.clip(salida * pre["desv_y"] + pre["media_y"], 0.0, None)


def predecir(modelos: dict, df_ml: pd.DataFrame, familia: str | None = None) -> np.ndarray:
    """
    Predicción en toneladas de CO2 con la familia pedida. `None` usa la de referencia del nivel.

    'mezcla' es la media 50/50 de las dos familias y solo existe en el nivel operacional, donde es
    el mejor modelo (R^2 0,9215 en test frente a 0,9164 y 0,9156 por separado) y donde la rejilla de
    pesos del notebook 05 mostró que el óptimo (0,60) solo aporta 0,0004 más. En el nivel
    operacional+tamaño la referencia es XGBoost solo (ver `familias_disponibles`).
    """
    familia = familia_por_defecto(modelos, familia)
    if familia == "xgb":
        return predecir_xgb(modelos, df_ml)
    if familia == "red":
        return predecir_red(modelos, df_ml)
    if familia == "mezcla":
        if familia not in familias_disponibles(modelos):
            raise ValueError(
                "'mezcla' no está disponible en el nivel "
                f"{modelos.get('nivel', 'operacional')!r}: promediar el XGBoost de este nivel con "
                "una red entrenada sobre menos información daría una predicción peor que la del "
                "mejor de los dos. Usa 'xgb' (la referencia) o 'red' (segunda opinión)."
            )
        return np.clip(0.5 * predecir_xgb(modelos, df_ml) + 0.5 * predecir_red(modelos, df_ml), 0.0, None)
    raise ValueError(f"familia desconocida: {familia!r} (usa una de {familias_disponibles(modelos)})")


# --------------------------------------------------------------------------------------------
# Construcción del escenario
# --------------------------------------------------------------------------------------------


def aplicar_escenario(
    df_ml: pd.DataFrame,
    delta_distancia: float = 0.0,
    delta_velocidad: float = 0.0,
    aplica_a: np.ndarray | pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Devuelve una copia del dataset con las horas y la velocidad modificadas según el escenario, y
    un diccionario con los avisos de lo que ha hecho falta recortar.

    Parámetros
    ----------
    delta_distancia : variación relativa de las millas recorridas (-0.10 = recorrer un 10% menos).
    delta_velocidad : variación relativa de la velocidad media (-0.10 = navegar un 10% más despacio).
    aplica_a : máscara booleana de las filas a las que se aplica el escenario. El resto se deja
        intacto. Sirve para escenarios focalizados (solo ciertos tipos de buque, solo una compañía,
        solo los buques por encima de cierta actividad...).

    Notas de honestidad del cálculo
    -------------------------------
    * Las horas se derivan de la identidad distancia = horas x velocidad; no se fijan a mano.
    * Las horas resultantes se recortan al máximo físico del año natural (8.784 h) y a cero. Si el
      recorte afecta a alguna fila, sale en los avisos: un escenario que necesita recortar es un
      escenario que ya no es el que se pidió.
    * En las filas sin velocidad conocida (6,1% del train) la palanca de velocidad modifica las
      horas pero el modelo no puede ver el cambio de velocidad. Salen contadas en los avisos para
      poder repetir el escenario solo sobre el subconjunto con velocidad conocida.
    """
    if not -1.0 < delta_distancia:
        raise ValueError("delta_distancia debe ser > -1 (no se puede recorrer menos de 0 millas)")
    if not -1.0 < delta_velocidad:
        raise ValueError("delta_velocidad debe ser > -1 (la velocidad no puede ser <= 0)")

    escenario = df_ml.copy()
    if aplica_a is None:
        mascara = np.ones(len(escenario), dtype=bool)
    else:
        mascara = np.asarray(aplica_a, dtype=bool)

    factor_horas = (1.0 + delta_distancia) / (1.0 + delta_velocidad)

    horas = escenario[COL_HORAS].to_numpy(dtype="float64", copy=True)
    horas_nuevas = horas.copy()
    horas_nuevas[mascara] = horas[mascara] * factor_horas
    recortadas = int((horas_nuevas > HORAS_MAX_FISICO).sum())
    horas_nuevas = np.clip(horas_nuevas, 0.0, HORAS_MAX_FISICO)
    escenario[COL_HORAS] = horas_nuevas

    velocidad = escenario[COL_VELOCIDAD].to_numpy(dtype="float64", copy=True)
    sin_velocidad = np.isnan(velocidad)
    velocidad[mascara] = velocidad[mascara] * (1.0 + delta_velocidad)
    escenario[COL_VELOCIDAD] = velocidad

    avisos = {
        "filas_afectadas": int(mascara.sum()),
        "horas_recortadas_al_maximo": recortadas,
        "filas_sin_velocidad_afectadas": int((sin_velocidad & mascara).sum()),
        "factor_horas": factor_horas,
        "horas_totales_base": float(np.nansum(horas[mascara])),
        "horas_totales_escenario": float(np.nansum(horas_nuevas[mascara])),
    }
    return escenario, avisos


# --------------------------------------------------------------------------------------------
# Simulación
# --------------------------------------------------------------------------------------------


def simular(
    modelos: dict,
    df_ml: pd.DataFrame,
    delta_distancia: float = 0.0,
    delta_velocidad: float = 0.0,
    aplica_a: np.ndarray | pd.Series | None = None,
    familias: tuple[str, ...] | None = None,
    prediccion_base: dict | None = None,
) -> dict:
    """
    Simula un escenario sobre un conjunto de filas y devuelve, por familia de modelo, la emisión
    predicha base, la del escenario y el ahorro (en toneladas y en porcentaje).

    `prediccion_base` permite reutilizar las predicciones del caso sin cambios entre escenarios;
    la mezcla de dos familias sobre 21.105 filas tarda ~160 ms, así que en una rejilla de 40
    escenarios ahorra la mitad del tiempo.
    """
    escenario, avisos = aplicar_escenario(df_ml, delta_distancia, delta_velocidad, aplica_a)
    familias = familias if familias is not None else familias_disponibles(modelos)

    resultado = {
        "nivel": modelos.get("nivel", "operacional"),
        "familia_principal": familia_por_defecto(modelos),
        "delta_distancia": delta_distancia,
        "delta_velocidad": delta_velocidad,
        "avisos": avisos,
        "por_familia": {},
        "predicciones": {},
    }

    for familia in familias:
        base = (prediccion_base or {}).get(familia)
        if base is None:
            base = predecir(modelos, df_ml, familia)
        nueva = predecir(modelos, escenario, familia)

        total_base = float(base.sum())
        total_nueva = float(nueva.sum())
        resultado["por_familia"][familia] = {
            "co2_base_t": total_base,
            "co2_escenario_t": total_nueva,
            "ahorro_t": total_base - total_nueva,
            "ahorro_pct": 100.0 * (total_base - total_nueva) / total_base if total_base else np.nan,
        }
        resultado["predicciones"][familia] = {"base": base, "escenario": nueva}

    return resultado


def rejilla_escenarios(
    modelos: dict,
    df_ml: pd.DataFrame,
    deltas_distancia: list[float],
    deltas_velocidad: list[float],
    familias: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Recorre el producto cartesiano de las dos palancas y devuelve una tabla larga con una fila por
    escenario. Las predicciones del caso base se calculan una sola vez.
    """
    familias = familias if familias is not None else familias_disponibles(modelos)
    base = {familia: predecir(modelos, df_ml, familia) for familia in familias}

    filas = []
    for dd in deltas_distancia:
        for dv in deltas_velocidad:
            salida = simular(modelos, df_ml, dd, dv, familias=familias, prediccion_base=base)
            fila = {"delta_distancia": dd, "delta_velocidad": dv}
            for familia in familias:
                metricas = salida["por_familia"][familia]
                fila[f"ahorro_t_{familia}"] = metricas["ahorro_t"]
                fila[f"ahorro_pct_{familia}"] = metricas["ahorro_pct"]
            fila["horas_recortadas"] = salida["avisos"]["horas_recortadas_al_maximo"]
            filas.append(fila)

    return pd.DataFrame(filas)


def banda_familias(resultado: dict, clave: str = "ahorro_t") -> tuple[float, float, float]:
    """
    Banda de incertidumbre por desacuerdo entre modelos: (mínimo, central, máximo).

    El central es siempre la **familia de referencia del nivel** cargado, y los extremos son el
    mínimo y el máximo entre XGBoost y la red. No es un intervalo de confianza estadístico y no debe
    presentarse como tal: mide hasta qué punto la respuesta depende de qué modelo se eligió.

    En el nivel operacional+tamaño la comparación es además entre dos **niveles de información**
    distintos —XGBoost ve la capacidad de carga y la red no—, así que la banda responde a una
    pregunta más exigente que la del nivel anterior: no solo "¿y si hubiera elegido la otra
    familia?", sino "¿y si además hubiera trabajado sin reconstruir el tamaño del buque?".
    """
    por_familia = resultado["por_familia"]
    valores = [por_familia[f][clave] for f in ("xgb", "red") if f in por_familia]
    principal = resultado.get("familia_principal", "mezcla")
    if principal not in por_familia:
        principal = "xgb" if "xgb" in por_familia else next(iter(por_familia))
    central = por_familia[principal][clave]
    if not valores:
        valores = [central]
    return float(min(valores)), float(central), float(max(valores))


def bootstrap_flota(
    ahorro_por_fila: np.ndarray,
    grupos: pd.Series,
    n_muestras: int = 500,
    semilla: int = 1234567,
) -> tuple[float, float]:
    """
    Intervalo del 95% del ahorro agregado por remuestreo **de buques** (no de filas): un buque
    entra o no entra con todos sus años, igual que la partición train/test del notebook 03. Remuestrear
    filas sueltas daría un intervalo artificialmente estrecho, porque los años del mismo buque no
    son observaciones independientes.
    """
    aleatorio = np.random.default_rng(semilla)
    codigos, indices = np.unique(np.asarray(grupos), return_inverse=True)
    por_buque = np.bincount(indices, weights=ahorro_por_fila, minlength=len(codigos))

    n = len(codigos)
    totales = np.empty(n_muestras)
    for i in range(n_muestras):
        muestra = aleatorio.integers(0, n, size=n)
        totales[i] = por_buque[muestra].sum()

    return float(np.percentile(totales, 2.5)), float(np.percentile(totales, 97.5))


# --------------------------------------------------------------------------------------------
# Lecturas agregadas
# --------------------------------------------------------------------------------------------


def agregar_por_tipo(resultado: dict, tipos: pd.Series, familia: str | None = None) -> pd.DataFrame:
    """
    Reparte el ahorro del escenario por tipo de buque y añade el ahorro por hora de navegación
    evitada, que es la cifra comparable con las pendientes SHAP del notebook 07.
    """
    familia = familia if familia is not None else resultado.get("familia_principal", "mezcla")
    base = resultado["predicciones"][familia]["base"]
    nueva = resultado["predicciones"][familia]["escenario"]

    tabla = pd.DataFrame(
        {
            "tipo": np.asarray(tipos),
            "co2_base_t": base,
            "co2_escenario_t": nueva,
            "ahorro_t": base - nueva,
        }
    )
    agregada = tabla.groupby("tipo", observed=True).agg(
        n=("ahorro_t", "size"),
        co2_base_t=("co2_base_t", "sum"),
        ahorro_t=("ahorro_t", "sum"),
    )
    agregada["ahorro_pct"] = 100.0 * agregada["ahorro_t"] / agregada["co2_base_t"]
    return agregada.sort_values("ahorro_t", ascending=False)


def escenario_focalizado(
    modelos: dict,
    df_ml: pd.DataFrame,
    tipos: pd.Series,
    horas_a_evitar: float,
    tipos_objetivo: list[str] | None = None,
    familia: str | None = None,
    prediccion_base: dict | None = None,
) -> dict:
    """
    Compara dos formas de conseguir **el mismo recorte total de horas de navegación** en la flota:
    repartido a partes iguales entre todos los buques, o concentrado en unos tipos concretos.

    Es la traducción operativa del resultado central del notebook 07 (una hora de pasaje cuesta el doble
    que una de carga general): con el mismo esfuerzo, ¿cuánto más se ahorra si se elige dónde?

    `horas_a_evitar` se expresa en horas totales de flota, no en porcentaje, precisamente para que
    las dos alternativas sean comparables a esfuerzo constante.
    """
    familia = familia_por_defecto(modelos, familia)
    tipos = pd.Series(np.asarray(tipos), index=df_ml.index)
    horas = df_ml[COL_HORAS].to_numpy(dtype="float64")
    horas_totales = float(np.nansum(horas))

    if tipos_objetivo is None:
        tipos_objetivo = TIPOS_ALTA_INTENSIDAD
    objetivo = tipos.isin(tipos_objetivo).to_numpy()
    horas_objetivo = float(np.nansum(horas[objetivo]))

    if horas_a_evitar > horas_objetivo:
        raise ValueError(
            f"No se pueden evitar {horas_a_evitar:,.0f} h dentro de los tipos elegidos: "
            f"solo navegan {horas_objetivo:,.0f} h en total."
        )

    # Reducción proporcional necesaria en cada caso para evitar las mismas horas de flota.
    delta_uniforme = -horas_a_evitar / horas_totales
    delta_focalizado = -horas_a_evitar / horas_objetivo

    # Velocidad constante: reducir millas a velocidad constante reduce horas en la misma proporción.
    uniforme = simular(modelos, df_ml, delta_distancia=delta_uniforme, familias=(familia,),
                       prediccion_base=prediccion_base)
    focalizado = simular(modelos, df_ml, delta_distancia=delta_focalizado, aplica_a=objetivo,
                         familias=(familia,), prediccion_base=prediccion_base)

    ahorro_uniforme = uniforme["por_familia"][familia]["ahorro_t"]
    ahorro_focalizado = focalizado["por_familia"][familia]["ahorro_t"]

    return {
        "horas_a_evitar": horas_a_evitar,
        "horas_totales_flota": horas_totales,
        "horas_totales_objetivo": horas_objetivo,
        "tipos_objetivo": list(tipos_objetivo),
        "recorte_uniforme_pct": 100.0 * delta_uniforme,
        "recorte_focalizado_pct": 100.0 * delta_focalizado,
        "ahorro_uniforme_t": ahorro_uniforme,
        "ahorro_focalizado_t": ahorro_focalizado,
        "ventaja_focalizado_t": ahorro_focalizado - ahorro_uniforme,
        "ventaja_focalizado_pct": 100.0 * (ahorro_focalizado - ahorro_uniforme) / ahorro_uniforme
        if ahorro_uniforme
        else np.nan,
        "resultado_uniforme": uniforme,
        "resultado_focalizado": focalizado,
    }


def simular_buque(
    modelos: dict,
    df_ml: pd.DataFrame,
    df_raw: pd.DataFrame,
    imo: str,
    anio: int,
    escenarios: list[tuple[float, float]],
    familia: str | None = None,
) -> pd.DataFrame:
    """
    Ficha de un buque-año concreto: emisión observada, predicha, y el efecto de cada escenario.

    Es la unidad que consume la API (un buque, unos deltas, una respuesta) y el caso demostrativo
    del vídeo.
    """
    seleccion = (df_raw["ship_imo_number"].astype(str) == str(imo)) & (df_raw["reporting_year"] == anio)
    if not seleccion.any():
        raise KeyError(f"No hay fila para el IMO {imo} en {anio}")

    fila_ml = df_ml.loc[seleccion.to_numpy()]
    fila_raw = df_raw.loc[seleccion.to_numpy()]

    base = float(predecir(modelos, fila_ml, familia)[0])
    observada = float(fila_raw[COL_TARGET].iloc[0])

    filas = []
    for dd, dv in escenarios:
        escenario, avisos = aplicar_escenario(fila_ml, dd, dv)
        nueva = float(predecir(modelos, escenario, familia)[0])
        filas.append(
            {
                "delta_distancia_pct": 100 * dd,
                "delta_velocidad_pct": 100 * dv,
                "horas": float(escenario[COL_HORAS].iloc[0]),
                "velocidad_nudos": float(escenario[COL_VELOCIDAD].iloc[0]),
                "co2_predicho_t": nueva,
                "ahorro_t": base - nueva,
                "ahorro_pct": 100.0 * (base - nueva) / base if base else np.nan,
            }
        )

    tabla = pd.DataFrame(filas)
    tabla.attrs["buque"] = fila_raw["ship_name"].iloc[0]
    tabla.attrs["tipo"] = fila_raw["ship_type_agrupado"].iloc[0]
    tabla.attrs["co2_observado_t"] = observada
    tabla.attrs["co2_base_predicho_t"] = base
    return tabla


def fiabilidad_por_tamano(
    y_real: np.ndarray, y_predicho: np.ndarray, n_quintiles: int = 5
) -> pd.DataFrame:
    """
    Error relativo mediano por quintil de emisión observada.

    El notebook 04 midió que el modelo es proporcionalmente menos fiable con los buques pequeños (24,7%
    de error relativo mediano en el quintil bajo frente a 15-17% en el resto). Cualquier salida del
    simulador para un buque pequeño tiene que ir acompañada de ese aviso, así que la función vive
    aquí y no en el notebook: la API la necesita igual.
    """
    tabla = pd.DataFrame({"real": np.asarray(y_real, dtype="float64"),
                          "predicho": np.asarray(y_predicho, dtype="float64")})
    tabla["quintil"] = pd.qcut(tabla["real"], n_quintiles, labels=False, duplicates="drop") + 1
    tabla["error_relativo"] = np.where(
        tabla["real"] > 0, np.abs(tabla["predicho"] - tabla["real"]) / tabla["real"], np.nan
    )
    resumen = tabla.groupby("quintil").agg(
        n=("real", "size"),
        co2_mediano_t=("real", "median"),
        error_relativo_mediano_pct=("error_relativo", lambda s: 100 * s.median()),
    )
    return resumen
