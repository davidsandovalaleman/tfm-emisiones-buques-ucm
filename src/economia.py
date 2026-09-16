"""
Capa economica: que vale, en euros y para quien, cada tonelada de CO2 que este trabajo predice.

Se construye **encima** del modelo, igual que la capa regulatoria: no reentrena nada, no toca el
`.pickle` verificado y no cambia una sola cifra de los notebooks. Lo que anade son cuatro cosas
que un profesional del sector necesita y que las toneladas por si solas no dan:

1. **El ahorro alcanzable a actividad constante** (`residuo_persistente`, `frontera_eficiencia`).
   El escenario del simulador (notebook 08) ahorra 13,75 Mt recortando un 10% de las millas, y la objecion obvia
   es que recortar millas es transportar menos. El residuo del modelo en el nivel
   operacional+tamano compara cada buque con lo que emitirian sus pares **haciendo exactamente lo
   mismo** -- mismo tamano, misma velocidad, mismas horas --, asi que el hueco hasta la frontera de
   su grupo es ahorro **sin perder una milla ni una tonelada de carga**.

   El problema de ese razonamiento, y por que aqui no se aplica en crudo: el residuo tambien
   contiene el error del modelo, y el procedimiento "que todos lleguen a la mediana" produciria un
   ahorro positivo **incluso si el residuo fuese ruido puro**. La prueba que separa senal de ruido
   es la **persistencia**: si el residuo es una propiedad del buque, debe repetirse ano tras ano.
   Se mide (correlacion con el propio residuo del ano anterior, y componentes de varianza) y se
   retiene solo la parte persistente mediante contraccion empirico-bayesiana. La diferencia entre
   el ahorro ingenuo y el persistente es, ella misma, un resultado.

2. **Los tres flujos de dinero** (`valor_economico`). Una tonelada evitada mueve tres cosas: el combustible que no se quema (el mayor), el
   cumplimiento evitado (ETS + FuelEU) y el ingreso perdido si el recorte es de actividad.

3. **El precio de equilibrio por dia de mar** (`breakeven_dia_mar`). Evita inventar tarifas de
   fletamento: en vez de suponer cuanto gana un buque al dia, se calcula **cuanto vale el dia de mar
   evitado** y se deja que quien decide lo compare con lo que el sabe que gana.

4. **La cuenta con dueno** (`exposicion_por_compania`, `trayectoria_cii`): la factura por naviera,
   el margen de compensacion interna de FuelEU y el ano en que cada buque cae a D/E si sigue
   operando como hoy.

Limitaciones declaradas, todas de las que cambian como hay que leer los numeros:

* El residuo persistente **no demuestra ineficiencia**: demuestra emision sistematicamente distinta
  de la de los pares con la misma actividad observada. Parte de ese hueco es heterogeneidad no
  medida -- rutas, meteorologia, espera en puerto, carga real --. Por eso la frontera se pone en la
  **mediana** del grupo (y no en el minimo) y el resultado se declara como **cota superior de lo
  alcanzable**, no como dinero garantizado.
* El precio del combustible es un **parametro con fuente y fecha**, como el derecho de emision, y
  las conclusiones se dan en banda porque el precio se mueve.
* El ahorro de FuelEU se prorratea con la energia, que supone mezcla de combustible constante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------
# Parametros con fuente y fecha (mismo criterio que ets.PRECIO_EUA_POR_DEFECTO)
# --------------------------------------------------------------------------------------------

#: Precio del combustible marino. Se deja como banda porque en agosto de 2026 el mercado esta alto
#: (Brent por encima de 90 $/bbl) y la conclusion cualitativa -- el combustible vale varias veces
#: mas que el derecho de emision -- se sostiene en todo el rango.
PRECIO_BUNKER_EUR_POR_T = 500.0
PRECIO_BUNKER_BANDA = (400.0, 700.0)
PRECIO_BUNKER_NOTA = "VLSFO, banda 400-700 EUR/t; parametro, no constante. Fijar con fuente y fecha al publicar."

#: Cociente CO2/combustible medido sobre la flota de 2025 en este mismo dataset (mediana).
#: Su inverso son las toneladas de combustible que se dejan de quemar por tonelada de CO2 evitada.
RATIO_CO2_FUEL = 3.1407

COL_TARGET = "total_co2_emissions_m_tonnes"
COL_HORAS = "time_spent_at_sea_hours"


def toneladas_de_fuel(ahorro_co2_t) -> float | np.ndarray:
    """Combustible no quemado que corresponde a un ahorro de CO2."""
    return np.asarray(ahorro_co2_t, dtype="float64") / RATIO_CO2_FUEL


# --------------------------------------------------------------------------------------------
# 1. Prediccion honesta para todas las filas
# --------------------------------------------------------------------------------------------


def _pliegues_por_buque(grupos: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reparte los buques en `n_splits` pliegues equilibrados, sin partir ninguno.

    Misma idea que `GroupKFold` -- los buques con mas filas primero, cada uno al pliegue que menos
    filas lleva -- pero con la regla escrita aqui en vez de heredada de la version de scikit-learn
    que este instalada. El desempate es por numero IMO, asi que el reparto es identico en cualquier
    maquina, hoy y dentro de dos anos.

    Devuelve la lista de pares (indices de entrenamiento, indices de validacion).
    """
    tamanos = grupos.value_counts()
    # Orden estable y explicito: primero por numero de filas descendente, y a igualdad, por IMO.
    orden = sorted(tamanos.index, key=lambda imo: (-int(tamanos[imo]), str(imo)))
    carga = [0] * n_splits
    pliegue_de = {}
    for imo in orden:
        destino = min(range(n_splits), key=lambda k: (carga[k], k))
        pliegue_de[imo] = destino
        carga[destino] += int(tamanos[imo])

    asignado = grupos.map(pliegue_de).to_numpy()
    indices = np.arange(len(grupos))
    return [(indices[asignado != k], indices[asignado == k]) for k in range(n_splits)]


def prediccion_honesta(ml: pd.DataFrame, pack: dict, n_splits: int = 3) -> pd.DataFrame:
    """Prediccion fuera de muestra para **todas** las filas, no solo para el conjunto de prueba.

    Sin esto el analisis de eficiencia estaria sesgado: en las filas de entrenamiento el modelo se
    ajusta a su propio residuo, el hueco hasta la frontera sale artificialmente pequeno y el
    resultado no seria comparable entre los buques de train y los de test.

    * Filas de entrenamiento -> prediccion *out-of-fold* agrupada por buque, el mismo esquema de
      validacion del notebook 04, reajustando el modelo con sus hiperparametros.
    * Filas de prueba -> el modelo guardado, que nunca las vio.

    **Reparto de pliegues propio.** El reparto de `GroupKFold` depende de la version de
    scikit-learn instalada, y con el las cifras de esta capa pueden moverse entre maquinas (la
    persistencia, de 0,482 a 0,479; el ahorro, de 7,41 a 7,27 Mt). Por eso se usa
    `_pliegues_por_buque`, que hace lo mismo que `GroupKFold` -- equilibrar el tamano de los
    pliegues sin partir ningun buque -- con una regla escrita aqui, deterministica y sin
    dependencias: esta capa da la misma cifra en cualquier maquina y con cualquier version de
    scikit-learn.
    """
    from xgboost import XGBRegressor

    cols = list(pack["columnas_predictoras"])
    par = {k: v for k, v in pack["modelo"].get_params().items()
           if k in ("objective", "gamma", "learning_rate", "max_depth",
                    "n_estimators", "random_state", "tree_method", "n_jobs")}

    tr = ml[ml["es_train"]].reset_index(drop=True)
    y = tr[COL_TARGET].to_numpy()
    oof = np.full(len(tr), np.nan)
    for idx_a, idx_b in _pliegues_por_buque(tr["ship_imo_number"], n_splits):
        modelo = XGBRegressor(**par)
        modelo.fit(tr[cols].iloc[idx_a], y[idx_a])
        oof[idx_b] = np.clip(modelo.predict(tr[cols].iloc[idx_b]), 0.0, None)

    salida_tr = tr[["ship_imo_number", "reporting_year"]].copy()
    salida_tr["pred_honesta"] = oof
    salida_tr["origen"] = "oof"

    te = ml[~ml["es_train"]].reset_index(drop=True)
    salida_te = te[["ship_imo_number", "reporting_year"]].copy()
    salida_te["pred_honesta"] = np.clip(pack["modelo"].predict(te[cols]), 0.0, None)
    salida_te["origen"] = "test"

    return pd.concat([salida_tr, salida_te], ignore_index=True)


# --------------------------------------------------------------------------------------------
# 2. Residuo persistente: separar la eficiencia del ruido
# --------------------------------------------------------------------------------------------


def residuo_persistente(d: pd.DataFrame, winsor: float = 0.01) -> tuple[pd.DataFrame, dict]:
    """Residuo relativo de cada fila y su componente **persistente** por buque.

    Devuelve el dataframe con tres columnas nuevas y un diccionario de diagnostico:

    * `residuo_rel`     -- (observado - predicho) / predicho.
    * `residuo_w`       -- el anterior winsorizado a los percentiles 1 y 99. Sin este recorte una
      sola fila con residuo 219 domina cualquier agregado.
    * `residuo_persistente` -- estimador contraido (empirico-bayesiano) de la media del buque:
      `w * media_del_buque`, con `w = var_entre / (var_entre + var_intra / n_anios)`. Un buque con
      un solo ano y mucha varianza intra apenas conserva residuo; uno con ocho anos coherentes lo
      conserva casi entero.

    Diagnostico: correlacion del residuo con el del ano anterior del **mismo buque** (Pearson y
    Spearman) y componentes de varianza con su ICC. Si la correlacion fuese ~0, todo lo que sigue
    seria ruido y habria que decirlo; sale ~0,48.
    """
    d = d.copy()
    d["residuo_rel"] = (d[COL_TARGET] - d["pred_honesta"]) / d["pred_honesta"]
    lo, hi = d["residuo_rel"].quantile([winsor, 1 - winsor])
    d["residuo_w"] = d["residuo_rel"].clip(lo, hi)

    d = d.sort_values(["ship_imo_number", "reporting_year"])
    lag = d.groupby("ship_imo_number")["residuo_w"].shift(1)
    anio_lag = d.groupby("ship_imo_number")["reporting_year"].shift(1)
    consecutivos = anio_lag == d["reporting_year"] - 1

    n = d.groupby("ship_imo_number")["residuo_w"].size()
    media = d.groupby("ship_imo_number")["residuo_w"].mean()
    intra = d["residuo_w"] - d["ship_imo_number"].map(media)
    con_repeticion = d["ship_imo_number"].isin(n[n >= 2].index)
    var_intra = float((intra[con_repeticion] ** 2).sum() / (int(con_repeticion.sum()) - int(n[n >= 2].size)))
    var_total = float(d["residuo_w"].var())
    var_entre = max(var_total - var_intra, 0.0)

    peso = var_entre / (var_entre + var_intra / d["ship_imo_number"].map(n))
    d["residuo_persistente"] = peso * d["ship_imo_number"].map(media)

    diagnostico = {
        "filas": int(len(d)),
        "winsor_limites": (float(lo), float(hi)),
        "pares_consecutivos": int(consecutivos.sum()),
        "corr_lag1_pearson": float(d.loc[consecutivos, "residuo_w"].corr(lag[consecutivos])),
        "corr_lag1_spearman": float(d.loc[consecutivos, "residuo_w"].corr(lag[consecutivos], method="spearman")),
        "var_total": var_total,
        "var_intra": var_intra,
        "var_entre": var_entre,
        "icc": var_entre / var_total if var_total else np.nan,
    }
    return d, diagnostico


# --------------------------------------------------------------------------------------------
# 3. Frontera de eficiencia y ahorro a actividad constante
# --------------------------------------------------------------------------------------------


def grupos_de_comparacion(a: pd.DataFrame, col_tipo: str = "ship_type_agrupado",
                          col_capacidad: str = "capacidad_estimada",
                          quintiles: int = 5, minimo: int = 30) -> pd.Series:
    """Grupo de pares de cada buque: **tipo x quintil de capacidad dentro del tipo**.

    Es la misma particion con la que la capa regulatoria midio el sesgo de tamano del CII, y por la
    misma razon: comparar un *feeder* con un portacontenedores de 20.000 TEU no dice nada. Los
    grupos con menos de `minimo` filas se colapsan en un resto por tipo.

    **Buques sin capacidad reconstruida.** Construir la clave concatenando
    `q.astype("Int64").astype(str)` es fragil: un buque sin capacidad sale de `qcut` con `NA` y,
    desde pandas 3.0, `astype(str)` lo convierte en `NaN`, que se propaga a la clave entera.
    `groupby` descartaria esas filas y `valor_economico` las sumaria con el `.sum()` de pandas, que
    **salta los NaN sin avisar**: con `pandas==3.0.2` se perderian **0,43 Mt del ahorro de
    cabecera, un 6%**. Por eso el grupo de los buques sin capacidad es explicito
    (`tipo|sin_capacidad`) y ninguna fila se queda sin clave.
    """
    q = a.groupby(col_tipo)[col_capacidad].transform(
        lambda s: pd.qcut(s.rank(method="first"), quintiles, labels=False) if s.notna().sum() >= 25 else np.nan)
    # La etiqueta se construye sin pasar por `astype(str)` sobre un tipo con nulos: primero se
    # decide el texto y solo despues se concatena, de modo que NUNCA puede salir una clave nula.
    etiqueta = pd.Series(
        np.where(q.notna(), "Q" + q.fillna(0).astype("int64").astype(str), "sin_capacidad"),
        index=a.index, dtype="object")
    grupo = a[col_tipo].astype(str) + "|" + etiqueta
    if grupo.isna().any():  # pragma: no cover - invariante
        raise AssertionError(f"{int(grupo.isna().sum())} filas se han quedado sin grupo de comparacion")
    pequeno = grupo.groupby(grupo).transform("size") < minimo
    return grupo.mask(pequeno, a[col_tipo].astype(str) + "|resto")


def frontera_eficiencia(a: pd.DataFrame, col_residuo: str = "residuo_persistente",
                        cuantil: float = 0.50, col_grupo: str = "grupo") -> pd.DataFrame:
    """Ahorro **a actividad constante** de cada buque hasta la frontera de su grupo.

    `cuantil` 0,50 es la lectura conservadora ("que la mitad peor opere como la mediana de sus
    pares") y 0,25 la ambiciosa ("como el mejor cuartil"). El ahorro se mide sobre la **prediccion**
    del buque, no sobre lo observado, para que no arrastre el ruido del ano concreto.
    """
    salida = a.copy()
    salida["frontera"] = salida.groupby(col_grupo)[col_residuo].transform(lambda s: s.quantile(cuantil))
    salida["ahorro_actividad_constante_t"] = np.maximum(
        0.0, salida["pred_honesta"] * (salida[col_residuo] - salida["frontera"]))
    return salida


# --------------------------------------------------------------------------------------------
# 4. Los tres flujos de dinero
# --------------------------------------------------------------------------------------------


def valor_economico(a: pd.DataFrame, ahorro_t: pd.Series | np.ndarray,
                    precio_eua: float, precio_bunker: float = PRECIO_BUNKER_EUR_POR_T,
                    col_factor_ambito: str = "factor_ambito",
                    col_penalizacion: str = "penalizacion_fueleu_eur") -> dict:
    """Traduce un ahorro de toneladas a los tres flujos, por separado y sumados.

    * **ETS**: solo la fraccion dentro de ambito paga derechos. Multiplicar el ahorro entero por el
      precio del derecho sobrestima el resultado en un ~70%.
    * **FuelEU**: la multa se prorratea con la energia, que baja en la misma proporcion que el CO2
      si la mezcla de combustible no cambia.
    * **Combustible**: el flujo mayor.

    **Un solo criterio de suma.** Las sumas de pandas **saltan los NaN** y las de numpy los
    **propagan**; mezclarlas haria que una sola fila sin ahorro calculable diera un `ets_eur` finito
    -- calculado sobre menos buques de los declarados -- junto a un `combustible_eur` y un
    `total_eur` a NaN. Por eso las filas sin ahorro calculable se cuentan, se declaran en
    `filas_sin_ahorro_calculable` y se tratan como cero de forma explicita, y todas las sumas usan
    el mismo criterio.
    """
    ahorro = pd.Series(np.asarray(ahorro_t, dtype="float64"), index=a.index)
    sin_calcular = int(ahorro.isna().sum())
    ahorro = ahorro.fillna(0.0)
    fraccion = (ahorro / a[COL_TARGET]).clip(0, 1).fillna(0.0)
    fuel_t = float(toneladas_de_fuel(ahorro).sum())
    ets_eur = float((ahorro * a[col_factor_ambito].fillna(0.0) * precio_eua).sum())
    fueleu_eur = float((a[col_penalizacion].fillna(0.0) * fraccion).sum())
    combustible_eur = fuel_t * precio_bunker
    return {
        "ahorro_t": float(ahorro.sum()),
        "buques": int((ahorro > 0).sum()),
        "filas_sin_ahorro_calculable": sin_calcular,
        "combustible_t": fuel_t,
        "combustible_eur": combustible_eur,
        "ets_eur": ets_eur,
        "fueleu_eur": fueleu_eur,
        "cumplimiento_eur": ets_eur + fueleu_eur,
        "total_eur": combustible_eur + ets_eur + fueleu_eur,
        "eur_por_t_cumplimiento": (ets_eur + fueleu_eur) / ahorro.sum() if ahorro.sum() else np.nan,
        "eur_por_t_combustible": combustible_eur / ahorro.sum() if ahorro.sum() else np.nan,
        "precio_bunker": precio_bunker,
        "precio_eua": precio_eua,
    }


def breakeven_dia_mar(valor_eur: float, horas_evitadas: float) -> float:
    """Cuanto vale un dia de mar evitado, en euros.

    Es el precio de equilibrio de la palanca: si un buque gana menos que esta cifra por dia de
    navegacion, recortar millas le sale a cuenta **sin necesidad de que nadie le pague por el CO2**.
    Se publica asi, y no como un ahorro neto, para no tener que inventar tarifas de fletamento: el
    armador ya sabe lo que gana al dia; lo que no sabia es contra que compararlo.
    """
    dias = horas_evitadas / 24.0
    return valor_eur / dias if dias else np.nan


# --------------------------------------------------------------------------------------------
# 5. La cuenta con dueno
# --------------------------------------------------------------------------------------------


def exposicion_por_compania(a: pd.DataFrame, precio_eua: float,
                            col_id: str = "company_imo_number",
                            col_nombre: str = "company_name") -> pd.DataFrame:
    """Factura de cumplimiento de 2026 por naviera, con su cuota y su ahorro alcanzable.

    Se agrupa por el **numero IMO de compania**, no por el nombre: el nombre viene en texto libre y
    la misma naviera aparece con varias grafias. Aun asi el IMO de compania identifica a la entidad
    gestora, no al grupo, de modo que un grupo grande sigue apareciendo repartido en varias filas;
    se declara y no se corrige a mano.
    """
    d = a.copy()
    d["_cid"] = d[col_id].astype(str).where(d[col_id].notna(), "SIN_ID:" + d[col_nombre].astype(str))
    nombre = d.sort_values(COL_TARGET, ascending=False).groupby("_cid")[col_nombre].first()
    g = d.groupby("_cid").agg(
        buques=("ship_imo_number", "nunique"),
        co2_t=(COL_TARGET, "sum"),
        ets_eur=("ets_2026_eur", "sum"),
        fueleu_eur=("penalizacion_fueleu_eur", "sum"),
        ahorro_t=("ahorro_actividad_constante_t", "sum"),
    )
    g["factura_eur"] = g["ets_eur"] + g["fueleu_eur"]
    g["nombre"] = g.index.map(nombre)
    g = g.sort_values("factura_eur", ascending=False)
    g["cuota_pct"] = 100 * g["factura_eur"] / g["factura_eur"].sum()
    g["cuota_acumulada_pct"] = g["cuota_pct"].cumsum()
    return g


def compensacion_interna_fueleu(a: pd.DataFrame, col_id: str = "company_imo_number") -> dict:
    """Cuanto deficit de FuelEU puede cada naviera compensar **dentro de su propia flota**.

    FuelEU permite agrupar saldos, y la agrupacion se organiza por armador: los buques con
    intensidad por debajo del objetivo generan excedente y pueden cubrir a los que no llegan. Lo que
    no se compensa dentro hay que comprarlo fuera o pagarlo, y eso es un mercado con precio.
    """
    d = a.dropna(subset=["balance_fueleu_gco2eq"]).copy()
    # Misma salvaguarda que `exposicion_por_compania`, y hace mas falta aqui: sin ella todos los
    # buques sin identificador de compania caerian en un unico grupo "nan" -- una naviera ficticia
    # -- y su excedente compensaria deficits de armadores que no tienen nada que ver, inflando el
    # porcentaje neteable. Hoy son 8 buques y no mueven la cifra, pero la salvaguarda lo garantiza.
    d["_cid"] = d[col_id].astype(str).where(
        d[col_id].notna(), "SIN_ID:" + d.index.astype(str))
    p = d.groupby("_cid").agg(
        excedente=("balance_fueleu_gco2eq", lambda s: s[s > 0].sum()),
        deficit=("balance_fueleu_gco2eq", lambda s: -s[s < 0].sum()),
        penalizacion=("penalizacion_fueleu_eur", "sum"))
    p["neteo"] = np.minimum(p["excedente"], p["deficit"])
    con = p[p["deficit"] > 0]
    residual = (con["deficit"] - con["neteo"]) / con["deficit"]
    return {
        "companias_con_deficit": int(len(con)),
        "companias_con_algun_excedente": int((con["excedente"] > 0).sum()),
        "pct_companias_con_excedente": float(100 * (con["excedente"] > 0).mean()),
        "pct_deficit_neteable": float(100 * con["neteo"].sum() / con["deficit"].sum()),
        "penalizacion_sin_pooling_eur": float(con["penalizacion"].sum()),
        "penalizacion_con_pooling_eur": float((con["penalizacion"] * residual).sum()),
        "companias_que_se_compensan_del_todo": int((con["deficit"] - con["neteo"] <= 0).sum()),
    }


def trayectoria_cii(q: pd.DataFrame, anios: range) -> pd.DataFrame:
    """Banda A-E de cada buque en cada ano futuro **si sigue operando como hoy**.

    El CII obtenido se deja fijo y solo se mueve el exigido, que se endurece con el calendario ya
    publicado (11% en 2026 ... 21,5% en 2030). Anade `anio_caida`: el primer ano en que el buque
    recibe D o E. No es una prediccion del comportamiento del buque: es la **fecha de caducidad de
    su operacion actual**, que es justo lo que un armador o un financiador necesita saber con
    antelacion.
    """
    from cii import banda as _banda, cii_requerido as _requerido

    d = q.copy()
    caida = pd.Series(np.nan, index=d.index)
    for anio in anios:
        req = np.array([_requerido(t, c, anio) for t, c in zip(d["cii_tipo"], d["cii_capacidad"])])
        bandas = [_banda(at, rq, t, c) for at, rq, t, c
                  in zip(d["cii_attained"], req, d["cii_tipo"], d["cii_capacidad"])]
        d[f"banda_{anio}"] = bandas
        nueva = pd.Series(bandas, index=d.index).isin(["D", "E"]) & caida.isna()
        caida[nueva] = anio
    d["anio_caida"] = caida
    return d
