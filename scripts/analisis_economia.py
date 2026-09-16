"""Ejecuta la capa economica y escribe los informes 14-17 en `reports/`.

    python scripts/analisis_economia.py

No reentrena el modelo de produccion ni toca ningun artefacto anterior: solo lee el `.pickle`
verificado, calcula predicciones fuera de muestra para poder comparar buques entre si y escribe
ficheros nuevos.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import cii, economia, ets, fueleu, simulador  # noqa: E402

DATOS = os.path.join(RAIZ, "data", "processed")
INFORMES = os.path.join(RAIZ, "reports")
ANIO = 2025


def _cargar():
    ml = pd.read_parquet(f"{DATOS}/mrv_features_operacional_tamano_ml.parquet")
    raw = pd.read_parquet(f"{DATOS}/mrv_features_operacional_tamano_raw.parquet")
    with open(f"{RAIZ}/models/modelo_xgb_operacional_tamano.pickle", "rb") as fichero:
        pack = pickle.load(fichero)
    return ml, raw, pack


def _sello_entorno() -> None:
    """Escribe con qué se generaron estos informes, porque la capa económica **se mueve**.

    Regenerados en otra máquina, con el mismo código y las versiones que fija
    `requirements-lock.txt`, el ahorro a actividad constante puede variar ligeramente (7,27 Mt
    frente a 7,41, con una persistencia de 0,479 frente a 0,482). La causa está en
    `prediccion_honesta`, que es el único punto del proyecto donde se **entrena** un modelo fuera
    del cuaderno 04: bastan diferencias mínimas en la predicción fuera de muestra para mover el
    hueco hasta la frontera del grupo, que es una diferencia y por tanto amplifica.

    Nada de esto invalida el resultado -- el orden de magnitud, el reparto entre los tres flujos y la
    comparación entre frontera mediana y mejor cuartil no se mueven --, pero sí obliga a dos cosas:
    publicar la cifra con el entorno que la produjo y regenerarla en la máquina de referencia. Este
    sello sirve para lo primero.
    """
    import platform
    import numpy, pandas, sklearn, xgboost

    pd.DataFrame([{
        "generado": pd.Timestamp.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sistema": f"{platform.system()} {platform.machine()}",
        "cpus_logicas": os.cpu_count(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }]).to_csv(f"{INFORMES}/14_entorno_de_generacion.csv", index=False)


def _misma_rejilla(a: pd.DataFrame, b: pd.DataFrame, que: str) -> None:
    """Falla en voz alta si dos tablas dejan de estar alineadas fila a fila por (IMO, año)."""
    if len(a) != len(b):
        raise AssertionError(f"{que}: {len(a)} filas frente a {len(b)}; ya no son la misma rejilla.")
    for col in ("ship_imo_number", "reporting_year"):
        if not (a[col].to_numpy() == b[col].to_numpy()).all():
            raise AssertionError(f"{que}: '{col}' no coincide fila a fila; el orden ha cambiado.")


def main() -> None:
    ml, raw, pack = _cargar()
    # Guarda de alineacion: varios bloques de este script y de sus hermanos suponen que `ml` y `raw`
    # son la misma tabla fila a fila. Las dos salen de la misma pasada de feature_engineering, pero
    # se comprueba: si dejara de serlo, los tipos de buque y los factores de ambito se mezclarian
    # sin que nada fallara.
    _misma_rejilla(ml, raw, "mrv_features_operacional_tamano_{ml,raw}")

    print("Prediccion fuera de muestra para las 106.521 filas (3 ajustes)...", flush=True)
    pred = economia.prediccion_honesta(ml, pack)
    d = raw.merge(pred, on=["ship_imo_number", "reporting_year"], how="left", validate="one_to_one")
    d = d[(d[economia.COL_TARGET] > 0) & (d["pred_honesta"] > 0) & (d[economia.COL_HORAS] > 0)].copy()

    d, diag = economia.residuo_persistente(d)
    print(f"  persistencia: corr(t, t-1) = {diag['corr_lag1_pearson']:.3f}"
          f" (Spearman {diag['corr_lag1_spearman']:.3f}), ICC = {diag['icc']:.3f}")
    pd.DataFrame([diag]).to_csv(f"{INFORMES}/14_persistencia_residuo.csv", index=False)

    a = d[d["reporting_year"] == ANIO].copy()
    a["grupo"] = economia.grupos_de_comparacion(a)

    # --- columnas regulatorias, reutilizando los modulos ya verificados
    completo = pd.read_parquet(f"{DATOS}/mrv_consolidado.parquet")
    completo = completo[completo["reporting_year"] == ANIO]
    # Dos precauciones:
    #  1. `reporting_year` es numerica, asi que entraria en la lista de columnas a sumar: un buque con
    #     dos informes saldria con reporting_year = 4050 y con tres, 6075. Esas filas no casarian en
    #     el merge posterior y 1.006 buques -8,085 Mt, el 5,4% de la flota- se quedarian sin columnas
    #     regulatorias y con el factor de ambito imputado por la mediana.
    #  2. Sumar informes completos y parciales duplica emisiones. Es la misma poblacion 'Full' con la
    #     que se entreno el modelo y la que usa la capa regulatoria, asi que las dos capas agregan
    #     sobre lo mismo y sus cifras son comparables fila a fila.
    completo = ets.solo_informes_completos(completo)
    numericas = [c for c in completo.columns
                 if completo[c].dtype.kind in "fi" and c != "reporting_year"]
    agregado = completo.groupby(["ship_imo_number", "reporting_year"], as_index=False)[numericas].sum(min_count=1)
    identidad = (completo[["ship_imo_number", "reporting_year", "company_imo_number"]]
                 .dropna().drop_duplicates(["ship_imo_number", "reporting_year"]))
    a = a.merge(agregado, on=["ship_imo_number", "reporting_year"], how="left",
                suffixes=("", "_mrv"), validate="one_to_one")
    a = a.merge(identidad, on=["ship_imo_number", "reporting_year"], how="left",
                validate="one_to_one")

    # `ets.factor_ambito` ya recorta a 1,0 los factores que el redondeo de la fuente deja en
    # 1,0008 (ver `ets.TOLERANCIA_AMBITO`). Lo que sigue en NaN es un buque sin
    # emision total declarada, y a ese se le imputa la mediana; el pasaporte dice cual es cual.
    factor = ets.factor_ambito(a)
    a["factor_ambito_fuente"] = np.where(factor.notna(), "declarado", "imputado_mediana")
    a["factor_ambito"] = factor.fillna(factor.median())
    n_imputados = int((a["factor_ambito_fuente"] == "imputado_mediana").sum())
    print(f"  factor de ambito: {len(a) - n_imputados} declarados, {n_imputados} imputados por la "
          f"mediana ({100 * n_imputados / len(a):.1f}%)")
    reg = fueleu.calcular(a, anio_regimen=ANIO, factor_ambito=a["factor_ambito"])
    a["penalizacion_fueleu_eur"] = reg["penalizacion_eur"].where(reg["calculable"], 0.0)
    a["balance_fueleu_gco2eq"] = reg["balance_gco2eq"].where(reg["calculable"])
    precio = ets.PRECIO_EUA_POR_DEFECTO
    # La factura sale de `ets.factura`, que es la unica funcion del proyecto que la calcula: base de
    # CO2 equivalente desde 2026 (entran CH4 y N2O) por el factor de entrega pleno del regimen.
    # Calcularla como emision total x factor de ambito x precio se saltaria el recargo de CO2eq y
    # daria 7.003 M EUR en vez de 7.462.
    a["ets_2026_eur"] = ets.factura(a, 2026, precio)

    # --- 14 · ahorro a actividad constante
    filas, grupos = [], None
    for etiqueta, cuantil in (("mediana", 0.50), ("mejor_cuartil", 0.25)):
        for col, nombre in (("residuo_w", "ingenuo"), ("residuo_persistente", "persistente")):
            f = economia.frontera_eficiencia(a, col_residuo=col, cuantil=cuantil)
            valor = economia.valor_economico(f, f["ahorro_actividad_constante_t"], precio)
            valor.update(frontera=etiqueta, residuo=nombre,
                         pct_emisiones=100 * valor["ahorro_t"] / f[economia.COL_TARGET].sum())
            filas.append(valor)
            if col == "residuo_persistente" and cuantil == 0.50:
                a["ahorro_actividad_constante_t"] = f["ahorro_actividad_constante_t"]
                a["frontera"] = f["frontera"]
                grupos = f.groupby("grupo").agg(
                    buques=("ship_imo_number", "size"),
                    capacidad_mediana=("capacidad_estimada", "median"),
                    frontera=("frontera", "first"),
                    co2_t=(economia.COL_TARGET, "sum"),
                    ahorro_t=("ahorro_actividad_constante_t", "sum"))
                grupos["ahorro_pct"] = 100 * grupos["ahorro_t"] / grupos["co2_t"]
    resumen = pd.DataFrame(filas)
    resumen.to_csv(f"{INFORMES}/14_ahorro_actividad_constante.csv", index=False)
    grupos.sort_values("ahorro_t", ascending=False).to_csv(f"{INFORMES}/14_frontera_por_grupo.csv")

    base = resumen[(resumen.frontera == "mediana") & (resumen.residuo == "persistente")].iloc[0]
    print(f"  ahorro a actividad constante (mediana, persistente): {base.ahorro_t/1e6:.3f} Mt"
          f" ({base.pct_emisiones:.2f}%), {base.total_eur/1e6:,.0f} M EUR")

    sensibilidad = []
    for etiqueta, cuantil in (("mediana", 0.50), ("mejor_cuartil", 0.25)):
        f = economia.frontera_eficiencia(a, cuantil=cuantil)
        for bunker in (400, 500, 600, 700):
            v = economia.valor_economico(f, f["ahorro_actividad_constante_t"], precio, bunker)
            v["frontera"] = etiqueta
            sensibilidad.append(v)
    pd.DataFrame(sensibilidad).to_csv(f"{INFORMES}/14_sensibilidad_precios.csv", index=False)

    # --- 15 · palancas: cuanto vale un dia de mar evitado
    modelos = simulador.cargar_modelos(os.path.join(RAIZ, "models"), "operacional_tamano")
    ml_anio = ml[ml["reporting_year"] == ANIO].reset_index(drop=True)
    raw_anio = raw[raw["reporting_year"] == ANIO].reset_index(drop=True)
    _misma_rejilla(ml_anio, raw_anio, f"filas de {ANIO} de ml y raw")
    clave = ml_anio[["ship_imo_number"]].merge(
        a[["ship_imo_number", "factor_ambito"]].drop_duplicates("ship_imo_number"),
        on="ship_imo_number", how="left", validate="one_to_one")
    famb = clave["factor_ambito"].fillna(a["factor_ambito"].median()).to_numpy()
    tipos = raw_anio["ship_type_agrupado"]
    prediccion_base = simulador.predecir(modelos, ml_anio, "xgb")

    palancas = []
    for etiqueta, (dd, dv) in {"millas_-10pct": (-0.10, 0.0), "velocidad_-10pct": (0.0, -0.10)}.items():
        escenario, _ = simulador.aplicar_escenario(ml_anio, dd, dv)
        ahorro = prediccion_base - simulador.predecir(modelos, escenario, "xgb")
        horas = (ml_anio[economia.COL_HORAS] - escenario[economia.COL_HORAS]).to_numpy()
        t = pd.DataFrame({"tipo": tipos, "ahorro_t": ahorro, "horas_evitadas": horas, "factor_ambito": famb})
        t["ets_eur"] = t.ahorro_t * t.factor_ambito * precio
        t["combustible_t"] = economia.toneladas_de_fuel(t.ahorro_t)
        g = t.groupby("tipo").agg(ahorro_t=("ahorro_t", "sum"), horas_evitadas=("horas_evitadas", "sum"),
                                  ets_eur=("ets_eur", "sum"), combustible_t=("combustible_t", "sum"))
        g["valor_eur"] = g.ets_eur + g.combustible_t * economia.PRECIO_BUNKER_EUR_POR_T
        g["eur_por_dia_mar"] = g.apply(lambda r: economia.breakeven_dia_mar(r.valor_eur, r.horas_evitadas), axis=1)
        g["palanca"] = etiqueta
        palancas.append(g.reset_index())
        print(f"  {etiqueta}: {g.ahorro_t.sum()/1e6:.3f} Mt, "
              f"{economia.breakeven_dia_mar(g.valor_eur.sum(), g.horas_evitadas.sum()):,.0f} EUR/dia de mar")
    pd.concat(palancas).to_csv(f"{INFORMES}/15_palancas_eur_por_dia.csv", index=False)

    # --- 16 · la cuenta con dueno
    companias = economia.exposicion_por_compania(a, precio)
    companias.to_csv(f"{INFORMES}/16_navieras_exposicion.csv")
    mitad = int((companias.cuota_acumulada_pct < 50).sum()) + 1
    print(f"  {len(companias)} navieras; {mitad} pagan la mitad de la factura de {companias.factura_eur.sum()/1e6:,.0f} M EUR")
    pool = economia.compensacion_interna_fueleu(a)
    pd.DataFrame([pool]).to_csv(f"{INFORMES}/16_fueleu_pooling.csv", index=False)
    print(f"  FuelEU: solo el {pool['pct_deficit_neteable']:.1f}% del deficit se compensa dentro de la naviera")

    # --- 17 · el ano de caducidad de la operacion actual
    # La calibracion del proxy de capacidad se estima sobre la flota de 2019, asi que necesita el
    # historico completo: pasarle solo el ano en curso devolveria un diccionario vacio y el CII
    # saldria sin calibrar, sistematicamente pesimista.
    factores = cii.calibrar(d)
    print(f"  calibracion del proxy por tipo: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(factores.items())))
    calificado = cii.calcular(a, factor_capacidad=factores)
    # Los dos mismos filtros regulatorios que aplica `analisis_regulatorio.bloque_cii`, y por el
    # mismo motivo (asi los dos scripts publican sobre la misma poblacion):
    #  - El CII de la regla 28 del anexo VI de MARPOL cubre desde 5.000 GT. Los 3.274 buques que el
    #    Reglamento (UE) 2023/957 mete en el MRV desde 2025 no estan calificados por nadie, asi que
    #    ponerles banda A-E en un fichero que la API sirve como "pasaporte de cumplimiento" es
    #    atribuirles una calificacion regulatoria que no tienen.
    #  - Los tipos sin calibracion fiable (`cii.TIPOS_NO_FIABLES`) entrarian con k = 1,0, es decir,
    #    sistematicamente pesimistas.
    # La poblacion regulada se marca sobre el marco COMPLETO, no sobre 2025: un buque veterano puede
    # tener un ano sin distancia declarada y seguir siendo veterano.
    veterano_por_buque = cii.marcar_poblacion_regulada(d).groupby(d["ship_imo_number"]).max()
    calificado["poblacion_regulada"] = calificado["ship_imo_number"].map(veterano_por_buque).fillna(False)
    calificado["cii_aplicable"] = (calificado["cii_calificable"]
                                   & calificado["poblacion_regulada"]
                                   & calificado["cii_tipo"].isin(factores))
    fuera = int((calificado["cii_calificable"] & ~calificado["cii_aplicable"]).sum())
    print(f"  CII: {int(calificado['cii_aplicable'].sum())} buques calificables y dentro del ambito "
          f"de MARPOL; {fuera} calculables pero fuera de ambito, que NO se califican")
    calificable = calificado[calificado["cii_aplicable"]].copy()
    trayectoria = economia.trayectoria_cii(calificable, range(ANIO, 2031))
    columnas = (["ship_imo_number", "ship_name", "company_name", "cii_tipo", "cii_capacidad",
                 "cii_attained", "anio_caida"] + [f"banda_{y}" for y in range(ANIO, 2031)])
    trayectoria[columnas].to_csv(f"{INFORMES}/17_cii_anio_caida.csv", index=False)
    reparto = pd.DataFrame({f"banda_{y}": trayectoria[f"banda_{y}"].value_counts(normalize=True).mul(100)
                            for y in range(ANIO, 2031)}).reindex(list("ABCDE")).round(2)
    reparto.to_csv(f"{INFORMES}/17_cii_trayectoria_bandas.csv")
    # --- 18 · la ficha por buque, que es lo que sirve la API como "pasaporte de cumplimiento"
    ficha = a[[
        "ship_imo_number", "ship_name", "company_name", "company_imo_number", "ship_type_agrupado",
        "reporting_year", "capacidad_estimada", economia.COL_TARGET, "pred_honesta", "origen",
        "residuo_rel", "residuo_persistente", "grupo", "frontera",
        "ahorro_actividad_constante_t", "factor_ambito", "factor_ambito_fuente", "ets_2026_eur",
        "penalizacion_fueleu_eur", "balance_fueleu_gco2eq", economia.COL_HORAS,
        "distancia_nm", "velocidad_nudos",
    ]].copy()
    ficha["ghgie_wtw"] = reg["ghgie_wtw"].where(reg["calculable"])
    ficha["objetivo_fueleu"] = reg["objetivo"]
    ficha["cumple_fueleu"] = reg["cumple"].where(reg["calculable"])
    ficha["combustible_evitado_t"] = economia.toneladas_de_fuel(ficha["ahorro_actividad_constante_t"])
    ficha["ahorro_eur"] = (ficha["ahorro_actividad_constante_t"] * ficha["factor_ambito"] * precio
                           + ficha["combustible_evitado_t"] * economia.PRECIO_BUNKER_EUR_POR_T)
    columnas_cii = ["cii_tipo", "cii_capacidad", "cii_attained", "cii_requerido", "anio_caida"] + [f"banda_{y}" for y in range(ANIO, 2031)]
    ficha = ficha.merge(
        trayectoria[["ship_imo_number"] + columnas_cii].drop_duplicates("ship_imo_number"),
        on="ship_imo_number", how="left", validate="one_to_one")
    # Un buque fuera del ambito del CII sale del pasaporte SIN banda y con la razon escrita, en vez
    # de con una calificacion que la norma no le da.
    ficha = ficha.merge(
        calificado[["ship_imo_number", "cii_aplicable", "poblacion_regulada"]]
        .drop_duplicates("ship_imo_number"),
        on="ship_imo_number", how="left", validate="one_to_one")
    ficha["cii_motivo_sin_banda"] = np.where(
        ficha["cii_aplicable"].fillna(False), None,
        np.where(~ficha["poblacion_regulada"].fillna(False),
                 "fuera del ambito de MARPOL (entra en el MRV con el Reglamento 2023/957)",
                 "sin capacidad, sin distancia o tipo sin calibracion fiable"))
    ficha.to_csv(f"{INFORMES}/18_pasaporte_buques.csv", index=False)
    print(f"  ficha por buque: {len(ficha)} filas, {ficha.anio_caida.notna().sum()} con fecha de caida, "
          f"{int((~ficha['cii_aplicable'].fillna(False)).sum())} sin banda CII por no aplicarles")

    _sello_entorno()

    sanos = trayectoria[trayectoria[f"banda_{ANIO}"].isin(list("ABC"))]
    caen = int(sanos["anio_caida"].notna().sum())
    print(f"  CII: de {len(sanos)} buques hoy en A-C, {caen} ({100*caen/len(sanos):.0f}%) caen a D/E antes de 2031")
    print("\nInformes 14-17 escritos en reports/.")


if __name__ == "__main__":
    main()
