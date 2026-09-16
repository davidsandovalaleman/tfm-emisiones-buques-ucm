"""Regenera la capa regulatoria: los once informes `11_*`, `12_*` y `13_*` de `reports/`.

    python scripts/analisis_economia.py        # primero: escribe 18_pasaporte_buques.csv
    python scripts/analisis_regulatorio.py [--salida DIR]

**Orden de ejecucion.** La conciliacion compara el registro contra el pasaporte
que publica la capa economica, no contra una reconstruccion propia, asi que `analisis_economia.py`
va antes. Si no esta, la conciliacion lo dice en vez de sustituirlo por otra cosa.

**Por que existe.** La reproducibilidad es un argumento explicito de este trabajo -- los cuadernos
se ejecutan en orden sin una celda sin salida --, y la cifra de cabecera que publica el README, la
factura de cumplimiento de la flota europea, tiene que poder rehacerse digito a digito desde un
script, no solo existir como CSV.

**Que no hace.** No reentrena nada ni toca el `.pickle` verificado. Lee los artefactos del notebook 03
y el registro consolidado, llama a `cii.py`, `ets.py` y `fueleu.py` y reescribe los once ficheros. El unico modelo que carga es el de
produccion, y solo para los escenarios, exactamente igual que hace `analisis_economia.py`.

**La union de las dos tablas.** El registro publica desde 2024
informes parciales (art. 11(2)): un buque que cambia de compania a mitad de ano declara dos tramos.
Sumar completos y parciales duplica emisiones, asi que se agrega solo sobre `Full`, que es la misma
poblacion con la que se entreno el modelo. Y `reporting_year` se excluye de las columnas que se
suman: es numerica, entraria en el `groupby(...).sum()` y un buque con dos informes saldria con el
ano 4050, con lo que no casaria en la union y perderia todas sus columnas regulatorias.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import cii, ets, fueleu, simulador  # noqa: E402

DATOS = os.path.join(RAIZ, "data", "processed")
ANIO = 2025
PRECIO = ets.PRECIO_EUA_POR_DEFECTO


def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (features + columnas del registro, registro completo del ano en curso)."""
    raw = pd.read_parquet(f"{DATOS}/mrv_features_operacional_tamano_raw.parquet")
    comp = pd.read_parquet(f"{DATOS}/mrv_consolidado.parquet")
    comp = ets.solo_informes_completos(comp)
    numericas = [c for c in comp.columns
                 if comp[c].dtype.kind in "fi" and c != "reporting_year"]
    agregado = comp.groupby(["ship_imo_number", "reporting_year"],
                            as_index=False)[numericas].sum(min_count=1)
    texto = ["ship_type", "report_coverage"]
    texto = [c for c in texto if c in comp.columns]
    if texto:
        etiquetas = (comp[["ship_imo_number", "reporting_year"] + texto]
                     .drop_duplicates(["ship_imo_number", "reporting_year"]))
        agregado = agregado.merge(etiquetas, on=["ship_imo_number", "reporting_year"], how="left")
    d = raw.merge(agregado, on=["ship_imo_number", "reporting_year"],
                  how="left", suffixes=("", "_mrv"), validate="one_to_one")
    sin_par = d[ets.COL_ETS_CO2].isna().sum() if ets.COL_ETS_CO2 in d else -1
    print(f"  union: {len(d)} filas, {len(d[d.reporting_year == ANIO])} de {ANIO}; "
          f"{sin_par} sin columna de ambito declarada (anteriores a 2024)")
    return d, comp[comp["reporting_year"] == ANIO]


# ------------------------------------------------------------------------------------------
# 11 · CII
# ------------------------------------------------------------------------------------------
def bloque_cii(d: pd.DataFrame, salida: str) -> pd.DataFrame:
    factores = cii.calibrar(d)
    pd.DataFrame({"tipo": list(factores),
                  "k_capacidad": [round(v, 4) for v in factores.values()],
                  "utilizacion_implicita_pct": [round(100 / v, 1) for v in factores.values()]}
                 ).to_csv(f"{salida}/11_cii_calibracion.csv", index=False)

    q = cii.calcular(d, factor_capacidad=factores)
    # Dos filtros, los dos con motivo regulatorio y no estadistico:
    #  - `marcar_poblacion_regulada` deja fuera los 3.274 buques que el Reglamento 2023/957 mete en
    #    el MRV a partir de 2025. El CII de MARPOL cubre desde 5.000 GT, asi que calificarlos
    #    mezclaria dos poblaciones y rompería la serie interanual. Se calcula sobre el marco
    #    completo, no sobre las filas calificables, porque un buque veterano puede tener un ano sin
    #    distancia declarada.
    #  - Solo los tipos con calibracion fiable: metaneros y de carga combinada quedan fuera por lo
    #    que documenta `cii.TIPOS_NO_FIABLES`.
    q["poblacion_regulada"] = cii.marcar_poblacion_regulada(q)
    cal = q[q["cii_calificable"] & q["poblacion_regulada"]
            & q["cii_tipo"].isin(factores)].copy()
    bandas = (pd.crosstab(cal["reporting_year"], cal["cii_banda"], normalize="index")
              .mul(100).round(1)[list("ABCDE")])
    bandas.to_csv(f"{salida}/11_cii_bandas_por_anio.csv")

    cal["quintil_tam"] = (cal.groupby("cii_tipo")["cii_capacidad"]
                          .transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop") + 1))
    g = cal.groupby(["cii_tipo", "quintil_tam"]).agg(
        n=("cii_attained", "size"),
        cap_mediana=("cii_capacidad", "median"),
        ratio_mediano=("cii_ratio", "median"),
        pct_D_E=("cii_banda", lambda s: 100 * s.isin(["D", "E"]).mean()))
    g = g.round({"cap_mediana": 2, "ratio_mediano": 2, "pct_D_E": 2}).reset_index()
    g.to_csv(f"{salida}/11_cii_sesgo_tamano.csv", index=False)

    # El titulo no puede ser "El CII penaliza al buque pequeno de cada tipo", porque no es lo que
    # dice el fichero. Medido tipo a tipo con la correlacion de Spearman
    # entre quintil y % de D/E: gaseros -1,00, carga general -0,90, portacontenedores y petroleros
    # -0,70; graneleros -0,30 (sin gradiente) y frigorificos +0,70 (al reves). El efecto es real y
    # grande donde esta, y el titulo dice en cuantos tipos esta. Los tipos donde el gradiente es monotono se marcan en el propio grafico.
    from scipy.stats import spearmanr
    rho = {t: float(spearmanr(s.sort_values("quintil_tam")["quintil_tam"],
                              s.sort_values("quintil_tam")["pct_D_E"]).statistic)
           for t, s in g.groupby("cii_tipo")}
    limpios = sorted(t for t, r in rho.items() if r <= -0.7)
    pd.DataFrame({"cii_tipo": list(rho), "rho_quintil_pct_D_E": [round(rho[t], 2) for t in rho],
                  "gradiente_limpio": [rho[t] <= -0.7 for t in rho]}
                 ).sort_values("rho_quintil_pct_D_E").to_csv(
        f"{salida}/11_cii_gradiente_por_tipo.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    for tipo, sub in g.groupby("cii_tipo"):
        sub = sub.sort_values("quintil_tam")
        limpio = rho[tipo] <= -0.7
        ax.plot(sub["quintil_tam"], sub["pct_D_E"], marker="o",
                lw=2.2 if limpio else 1.2, alpha=1.0 if limpio else 0.55,
                ls="-" if limpio else "--",
                label=f"{tipo} (ρ={rho[tipo]:+.2f})")
    ax.set_xlabel("Quintil de capacidad (1 = los más pequeños de su tipo)")
    ax.set_ylabel("% de buque-año en banda D o E")
    ax.set_title(f"El CII penaliza al buque pequeño en {len(limpios)} de los {len(rho)} tipos "
                 f"calificados")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=2, title="línea continua = gradiente monótono (ρ ≤ −0,7)",
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{salida}/11_cii_sesgo_tamano.png", dpi=150)
    plt.close(fig)
    return cal


# ------------------------------------------------------------------------------------------
# 12 · EU ETS
# ------------------------------------------------------------------------------------------
def bloque_ets(d: pd.DataFrame, registro: pd.DataFrame, salida: str) -> dict:
    filas = []
    for anio in (2024, 2025, 2026):
        base = ets.base_imponible(registro, anio).sum()
        filas.append({"anio_regimen": anio,
                      "base_imponible_Mt": round(base / 1e6, 2),
                      "factor_entrega": ets.factor_entrega(anio),
                      "mil_millones_EUR": round(ets.coste(base, anio, PRECIO) / 1e9, 3)})
    pd.DataFrame(filas).to_csv(f"{salida}/12_ets_factura_flota.csv", index=False)

    a = d[d["reporting_year"] == ANIO].copy()
    f = ets.factor_ambito(a)
    a["factor_ambito"] = f.clip(0, 1)
    (a.groupby("ship_type_agrupado")
       .agg(n=("factor_ambito", "size"), f_ets_mediano=("factor_ambito", "median"))
       .round({"f_ets_mediano": 3})
       .to_csv(f"{salida}/12_ets_ambito_por_tipo.csv"))
    return {"registro": registro, "a": a}


def bloque_escenarios(a: pd.DataFrame, salida: str) -> pd.DataFrame:
    """Los tres escenarios del simulador, traducidos a derechos. Usa el modelo de produccion."""
    ml = pd.read_parquet(f"{DATOS}/mrv_features_operacional_tamano_ml.parquet")
    ml = ml[ml["reporting_year"] == ANIO].reset_index(drop=True)
    # Guarda de alineacion: mas abajo `tipos` se toma de `a` por POSICION contra las filas de `ml`,
    # asi que se comprueba que coinciden.
    a_ord = a.reset_index(drop=True)
    if len(a_ord) != len(ml) or not (a_ord["ship_imo_number"].to_numpy()
                                     == ml["ship_imo_number"].to_numpy()).all():
        raise AssertionError(
            f"las filas de {ANIO} de ml y del marco regulatorio ya no estan alineadas fila a fila "
            f"({len(ml)} frente a {len(a_ord)}): revisa el orden antes de fiarte de `tipos`.")
    modelos = simulador.cargar_modelos(os.path.join(RAIZ, "models"), "operacional_tamano")
    base = simulador.predecir(modelos, ml, "xgb")
    famb = (ml[["ship_imo_number"]]
            .merge(a[["ship_imo_number", "factor_ambito"]].drop_duplicates("ship_imo_number"),
                   on="ship_imo_number", how="left", validate="one_to_one")["factor_ambito"])
    famb = famb.fillna(famb.median()).to_numpy()

    def ahorro(escenario):
        return base - simulador.predecir(modelos, escenario, "xgb")

    esc10, _ = simulador.aplicar_escenario(ml, -0.10, 0.0)
    a10 = ahorro(esc10)
    # El esfuerzo del escenario focalizado es el mismo que publica el notebook 08: las horas que evita un
    # recorte del 5% de millas en toda la flota. No son las del -10%: concentrar 3,66 M h en cuatro
    # tipos que solo navegan 10,28 M h exigiria recortarles un 36% de sus millas, una extrapolacion
    # que el modelo no puede sostener. Con 1,83 M h el recorte focalizado es del 17,8%, dentro del
    # rango observado.
    esc5, _ = simulador.aplicar_escenario(ml, -0.05, 0.0)
    horas = float((ml[simulador.COL_HORAS] - esc5[simulador.COL_HORAS]).sum())
    tipos = a_ord["ship_type_agrupado"]
    foc = simulador.escenario_focalizado(modelos, ml, tipos, horas)
    etiqueta = f"{horas/1e6:.2f} M h evitadas".replace(".", ",")
    filas = [("-10% de millas (toda la flota)", float(np.asarray(a10).sum())),
             (f"{etiqueta}, repartidas", float(foc["ahorro_uniforme_t"])),
             (f"{etiqueta}, focalizadas", float(foc["ahorro_focalizado_t"]))]
    # La fraccion del ahorro que cae dentro del ambito del ETS se mide sobre el escenario del -10%,
    # que es el unico con ahorro fila a fila; para los otros dos se aplica esa misma proporcion,
    # porque las tres alternativas reparten el recorte entre los mismos buques.
    a10 = np.asarray(a10, dtype="float64")
    fraccion_ambito = float((a10 * famb).sum() / a10.sum())
    salidas = []
    for nombre, total in filas:
        en_ambito = total * fraccion_ambito
        salidas.append({"escenario": nombre,
                        "ahorro_Mt": round(total / 1e6, 3),
                        "ahorro_en_ambito_ETS_Mt": round(en_ambito / 1e6, 3),
                        "pct_del_ahorro_en_ambito": round(100 * fraccion_ambito, 1),
                        "M_EUR_regimen_2025": round(ets.coste(en_ambito, 2025, PRECIO) / 1e6, 1),
                        "M_EUR_regimen_2026": round(ets.coste(en_ambito, 2026, PRECIO) / 1e6, 1)})
    out = pd.DataFrame(salidas)
    out.to_csv(f"{salida}/12_ets_escenarios.csv", index=False)
    return out


# ------------------------------------------------------------------------------------------
# 13 · FuelEU Maritime
# ------------------------------------------------------------------------------------------
def bloque_fueleu(a: pd.DataFrame, salida: str, escenarios: pd.DataFrame) -> None:
    reg = fueleu.calcular(a, anio_regimen=ANIO, factor_ambito=a["factor_ambito"].fillna(
        a["factor_ambito"].median()))
    v = a.assign(ghgie_wtw=reg["ghgie_wtw"], cumple=reg["cumple"],
                 penal=reg["penalizacion_eur"].where(reg["calculable"], np.nan),
                 calculable=reg["calculable"])
    v = v[v["calculable"]]
    col_tipo = "ship_type" if "ship_type" in v.columns else "ship_type_agrupado"
    (v.groupby(col_tipo)
       .agg(n=("ghgie_wtw", "size"), ghgie_mediana=("ghgie_wtw", "median"),
            pct_cumple=("cumple", lambda s: 100 * s.mean()),
            penalizacion_M_EUR=("penal", lambda s: s.sum() / 1e6))
       .round({"ghgie_mediana": 2, "pct_cumple": 1, "penalizacion_M_EUR": 2})
       .sort_values("penalizacion_M_EUR", ascending=False)
       .rename_axis("ship_type")
       .to_csv(f"{salida}/13_fueleu_por_tipo.csv"))

    filas = []
    for anio in (2025, 2030, 2035, 2040, 2045, 2050):
        r = fueleu.calcular(a, anio_regimen=anio,
                            factor_ambito=a["factor_ambito"].fillna(a["factor_ambito"].median()))
        ok = r[r["calculable"]]
        filas.append({"anio_objetivo": anio,
                      "objetivo_gco2eq_MJ": round(fueleu.objetivo(anio), 3),
                      "pct_flota_que_cumple": round(100 * ok["cumple"].mean(), 1),
                      "penalizacion_mil_millones_EUR": round(ok["penalizacion_eur"].sum() / 1e9, 3)})
    pd.DataFrame(filas).to_csv(f"{salida}/13_fueleu_trayectoria.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ok = reg[reg["calculable"]]
    ax.hist(ok["ghgie_wtw"].clip(upper=100), bins=80, color="#2f6f9f")
    ax.axvline(fueleu.objetivo(ANIO), color="#c1121f", lw=2,
               label=f"objetivo {ANIO}: {fueleu.objetivo(ANIO):.2f} gCO₂eq/MJ")
    ax.set_xlabel("Intensidad GEI de tanque a estela + pozo a tanque (gCO₂eq/MJ)")
    ax.set_ylabel("Buques")
    ax.set_title(f"Solo el {100*ok['cumple'].mean():.1f}% de la flota cumple FuelEU en {ANIO}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{salida}/13_fueleu_intensidad.png", dpi=150)
    plt.close(fig)

    # Coste de cumplimiento evitado del escenario de cabecera
    base = escenarios.iloc[0]
    ahorro_rel = base["ahorro_Mt"] * 1e6 / a[ets.COL_TOTAL].sum()
    fueleu_evitado = ok["penalizacion_eur"].sum() * ahorro_rel
    ets_evitado = base["M_EUR_regimen_2026"] * 1e6
    pd.DataFrame([{"escenario": "-10% de millas",
                   "co2_Mt": base["ahorro_Mt"],
                   "ets_M_EUR": round(ets_evitado / 1e6, 1),
                   "fueleu_M_EUR": round(fueleu_evitado / 1e6, 1),
                   "total_M_EUR": round((ets_evitado + fueleu_evitado) / 1e6, 1)}]
                 ).to_csv(f"{salida}/13_cumplimiento_escenario.csv", index=False)


# ------------------------------------------------------------------------------------------
def conciliacion(registro: pd.DataFrame, a: pd.DataFrame, salida: str) -> None:
    """Deja escrito, en un fichero, por que la factura tiene dos agregados y en cuanto difieren.

    La segunda fila no se calcula sobre `a`, la tabla de variables del ano entera: la capa economica
    filtra las filas sin emision, sin horas o sin prediccion y publica sobre 14.863 buques, no sobre
    16.939. Por eso se lee de `18_pasaporte_buques.csv`, que es **lo que la capa economica publica
    de verdad**. Si el pasaporte no existe todavia, se dice y no se inventa un sustituto.
    Implica un orden: `analisis_economia.py` antes que este script.
    """
    reg_base = ets.base_imponible(registro, 2026).sum()
    filas = [
        {"perimetro": "capa regulatoria (informes Full del registro)",
         "buques": registro["ship_imo_number"].nunique(),
         "co2_total_Mt": round(registro[ets.COL_TOTAL].sum() / 1e6, 3),
         "base_imponible_2026_Mt": round(reg_base / 1e6, 3),
         "ets_2026_M_EUR": round(ets.coste(reg_base, 2026, PRECIO) / 1e6, 1)},
    ]

    pasaporte = os.path.join(salida, "18_pasaporte_buques.csv")
    if not os.path.exists(pasaporte):
        print("  conciliacion: falta 18_pasaporte_buques.csv; ejecuta antes analisis_economia.py")
        pd.DataFrame(filas + [{"perimetro": "capa economica: NO CALCULADA "
                                            "(ejecuta antes scripts/analisis_economia.py)"}]
                     ).to_csv(f"{salida}/12_ets_conciliacion.csv", index=False)
        return

    eco = pd.read_csv(pasaporte, low_memory=False)
    filas.append(
        {"perimetro": "capa economica (pasaporte publicado: buques con prediccion y actividad)",
         "buques": eco["ship_imo_number"].nunique(),
         "co2_total_Mt": round(eco[ets.COL_TOTAL].sum() / 1e6, 3),
         # La base imponible de la capa economica es la que ya lleva escrita cada fila del
         # pasaporte, dividida por el factor de entrega pleno y el precio: no se recalcula por otra
         # via: `ets.factura` es la unica funcion que calcula la factura.
         "base_imponible_2026_Mt": round(
             eco["ets_2026_eur"].sum() / (ets.factor_entrega(2026) * PRECIO) / 1e6, 3),
         "ets_2026_M_EUR": round(eco["ets_2026_eur"].sum() / 1e6, 1)})
    t = pd.DataFrame(filas)
    t.loc[len(t)] = {"perimetro": "diferencia (%)",
                     "buques": filas[0]["buques"] - filas[1]["buques"],
                     "co2_total_Mt": round(100 * (filas[1]["co2_total_Mt"] / filas[0]["co2_total_Mt"] - 1), 3),
                     "base_imponible_2026_Mt": round(100 * (filas[1]["base_imponible_2026_Mt"] / filas[0]["base_imponible_2026_Mt"] - 1), 3),
                     "ets_2026_M_EUR": round(100 * (filas[1]["ets_2026_M_EUR"] / filas[0]["ets_2026_M_EUR"] - 1), 3)}
    t.to_csv(f"{salida}/12_ets_conciliacion.csv", index=False)
    print(f"  conciliacion: regulatoria {filas[0]['ets_2026_M_EUR']:,.0f} M EUR vs "
          f"economica {filas[1]['ets_2026_M_EUR']:,.0f} M EUR "
          f"({t.iloc[-1]['ets_2026_M_EUR']:+.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=os.path.join(RAIZ, "reports"))
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    print("Capa regulatoria: CII, EU ETS y FuelEU sobre el registro publico.", flush=True)
    d, registro = cargar()
    cal = bloque_cii(d, args.salida)
    print(f"  CII: {cal['ship_imo_number'].nunique()} buques calificados, "
          f"{len(cal)} filas buque-ano")
    ctx = bloque_ets(d, registro, args.salida)
    esc = bloque_escenarios(ctx["a"], args.salida)
    print(f"  ETS: escenario de cabecera {esc.iloc[0]['ahorro_Mt']} Mt, "
          f"{esc.iloc[0]['M_EUR_regimen_2026']} M EUR en regimen 2026")
    bloque_fueleu(ctx["a"], args.salida, esc)
    conciliacion(registro, ctx["a"], args.salida)
    print(f"\nOnce informes reescritos en {args.salida}.")


if __name__ == "__main__":
    main()
