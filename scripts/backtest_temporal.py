"""Backtest temporal: con datos hasta 2024, ¿cuánto se equivoca la factura de 2025?

    python scripts/backtest_temporal.py [--salida DIR]

**Qué prueba y por qué es distinta de todo lo anterior.** Las métricas del trabajo se miden sobre
una partición 80/20 agrupada por buque: entrena con unos buques y predice otros, con los mismos
años en los dos lados. Es la validación correcta para responder «¿cuánto emite *este* buque?», pero
no responde a la pregunta que hace quien va a usar el sistema: «si te doy lo que sé hoy, ¿aciertas
lo que va a pasar el año que viene?». Eso solo lo contesta una partición **temporal**: entrenar con
2018-2024 y predecir 2025 sin haberlo visto nunca.

Además, las métricas se traducen a euros: las
predicciones se pasan por `ets.factura`, el mismo módulo que calcula la factura publicada, y se
compara **la factura de derechos de emisión predicha con la que sale de lo que los buques
declararon de verdad**. Un error del 2% en euros sobre 7.400 millones dice más que un R² del 0,87.

**Tres cuidados metodológicos, los tres declarados en la salida:**

1. **La población de 2025 no es la de 2024.** El Reglamento (UE) 2023/957 mete en el registro 3.274
   buques nuevos desde 2025. Predecir buques de una clase que el entrenamiento no vio no es un
   error del modelo, es una pregunta distinta, así que la comparación principal se hace sobre los
   buques presentes antes de 2025 (`cii.marcar_poblacion_regulada`) y la otra se reporta aparte.
2. **El modelo de producción no vale para esto**: se entrenó con filas de 2025 dentro. Aquí se
   reentrena desde cero con los mismos hiperparámetros y la misma semilla, solo que sin 2025.
3. **Nadie presupuesta derechos sobre una estimación puntual.** La banda se construye con
   *conformal prediction* dividida: los cuantiles empíricos del error relativo se calibran sobre
   2024 -- el último año que el modelo no usa para nada más -- y se aplican a 2025. Después se
   **mide la cobertura real** sobre 2025, que es la única forma de saber si la banda vale. Es
   aritmética de cuantiles aplicada al residuo, y se contrasta con el error mediano por quintil que
   ya publica el simulador (notebook 08).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import cii, ets  # noqa: E402

DATOS = os.path.join(RAIZ, "data", "processed")
ANIO = 2025
ANIO_CALIBRACION = 2024
NO_PREDICTORAS = ["reporting_year", "ship_imo_number", "ship_name", "company_name",
                  "total_co2_emissions_m_tonnes", "es_train", "target_reconstruido"]
NIVELES = (0.80, 0.90)


def _pack() -> dict:
    with open(f"{RAIZ}/models/modelo_xgb_operacional_tamano.pickle", "rb") as f:
        return pickle.load(f)


def _modelo() -> XGBRegressor:
    """Los hiperparámetros del modelo de producción, leídos de su propio `.pickle`."""
    params = _pack()["modelo"].get_params()
    return XGBRegressor(**{k: v for k, v in params.items()
                           if k in ("objective", "learning_rate", "max_depth", "n_estimators",
                                    "random_state", "tree_method", "gamma", "n_jobs")})


def entrenar_hasta(ml: pd.DataFrame, anio_corte: int) -> tuple[XGBRegressor, list[str]]:
    """Reentrena el modelo **de producción** -- sus hiperparámetros y sus variables -- sin 2025.

    Las predictoras se leen del propio `.pickle`, no se deducen de las columnas de `ml`: esa
    deducción arrastraría `distancia_nm`, que el cuaderno 04 excluye a propósito porque se despeja
    del combustible total, y `veterano`, que añade este mismo script. Se entrenaría con 44 variables
    y se mediría un modelo distinto del que se entrega. El efecto numérico es mínimo (R² de
    veteranos 0,9670 con las 44 frente a 0,9666 con las 42), pero el backtest debe medir
    exactamente el modelo de producción.
    """
    predictoras = list(_pack()["columnas_predictoras"])
    faltan = [c for c in predictoras if c not in ml.columns]
    if faltan:
        raise KeyError(f"faltan en el dataset columnas del modelo de produccion: {faltan}")
    tr = ml[ml["reporting_year"] <= anio_corte]
    m = _modelo().fit(tr[predictoras], tr["total_co2_emissions_m_tonnes"])
    return m, predictoras


def _predecir(m, x) -> np.ndarray:
    return np.clip(np.asarray(m.predict(x), dtype=float), 0.0, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=os.path.join(RAIZ, "reports"))
    args = ap.parse_args()
    sal = args.salida
    os.makedirs(sal, exist_ok=True)

    ml = pd.read_parquet(f"{DATOS}/mrv_features_operacional_tamano_ml.parquet")
    raw = pd.read_parquet(f"{DATOS}/mrv_features_operacional_tamano_raw.parquet")
    # Guarda de alineacion: `veterano` se calcula sobre `raw` y se pega a `ml` por POSICION. Las dos
    # salen de la misma pasada de feature_engineering, y se comprueba: un desajuste marcaria como
    # veteranos a buques que no lo son sin avisar.
    if len(ml) != len(raw) or not (ml["ship_imo_number"].to_numpy()
                                   == raw["ship_imo_number"].to_numpy()).all() \
            or not (ml["reporting_year"].to_numpy() == raw["reporting_year"].to_numpy()).all():
        raise AssertionError("mrv_features_operacional_tamano_{ml,raw} ya no estan alineados fila a fila")
    veterano = cii.marcar_poblacion_regulada(raw)
    ml = ml.assign(veterano=veterano.to_numpy())

    print("Backtest temporal: entrenar <=2024, predecir 2025.", flush=True)
    modelo, predictoras = entrenar_hasta(ml, 2024)
    a = ml[ml["reporting_year"] == ANIO].copy()
    a["pred"] = _predecir(modelo, a[predictoras])
    y = a["total_co2_emissions_m_tonnes"].to_numpy()

    # --- 20 · exactitud, en toneladas y por población
    filas = []
    for etiqueta, sub in (("todos los buques de 2025", a),
                          ("buques ya presentes antes de 2025", a[a["veterano"]]),
                          ("buques nuevos del Reglamento 2023/957", a[~a["veterano"]])):
        if not len(sub):
            continue
        filas.append({
            "poblacion": etiqueta,
            "n": len(sub),
            "co2_real_Mt": round(sub["total_co2_emissions_m_tonnes"].sum() / 1e6, 3),
            "co2_predicho_Mt": round(sub["pred"].sum() / 1e6, 3),
            "sesgo_agregado_pct": round(100 * (sub["pred"].sum() /
                                               sub["total_co2_emissions_m_tonnes"].sum() - 1), 2),
            "R2": round(r2_score(sub["total_co2_emissions_m_tonnes"], sub["pred"]), 4),
            "MAE_t": round(mean_absolute_error(sub["total_co2_emissions_m_tonnes"], sub["pred"]), 1),
            "error_rel_mediano_pct": round(100 * float(np.nanmedian(
                (np.abs(sub["pred"] - sub["total_co2_emissions_m_tonnes"])
                 / sub["total_co2_emissions_m_tonnes"].replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan))), 2),
        })
    exactitud = pd.DataFrame(filas)
    exactitud.to_csv(f"{sal}/20_backtest_exactitud.csv", index=False)
    for f in filas:
        print(f"  {f['poblacion']:<40} n={f['n']:>6}  R2={f['R2']:.4f}  "
              f"sesgo agregado {f['sesgo_agregado_pct']:+.2f}%")

    # --- 21 · la misma prueba, en euros
    comp = pd.read_parquet(f"{DATOS}/mrv_consolidado.parquet")
    comp = ets.solo_informes_completos(comp)
    comp = comp[comp["reporting_year"] == ANIO]
    numericas = [c for c in comp.columns
                 if comp[c].dtype.kind in "fi" and c != "reporting_year"]
    agregado = comp.groupby(["ship_imo_number", "reporting_year"],
                            as_index=False)[numericas].sum(min_count=1)
    b = a.merge(agregado, on=["ship_imo_number", "reporting_year"], how="left", suffixes=("", "_mrv"))
    factor = ets.factor_ambito(b).clip(0, 1)
    b["factor_ambito"] = factor.fillna(factor.median())
    uplift = ets.uplift_co2eq(b)

    filas = []
    for regimen in (2025, 2026):
        real = ets.factura(b, regimen).sum()
        # La factura predicha aplica al CO2 predicho la misma regla: fraccion en ambito, recargo de
        # CO2eq desde 2026 y factor de entrega del regimen.
        pred = (b["pred"] * b["factor_ambito"]
                * (uplift if regimen >= ets.ANIO_CO2EQ else 1.0)
                * ets.factor_entrega(regimen) * ets.PRECIO_EUA_POR_DEFECTO).sum()
        filas.append({"anio_regimen": regimen,
                      "factura_declarada_M_EUR": round(real / 1e6, 1),
                      "factura_predicha_M_EUR": round(pred / 1e6, 1),
                      "error_pct": round(100 * (pred / real - 1), 2)})
    factura = pd.DataFrame(filas)
    factura.to_csv(f"{sal}/21_backtest_factura.csv", index=False)
    f26 = factura[factura.anio_regimen == 2026].iloc[0]
    print(f"  factura del regimen 2026: declarada {f26.factura_declarada_M_EUR:,.0f} M EUR, "
          f"predicha {f26.factura_predicha_M_EUR:,.0f} M EUR ({f26.error_pct:+.2f}%)")

    # --- 22 · banda de prediccion calibrada sobre 2024 y verificada sobre 2025
    modelo_cal, _ = entrenar_hasta(ml, ANIO_CALIBRACION - 1)
    cal = ml[ml["reporting_year"] == ANIO_CALIBRACION].copy()
    cal["pred"] = _predecir(modelo_cal, cal[predictoras])
    # Error relativo al valor predicho: es la escala en la que el error de este modelo es homogeneo
    # (el absoluto crece con el tamano del buque, el relativo no).
    ratio = (cal["total_co2_emissions_m_tonnes"] / cal["pred"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan).dropna()

    filas = []
    for nivel in NIVELES:
        lo, hi = np.quantile(ratio, [(1 - nivel) / 2, 1 - (1 - nivel) / 2])
        dentro = ((y >= a["pred"] * lo) & (y <= a["pred"] * hi))
        filas.append({
            "nivel_nominal_pct": round(100 * nivel, 1),
            "factor_inferior": round(float(lo), 4),
            "factor_superior": round(float(hi), 4),
            "cobertura_real_2025_pct": round(100 * float(dentro.mean()), 2),
            "cobertura_real_veteranos_pct": round(
                100 * float(dentro[a["veterano"].to_numpy()].mean()), 2),
            "anchura_mediana_pct_de_la_prediccion": round(100 * float(hi - lo), 1),
        })
    banda = pd.DataFrame(filas)
    banda.to_csv(f"{sal}/22_backtest_banda.csv", index=False)
    for f in filas:
        print(f"  banda {f['nivel_nominal_pct']:.0f}% nominal -> cobertura real "
              f"{f['cobertura_real_2025_pct']:.1f}% (veteranos "
              f"{f['cobertura_real_veteranos_pct']:.1f}%)")

    # La banda de la FACTURA agregada no es la suma de las bandas por buque: los errores se
    # compensan. Se estima remuestreando buques (bootstrap), que es lo que hace el simulador.
    rng = np.random.default_rng(1234567)
    eur = (b["pred"] * b["factor_ambito"] * uplift * ets.PRECIO_EUA_POR_DEFECTO).to_numpy()
    real_i = (b["total_co2_emissions_m_tonnes"] * b["factor_ambito"] * uplift
              * ets.PRECIO_EUA_POR_DEFECTO).to_numpy()
    n = len(eur)
    muestras = np.array([
        eur[idx].sum() / real_i[idx].sum()
        for idx in (rng.integers(0, n, n) for _ in range(500))])
    p5, p95 = np.quantile(muestras, [0.05, 0.95])
    pd.DataFrame([{"remuestreos": 500,
                   "sesgo_central_pct": round(100 * float(np.median(muestras) - 1), 2),
                   "p5_pct": round(100 * float(p5 - 1), 2),
                   "p95_pct": round(100 * float(p95 - 1), 2)}]
                 ).to_csv(f"{sal}/22_backtest_factura_banda.csv", index=False)
    print(f"  factura agregada: sesgo {100*(np.median(muestras)-1):+.2f}% "
          f"(bootstrap 90%: {100*(p5-1):+.2f}% a {100*(p95-1):+.2f}%)")

    # --- figura
    #  Panel izquierdo: predicho contra declarado, la comprobacion visual de siempre.
    #  Panel derecho: la factura, con la banda del bootstrap dibujada sobre la barra predicha. Sin
    #  la banda, dos barras casi iguales no dicen si el acierto es solido o casualidad.
    TINTA, TINTA_2, DATO, LINEA = "#0b0b0b", "#52514e", "#2a78d6", "#c1121f"
    # Formato espanol: coma decimal y punto de millar.
    def esp(x: float, dec: int = 0) -> str:
        return f"{x:,.{dec}f}".replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")
    fb = pd.read_csv(f"{sal}/22_backtest_factura_banda.csv").iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    v = a[a["veterano"]]
    ax1.scatter(v["total_co2_emissions_m_tonnes"] / 1e3, v["pred"] / 1e3, s=4, alpha=.22,
                color=DATO, edgecolors="none", zorder=3)
    tope = float(np.nanpercentile(v["total_co2_emissions_m_tonnes"] / 1e3, 99.5))
    ax1.plot([0, tope], [0, tope], color=LINEA, lw=1.4, zorder=4)
    ax1.set_xlim(0, tope); ax1.set_ylim(0, tope)
    ax1.set_xlabel("CO₂ declarado en 2025 (kt)", fontsize=9, color=TINTA_2)
    ax1.set_ylabel("CO₂ predicho sin haber visto 2025 (kt)", fontsize=9, color=TINTA_2)
    r2_vet = float(exactitud.loc[exactitud.poblacion.str.startswith("buques ya"), "R2"].iloc[0])
    n_vet = int(exactitud.loc[exactitud.poblacion.str.startswith("buques ya"), "n"].iloc[0])
    ax1.set_title(f"{esp(n_vet)} buques ya presentes antes de 2025 · R² {esp(r2_vet, 3)}",
                  loc="left", fontsize=9.5, color=TINTA, pad=8)

    # Barras con la banda del remuestreo sobre la predicha.
    declarada = float(f26.factura_declarada_M_EUR)
    predicha = float(f26.factura_predicha_M_EUR)
    err = np.array([[predicha - declarada * (1 + fb.p5_pct / 100)],
                    [declarada * (1 + fb.p95_pct / 100) - predicha]])
    barras = ax2.bar(["declarada", "predicha"], [declarada, predicha],
                     color=["#3a3a3a", DATO], width=.5, zorder=3)
    ax2.errorbar([1], [predicha], yerr=np.abs(err), fmt="none", ecolor=TINTA,
                 elinewidth=1.4, capsize=7, capthick=1.4, zorder=5)
    for barra, valor in zip(barras, (declarada, predicha)):
        ax2.text(barra.get_x() + barra.get_width() / 2, valor * 1.02,
                 esp(valor) + " M€", ha="center", va="bottom",
                 fontsize=9.5, color=TINTA, fontweight="bold", zorder=6)
    ax2.set_ylabel("Factura de derechos del régimen 2026 (M€)", fontsize=9, color=TINTA_2)
    ax2.set_title(f"Error {'+' if f26.error_pct >= 0 else '−'}{esp(abs(f26.error_pct), 2)} %; "
                  f"banda del 90 %: +{esp(fb.p5_pct, 2)} a +{esp(fb.p95_pct, 2)} %",
                  loc="left", fontsize=9.5, color=TINTA, pad=8)
    ax2.set_ylim(0, max(declarada, predicha) * 1.22)

    for ax in (ax1, ax2):
        ax.grid(axis="both" if ax is ax1 else "y", color="#e1e0d9", lw=.8, zorder=0)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(length=0, labelsize=8.5, colors=TINTA_2)
    fig.suptitle("Lo que el sistema habría dicho de 2025 con lo que se sabía en 2024",
                 x=0.008, ha="left", fontsize=11, color=TINTA, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{sal}/20_backtest_temporal.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"\nEscritos 20_backtest_*, 21_backtest_factura y 22_backtest_* en {sal}.")


if __name__ == "__main__":
    main()
