#!/usr/bin/env python3
"""Capa de pabellon: quien paga la factura climatica europea, y desde donde.

Cruza el pais de registro (`src/pabellon.py`) con la ficha por buque que ya produce la capa
economica (`reports/18_pasaporte_buques.csv`: factura ETS, multa FuelEU, banda CII y ahorro
alcanzable a actividad constante). **No toca el modelo ni recalcula ninguna cifra anterior**: solo
agrupa por pais de registro.

Escribe en `reports/`:

    19_pabellon_cobertura.csv   diagnostico: que parte del universo queda identificada
    19_pabellon_pais.csv        una fila por pais de registro, con factura y bandas CII
    19_pabellon_bloques.csv     agregado dentro / fuera del EEE / sin identificar
    19_pabellon_mapa.csv        el mismo contenido en formato ancho con ISO-3, para Tableau
    19_pabellon.png / .pdf      figura de los doce primeros paises por factura

    python scripts/analisis_pabellon.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import pabellon  # noqa: E402

DATOS = RAIZ / "data" / "processed"
INFORMES = RAIZ / "reports"
ANIO = 2025

COL_CO2 = "total_co2_emissions_m_tonnes"
COLOR_EEE = "#1f4e79"
COLOR_FUERA = "#c1651f"
COLOR_TEXTO = "#333333"


def _registro_por_buque() -> pd.DataFrame:
    """Un puerto de registro por buque y ano, desde el consolidado."""
    cols = ["reporting_year", "report_coverage", "ship_imo_number", "port_of_registry", COL_CO2]
    c = pd.read_parquet(DATOS / "mrv_consolidado.parquet", columns=cols)
    c = c[(c.reporting_year == ANIO) & (c.report_coverage == "Full")].copy()
    c["ship_imo_number"] = c.ship_imo_number.astype(str).str.strip()
    # Un buque puede traer varias filas; se queda la primera con puerto declarado.
    c = c.sort_values("port_of_registry", na_position="last")
    uno = c.groupby("ship_imo_number", as_index=False).agg(
        port_of_registry=("port_of_registry", "first"), co2_mrv=(COL_CO2, "sum"))
    return c, uno


def main() -> None:
    crudo, uno = _registro_por_buque()

    diag = pabellon.cobertura(crudo)
    pd.DataFrame([diag]).to_csv(INFORMES / "19_pabellon_cobertura.csv", index=False)
    print("Cobertura: %(pct_emisiones_con_pais).2f%% de las emisiones de %(filas)d filas "
          "quedan con pais; %(pct_emisiones_sin_declarar).2f%% sin declarar; "
          "%(puertos_distintos)d puertos distintos" % {**diag})

    p = pd.read_csv(INFORMES / "18_pasaporte_buques.csv")
    p["ship_imo_number"] = p.ship_imo_number.astype(str).str.strip()
    d = p.merge(uno, on="ship_imo_number", how="left")
    d = pd.concat([d, pabellon.clasificar(d["port_of_registry"])], axis=1)
    print(f"Buques cruzados: {len(d)} de la ficha; con pais {d.iso3.notna().sum()}")

    d["de_2025"] = d.banda_2025.isin(["D", "E"])
    d["de_2030"] = d.banda_2030.isin(["D", "E"])
    d["factura_eur"] = d.ets_2026_eur.fillna(0) + d.penalizacion_fueleu_eur.fillna(0)

    g = d.groupby("pais").agg(
        iso3=("iso3", "first"),
        eee=("eee", "first"),
        buques=("ship_imo_number", "nunique"),
        co2_t=("total_co2_emissions_m_tonnes", "sum"),
        ets_eur=("ets_2026_eur", "sum"),
        fueleu_eur=("penalizacion_fueleu_eur", "sum"),
        factura_eur=("factura_eur", "sum"),
        ahorro_actividad_constante_t=("ahorro_actividad_constante_t", "sum"),
        pct_de_2025=("de_2025", "mean"),
        pct_de_2030=("de_2030", "mean"),
        pct_registro_abierto=("registro_abierto", "mean"),
    ).sort_values("factura_eur", ascending=False)
    for c in ("pct_de_2025", "pct_de_2030", "pct_registro_abierto"):
        g[c] = (100 * g[c]).round(1)
    g["cuota_pct"] = (100 * g.factura_eur / g.factura_eur.sum()).round(3)
    g["cuota_acumulada_pct"] = g.cuota_pct.cumsum().round(3)
    g.reset_index().to_csv(INFORMES / "19_pabellon_pais.csv", index=False)

    identificado = d[d.iso3.notna()]
    bloque = identificado.assign(
        bloque=lambda x: x.eee.map({True: "Dentro del EEE", False: "Fuera del EEE"}))
    b = bloque.groupby("bloque").agg(
        buques=("ship_imo_number", "nunique"),
        co2_t=("total_co2_emissions_m_tonnes", "sum"),
        factura_eur=("factura_eur", "sum"),
        ahorro_actividad_constante_t=("ahorro_actividad_constante_t", "sum"),
        pct_de_2025=("de_2025", "mean"),
    )
    b["pct_de_2025"] = (100 * b.pct_de_2025).round(1)
    b["cuota_factura_pct"] = (100 * b.factura_eur / b.factura_eur.sum()).round(2)
    abierto = identificado[identificado.registro_abierto]
    b.loc["— de ellos, registro abierto (ITF)"] = [
        abierto.ship_imo_number.nunique(), abierto.total_co2_emissions_m_tonnes.sum(),
        abierto.factura_eur.sum(), abierto.ahorro_actividad_constante_t.sum(),
        round(100 * abierto.de_2025.mean(), 1),
        round(100 * abierto.factura_eur.sum() / identificado.factura_eur.sum(), 2)]
    b.reset_index().to_csv(INFORMES / "19_pabellon_bloques.csv", index=False)
    print(b.assign(Mt=(b.co2_t / 1e6).round(2), MEUR=(b.factura_eur / 1e6).round(0))
           [["buques", "Mt", "MEUR", "cuota_factura_pct", "pct_de_2025"]].to_string())

    mapa = g.reset_index()
    mapa = mapa[mapa.iso3.notna()].copy()
    mapa["co2_Mt"] = (mapa.co2_t / 1e6).round(3)
    mapa["factura_MEUR"] = (mapa.factura_eur / 1e6).round(2)
    mapa["ahorro_Mt"] = (mapa.ahorro_actividad_constante_t / 1e6).round(4)
    mapa["bloque"] = mapa.eee.map({True: "Dentro del EEE", False: "Fuera del EEE"})
    mapa[["pais", "iso3", "bloque", "buques", "co2_Mt", "factura_MEUR", "ahorro_Mt",
          "pct_de_2025", "pct_de_2030", "pct_registro_abierto", "cuota_pct"]].to_csv(
        INFORMES / "19_pabellon_mapa.csv", index=False)

    _figura(g)
    print("Escritos 19_pabellon_{cobertura,pais,bloques,mapa}.csv y 19_pabellon.{png,pdf}")


def _figura(g: pd.DataFrame) -> None:
    top = g[g.iso3.notna()].head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=300)
    colores = [COLOR_EEE if e else COLOR_FUERA for e in top.eee]
    y = range(len(top))
    ax.barh(list(y), top.factura_eur / 1e6, color=colores, height=0.72)
    ax.set_yticks(list(y))
    ax.set_yticklabels(top.index, fontsize=9, color=COLOR_TEXTO)
    for i, (v, de) in enumerate(zip(top.factura_eur / 1e6, top.pct_de_2025)):
        etiqueta = f"{v:,.0f}".replace(",", ".")
        ax.text(v + max(top.factura_eur / 1e6) * 0.012, i, f"{etiqueta} M€  ·  {de:.0f}% en D o E",
                va="center", fontsize=8.5, color=COLOR_TEXTO)
    ax.set_xlim(0, max(top.factura_eur / 1e6) * 1.32)
    ax.set_xlabel("Factura de cumplimiento (EU ETS + FuelEU) sobre las emisiones de 2025,\n"
                  "en millones de euros. Derecho de emisión a 82,42 €/t (cierre del 27/08/2026).\n"
                  "El porcentaje es la parte de esa flota que el CII califica D o E.",
                  fontsize=8, color=COLOR_TEXTO)
    ax.set_title("Más de la mitad de la factura climática europea la pagan buques\n"
                 "registrados fuera de Europa",
                 fontsize=12.5, fontweight="bold", color=COLOR_TEXTO, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5, colors=COLOR_TEXTO)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    manijas = [plt.Rectangle((0, 0), 1, 1, color=COLOR_EEE),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_FUERA)]
    ax.legend(manijas, ["Registro dentro del EEE", "Registro fuera del EEE"],
              loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(INFORMES / "19_pabellon.png", bbox_inches="tight")
    fig.savefig(INFORMES / "19_pabellon.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
