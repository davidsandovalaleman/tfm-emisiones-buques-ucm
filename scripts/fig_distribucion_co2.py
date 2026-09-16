#!/usr/bin/env python3
"""
Figura 2 de la memoria (§3) — Distribucion de las emisiones y el umbral de atipicos.

Muestra en una sola imagen por que el criterio estadistico de atipicos no se aplica al CO2: donde
caeria su umbral, que parte de la cola dejaria fuera, y el contraste entre "pocas filas" y
"muchas emisiones".

Panel izquierdo: histograma del CO2 en escala logaritmica con el umbral marcado y la
region excluida sombreada. Panel derecho: dos barras, % de filas frente a % de emisiones.

Escribe reports/02_distribucion_co2_umbral.png y .pdf. No sobreescribe el
02_distribucion_co2.png original del notebook.

Uso:
    python scripts/fig_distribucion_co2.py
    python scripts/fig_distribucion_co2.py --umbral 49664
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PARQUET = RAIZ / "data" / "processed" / "mrv_features_principal_ml.parquet"
SALIDA = RAIZ / "reports" / "02_distribucion_co2_umbral"
COL = "total_co2_emissions_m_tonnes"

AZUL = "#1f4e79"
AZUL_MEDIO = "#5b89b5"
AZUL_CLARO = "#a8c4de"
ROJO = "#b3452c"
GRIS = "#4a4a4a"
GRIS_SUAVE = "#7a7a7a"


def _es(x: float, dec: int = 0) -> str:
    """Formatea un numero al estilo espanol: miles con punto, decimales con coma."""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def construir(y: pd.Series, umbral: float):
    positivos = y[y > 0]
    encima = y > umbral
    pct_filas = 100 * encima.mean()
    pct_emis = 100 * y[encima].sum() / y.sum()

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(6.9, 2.7), dpi=300, gridspec_kw={"width_ratios": [2.6, 1]}
    )

    # ---- Panel izquierdo: distribucion en escala logaritmica -------------------------
    bins = np.logspace(np.log10(positivos.min()), np.log10(positivos.max()), 60)
    ax.hist(positivos, bins=bins, color=AZUL_CLARO, edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_xscale("log")

    ymax = ax.get_ylim()[1]
    ax.axvspan(umbral, positivos.max() * 1.15, color=ROJO, alpha=0.07, zorder=1)
    ax.axvline(umbral, color=ROJO, linewidth=1.2, linestyle="--", zorder=4)

    ax.annotate(
        f"umbral del criterio\nestadístico: {_es(umbral)} t",
        xy=(umbral, ymax * 0.74), xytext=(umbral * 0.60, ymax * 0.74),
        ha="right", va="top", fontsize=6.4, color=ROJO, linespacing=1.35, zorder=5,
    )
    ax.text(
        umbral * 1.45, ymax * 0.42,
        f"{_es(encima.sum())} registros\npor encima",
        ha="left", va="center", fontsize=6.4, color=GRIS, linespacing=1.35, zorder=5,
    )

    ax.set_xlabel("Emisiones anuales de CO$_2$ por buque (t, escala logarítmica)",
                  fontsize=7.2, color=GRIS)
    ax.set_ylabel("Número de registros", fontsize=7.2, color=GRIS)
    ax.tick_params(labelsize=6.4, colors=GRIS_SUAVE)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#bbbbbb")

    # ---- Panel derecho: pocas filas, muchas emisiones --------------------------------
    etiquetas = ["% de los\nregistros", "% de las\nemisiones"]
    valores = [pct_filas, pct_emis]
    colores = [AZUL_CLARO, AZUL]
    barras = ax2.bar(etiquetas, valores, color=colores, width=0.55, zorder=2)
    for b, v in zip(barras, valores):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{_es(v, 1)} %",
                 ha="center", va="bottom", fontsize=7.6, fontweight="bold",
                 color=b.get_facecolor(), zorder=3)

    ax2.set_ylim(0, max(valores) * 1.32)
    ax2.set_title("Lo que quedaba excluido", fontsize=7.2, color=GRIS, pad=6)
    ax2.tick_params(axis="x", length=0, labelsize=6.4, colors=GRIS)
    ax2.tick_params(axis="y", labelsize=6.4, colors=GRIS_SUAVE)
    ax2.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax2.set_axisbelow(True)
    for lado in ("top", "right", "bottom"):
        ax2.spines[lado].set_visible(False)
    ax2.spines["left"].set_color("#bbbbbb")

    fig.tight_layout(pad=0.4, w_pad=2.0)
    return fig, pct_filas, pct_emis, int(encima.sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", type=Path, default=PARQUET)
    p.add_argument("--umbral", type=float, default=49664.0)
    p.add_argument("--salida", type=Path, default=SALIDA, help="ruta sin extension")
    args = p.parse_args()

    if not args.parquet.exists():
        print(f"ERROR: no encuentro {args.parquet}")
        return 1

    y = pd.read_parquet(args.parquet, columns=[COL])[COL]
    fig, pf, pe, n = construir(y, args.umbral)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(args.salida.with_suffix(ext), bbox_inches="tight")
        print(f"Escrito: {args.salida.with_suffix(ext)}")
    plt.close(fig)

    print(f"Comprobación: {n} registros por encima de {args.umbral:.0f} t "
          f"= {pf:.2f} % de las filas y {pe:.1f} % de las emisiones.")
    print("El texto de §3 debe decir estas mismas cifras.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
