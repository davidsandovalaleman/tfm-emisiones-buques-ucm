#!/usr/bin/env python3
"""Mapa de cobertura del temario del máster: los 26 bloques y dónde está cada uno.

Escribe `reports/23_mapa_temario.png` y `.pdf`.

Resume en una sola imagen qué técnicas se aplican en el trabajo y en qué pieza del repositorio.

**Incluye los descartes.** Tres bloques no se aplican, y la figura lo dice con la cifra que lo
justifica en vez de dejar el hueco en blanco. Descartar una técnica con una medición
**es** aplicar la metodología; dejar el hueco sin explicar parece un olvido.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "reports" / "23_mapa_temario"

TINTA, TINTA_2, MUTE = "#0b0b0b", "#52514e", "#898781"
APLICADO_BORDE, APLICADO_FONDO = "#2a78d6", "#e8f1fc"
DESCARTE_BORDE, DESCARTE_FONDO = "#898781", "#f2f1ee"

# (nº, nombre corto, dónde está / por qué no, aplicado)
TEMAS = [
    (1,  "Bases de datos SQL",        "notebook 09 · SQLite, 2 tablas", True),
    (2,  "Bases de datos NoSQL",      "notebook 10 · pasaporte documental", True),
    (3,  "Programación Python",       "12 módulos en src/, 13 scripts", True),
    (4,  "BI con Tableau",            "tablero de riesgo por pabellón", True),
    (5,  "GNU, Linux, GIT",           "repositorio, scripts, entorno fijado", True),
    (6,  "Estadística",               "notebook 02 · banda conformal", True),
    (7,  "Minería y modelización 01", "notebook 03 · ingeniería de variables", True),
    (8,  "Minería y modelización 02", "notebook 03 · atípicos con señal", True),
    (9,  "Minería y modelización 03", "notebook 04 · validación agrupada", True),
    (10, "Minería y modelización 04", "notebook 04 · imp. por permutación", True),
    (11, "Machine learning 01",       "notebook 04 · lineales y regularizados", True),
    (12, "Machine learning 02",       "notebook 04 · árboles y bosques", True),
    (13, "Machine learning 03",       "notebook 04 · XGBoost, R² 0,948", True),
    (14, "Machine learning 04",       "notebook 04 · stacking y mezclas", True),
    (15, "Machine learning",          "cinco niveles de información", True),
    (16, "Deep learning 01",          "notebook 05 · red densa", True),
    (17, "Deep learning 02",          "notebook 05 · Keras Tuner", True),
    (18, "Deep learning 03",          "notebook 05 · pierde, y se mide", True),
    (19, "Series temporales (RNN)",   "descartado: 8 pasos por buque;\nel 21,6% tiene un solo año", False),
    (20, "NLP",                       "descartado: el texto libre del MRV\nestá relleno en el 0,1% de las filas", False),
    (21, "Modelos generativos",       "descartado: la guía del TFM\ndesaconseja el dato sintético", False),
    (22, "Visualización avanzada",    "11 figuras + panel web", True),
    (23, "Productivizar un modelo",   "API FastAPI, 7 endpoints", True),
    (24, "Tecnologías de big data",   "notebook 11 · ingesta distribuida", True),
    (25, "Spark",                     "notebook 11 · y pierde contra pandas", True),
    (26, "Data Science en la empresa","la factura, con dueño y con nombre", True),
]

COLUMNAS = 4


def construir(ancho: float):
    filas = (len(TEMAS) + COLUMNAS - 1) // COLUMNAS
    alto = ancho * 0.40
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=300)
    ax.set_xlim(0, COLUMNAS)
    ax.set_ylim(0, filas)
    ax.axis("off")

    hueco_x, hueco_y = 0.06, 0.14
    for i, (numero, nombre, donde, aplicado) in enumerate(TEMAS):
        col, fila = i % COLUMNAS, filas - 1 - i // COLUMNAS
        x, y = col + hueco_x / 2, fila + hueco_y / 2
        w, h = 1 - hueco_x, 1 - hueco_y
        borde = APLICADO_BORDE if aplicado else DESCARTE_BORDE
        fondo = APLICADO_FONDO if aplicado else DESCARTE_FONDO
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                    linewidth=1.0, edgecolor=borde, facecolor=fondo, zorder=2))
        # Número del bloque
        ax.text(x + 0.055, y + h - 0.10, f"{numero}", fontsize=11, fontweight="bold",
                color=borde, va="top", ha="left", zorder=3)
        ax.text(x + 0.155, y + h - 0.115, nombre, fontsize=7.6, fontweight="600",
                color=TINTA if aplicado else TINTA_2, va="top", ha="left", zorder=3)
        ax.text(x + 0.155, y + h - 0.30, donde, fontsize=6.0,
                color=TINTA_2 if aplicado else MUTE, va="top", ha="left", zorder=3,
                style="normal" if aplicado else "italic")

    aplicados = sum(1 for t in TEMAS if t[3])
    fig.suptitle(f"Los 26 bloques del máster: {aplicados} aplicados y "
                 f"{len(TEMAS) - aplicados} descartados con su medida",
                 x=0.012, y=0.995, ha="left", fontsize=11.5, fontweight="bold", color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ancho", type=float, default=12.0)
    fig = construir(ap.parse_args().ancho)
    for ext in ("png", "pdf"):
        fig.savefig(f"{SALIDA}.{ext}", bbox_inches="tight")
    print(f"{SALIDA}.png y .pdf")


if __name__ == "__main__":
    main()
