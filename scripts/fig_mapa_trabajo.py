#!/usr/bin/env python3
"""
Figura 1 de la memoria (§1) — Mapa del trabajo.

Siete bloques en una fila, cada uno con su cifra de cabecera. Es la unica figura del
trabajo que no sale de un analisis sino de un diseno, y la que se reutiliza en el video.

Escribe reports/01_mapa_trabajo.png y .pdf (incrusta el PDF en Word: es vectorial).

Uso:
    python scripts/fig_mapa_trabajo.py
    python scripts/fig_mapa_trabajo.py --ancho 6.9      # pulgadas, ajusta al ancho de la pagina
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "reports" / "01_mapa_trabajo"

AZUL_OSCURO = "#1f4e79"
AZUL_CLARO = "#a8c4de"
GRIS_TEXTO = "#4a4a4a"
GRIS_SUAVE = "#7a7a7a"
BORDE = "#c8d6e4"

# (etiqueta, cifra, pie). El orden es el del documento.
BLOQUES = [
    ("Datos",       "106.521",  "registros\nbuque-año"),
    ("Exploración", "3.140",    "atípicos que\neran señal"),
    ("Variables",   "94,0 %",   "cobertura de\nvelocidad"),
    ("Modelos",     "0,8337",   "R² sobre\nel test"),
    ("Explicación", "3,0–6,5",  "t CO₂ por hora\nnavegada"),
    ("Simulación",  "13,7 Mt",  "ahorro con\n−10 % de millas"),
    ("Servicio",    "7",        "endpoints\nverificados"),
]


def construir(ancho: float):
    n = len(BLOQUES)
    alto = ancho * 0.215
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=300)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    hueco = 0.13          # separacion entre cajas, en unidades de bloque
    caja_w = 1 - hueco
    y0, caja_h = 0.06, 0.88

    fondo = matplotlib.colors.LinearSegmentedColormap.from_list(
        "azul_fondo", ["#eef4fa", "#dce9f5"]
    )
    tinta = matplotlib.colors.LinearSegmentedColormap.from_list(
        "azul_tinta", [AZUL_CLARO, AZUL_OSCURO]
    )

    for i, (etiqueta, cifra, pie) in enumerate(BLOQUES):
        x0 = i + hueco / 2
        cx = x0 + caja_w / 2

        # Intensidad creciente: el recorrido avanza de izquierda a derecha.
        t = i / (n - 1)
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0), caja_w, caja_h,
                boxstyle="round,pad=0,rounding_size=0.05",
                linewidth=0.8, edgecolor=BORDE, facecolor=fondo(t), zorder=2,
            )
        )

        color_cifra = tinta(0.35 + 0.65 * t)

        ax.text(cx, y0 + caja_h - 0.10, etiqueta, ha="center", va="top",
                fontsize=7.6, fontweight="bold", color=GRIS_TEXTO, zorder=3)
        ax.text(cx, y0 + caja_h * 0.50, cifra, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=color_cifra, zorder=3)
        ax.text(cx, y0 + 0.17, pie, ha="center", va="center",
                fontsize=5.9, color=GRIS_SUAVE, linespacing=1.30, zorder=3)

        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + caja_w + 0.012, y0 + caja_h / 2),
                    (i + 1 + hueco / 2 - 0.012, y0 + caja_h / 2),
                    arrowstyle="-|>", mutation_scale=6.5,
                    linewidth=0.9, color="#9db6cd", zorder=1,
                )
            )

    fig.tight_layout(pad=0.2)
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ancho", type=float, default=6.9,
                   help="ancho en pulgadas (por defecto 6.9 ~ 17,5 cm)")
    p.add_argument("--salida", type=Path, default=SALIDA, help="ruta sin extension")
    args = p.parse_args()

    fig = construir(args.ancho)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(args.salida.with_suffix(ext), bbox_inches="tight")
        print(f"Escrito: {args.salida.with_suffix(ext)}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
