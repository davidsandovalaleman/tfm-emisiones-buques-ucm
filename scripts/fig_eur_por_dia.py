#!/usr/bin/env python3
"""
Figura de la capa economica - Cuanto vale un dia de mar evitado, por tipo de buque.

Es el precio de equilibrio de la palanca operativa: si un buque gana menos que esta cifra por dia
de navegacion, recortar millas le sale a cuenta sin que nadie le pague por el CO2. Se publica asi,
y no como un ahorro neto, para no tener que suponer tarifas de fletamento.

Lee reports/15_palancas_eur_por_dia.csv y escribe reports/15_eur_por_dia.png y .pdf.

    python scripts/fig_eur_por_dia.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
COLOR_ALTO, COLOR_BAJO, COLOR_TEXTO = "#1f4e79", "#a8c4de", "#333333"
DESTACADOS = {"LNG carrier", "Passenger ship", "Ro-pax ship", "Container ship"}
TRADUCCION = {
    "LNG carrier": "Metanero (LNG)", "Passenger ship": "Buque de pasaje", "Ro-pax ship": "Ro-pax",
    "Container ship": "Portacontenedores", "Container/ro-ro cargo ship": "Contenedores / ro-ro",
    "Refrigerated cargo carrier": "Carga refrigerada", "Oil tanker": "Petrolero",
    "Chemical tanker": "Quimiquero", "Bulk carrier": "Granelero", "Gas carrier": "Gasero",
    "General cargo ship": "Carga general", "Vehicle carrier": "Portavehiculos",
    "Ro-ro ship": "Ro-ro", "Combination carrier": "Buque combinado", "Other ship types": "Otros",
}


def main() -> None:
    d = pd.read_csv(RAIZ / "reports" / "15_palancas_eur_por_dia.csv")
    d = d[d.palanca == "millas_-10pct"].copy()
    d = d[d.ahorro_t > 10_000].sort_values("eur_por_dia_mar")
    media = d.valor_eur.sum() / (d.horas_evitadas.sum() / 24)

    valores = d.eur_por_dia_mar.to_numpy()
    etiquetas = [TRADUCCION.get(t, t) for t in d.tipo]
    mapa = matplotlib.colors.LinearSegmentedColormap.from_list("azul", [COLOR_BAJO, COLOR_ALTO])
    rango = (valores.max() - valores.min()) or 1.0
    colores = [mapa((v - valores.min()) / rango) for v in valores]

    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(valores) + 1.4), dpi=300)
    ax.barh(etiquetas, valores, color=colores, height=0.68, zorder=3)
    for i, (v, t) in enumerate(zip(valores, d.tipo)):
        ax.text(v + valores.max() * 0.015, i, f"{v:,.0f}".replace(",", "."), va="center",
                fontsize=8.5, color=COLOR_TEXTO,
                fontweight="bold" if t in DESTACADOS else "normal")
    ax.axvline(media, color="#c0504d", lw=1.1, ls="--", zorder=4)
    ax.text(media, len(valores) - 0.3, f"  media de la flota {media:,.0f}".replace(",", "."),
            color="#c0504d", fontsize=8, va="center")

    ax.set_xlabel("Valor de un d\u00eda de mar evitado (\u20ac/d\u00eda): combustible no quemado + derechos del ETS",
                  fontsize=8.5, color=COLOR_TEXTO)
    ax.set_xlim(0, valores.max() * 1.16)
    ax.grid(axis="x", color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=8.5, colors=COLOR_TEXTO)
    fig.suptitle("Si un buque gana menos que esto por d\u00eda de mar, navegar menos ya le sale a cuenta",
                 x=0.01, ha="left", fontsize=10.5, color=COLOR_TEXTO, fontweight="bold")
    ax.set_title("En negrita, los cuatro tipos donde concentrar el recorte duplica el ahorro",
                 loc="left", fontsize=8, color="#666666", pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ("png", "pdf"):
        fig.savefig(RAIZ / "reports" / f"15_eur_por_dia.{ext}", bbox_inches="tight")
    print("reports/15_eur_por_dia.png y .pdf")


if __name__ == "__main__":
    main()
