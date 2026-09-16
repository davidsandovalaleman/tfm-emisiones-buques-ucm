#!/usr/bin/env python3
"""
Figura de la capa economica - Los tres flujos de dinero de una tonelada evitada.

Compara el escenario de cabecera del simulador (recortar un 10% de las millas, que cuesta
actividad) con el ahorro alcanzable **a actividad constante** en sus dos lecturas, y descompone
el valor en combustible no quemado, derechos del ETS y multa de FuelEU.

Lee reports/14_sensibilidad_precios.csv y reports/12_ets_escenarios.csv (si existe) y escribe
reports/14_flujos_dinero.png y .pdf.

    python scripts/fig_flujos_dinero.py
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
import economia  # noqa: E402

COLORES = {"Combustible no quemado": "#1f4e79", "Derechos EU ETS": "#5b8db8", "Multa FuelEU evitada": "#a8c4de"}
COLOR_TEXTO = "#333333"


def main() -> None:
    sens = pd.read_csv(RAIZ / "reports" / "14_sensibilidad_precios.csv")
    sens = sens[sens.precio_bunker == economia.PRECIO_BUNKER_EUR_POR_T]
    fila = {r.frontera: r for r in sens.itertuples()}

    # El escenario del simulador, recalculado con los mismos precios para que sea comparable.
    pal = pd.read_csv(RAIZ / "reports" / "15_palancas_eur_por_dia.csv")
    millas = pal[pal.palanca == "millas_-10pct"]
    ahorro_millas = millas.ahorro_t.sum()
    fila_millas = {
        "combustible_eur": millas.combustible_t.sum() * economia.PRECIO_BUNKER_EUR_POR_T,
        "ets_eur": millas.ets_eur.sum(),
        "fueleu_eur": fila["mediana"].fueleu_eur * ahorro_millas / fila["mediana"].ahorro_t,
        "ahorro_t": ahorro_millas,
    }

    barras = [
        (f"-10% de millas\n{ahorro_millas/1e6:.2f} Mt · cuesta actividad", fila_millas),
        (f"Actividad constante, mejor cuartil\n{fila['mejor_cuartil'].ahorro_t/1e6:.2f} Mt",
         fila["mejor_cuartil"]._asdict()),
        (f"Actividad constante, mediana\n{fila['mediana'].ahorro_t/1e6:.2f} Mt",
         fila["mediana"]._asdict()),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=300)
    for i, (etiqueta, d) in enumerate(barras):
        izquierda = 0.0
        for clave, nombre in (("combustible_eur", "Combustible no quemado"),
                              ("ets_eur", "Derechos EU ETS"),
                              ("fueleu_eur", "Multa FuelEU evitada")):
            valor = float(d[clave]) / 1e6
            ax.barh(i, valor, left=izquierda, color=COLORES[nombre], height=0.62, zorder=3,
                    label=nombre if i == 0 else None)
            izquierda += valor
        ax.text(izquierda + 40, i, f"{izquierda:,.0f} M\u20ac".replace(",", "."),
                va="center", ha="left", fontsize=9, color=COLOR_TEXTO, fontweight="bold")

    ax.set_yticks(range(len(barras)))
    ax.set_yticklabels([e for e, _ in barras], fontsize=8.5, color=COLOR_TEXTO)
    ax.set_xlabel(
        "Valor anual del ahorro, flota del EEE sobre emisiones de 2025 (millones de euros)\n"
        f"Derecho de emisi\u00f3n {sens.precio_eua.iloc[0]:.2f} \u20ac/t (cierre del 27/08/2026); "
        f"combustible {economia.PRECIO_BUNKER_EUR_POR_T:.0f} \u20ac/t (par\u00e1metro)",
        fontsize=8.5, color=COLOR_TEXTO)
    ax.set_xlim(0, max(sum(float(d[k]) for k in ("combustible_eur", "ets_eur", "fueleu_eur"))
                       for _, d in barras) / 1e6 * 1.18)
    ax.grid(axis="x", color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=8, colors=COLOR_TEXTO)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Una tonelada de CO\u2082 evitada mueve tres flujos de dinero: el mayor es el combustible",
                 x=0.01, y=1.0, ha="left", fontsize=10.5, color=COLOR_TEXTO, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(RAIZ / "reports" / f"14_flujos_dinero.{ext}", bbox_inches="tight")
    print("reports/14_flujos_dinero.png y .pdf")


if __name__ == "__main__":
    main()
