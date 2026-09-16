"""
FIGURA 3 de la memoria — evolución 2018-2025 de la flota declarante y de sus emisiones.

La figura que producía el notebook 02 (`reports/02_evolucion_temporal.png`) sirve para el cuaderno
pero no para la memoria: lleva los nombres de columna en crudo (`reporting_year`), dibuja la
*mediana* de emisiones por buque en vez del total de la flota, y no marca el cambio de ámbito de
2025, que es justo lo que el apartado 2 necesita explicar.

Esta la genera con lo que el texto describe y una decisión de forma:

**Dos paneles y no un eje doble.** El número de buques y las megatoneladas son magnitudes de
escalas distintas; superponerlas en un eje secundario es la forma más rápida de sugerir una
correlación que nadie ha medido. Van en dos paneles que comparten el eje temporal.

El salto de 2025 se marca en el panel de la flota, porque es el punto del trabajo: **el 70% del
crecimiento de ese año es regulatorio y no de flota** — desde el 1/1/2025 el Reglamento (UE)
2023/957 mete en el sistema a los buques de carga general de 400-5.000 GT y a los *offshore* de
400 GT en adelante.

Uso:  python scripts/fig_evolucion_anual.py
Deja: reports/03_evolucion_anual.png  (+ .pdf)

Las series están escritas a mano en el propio fichero, y a propósito: salen de agregar
`data/processed/mrv_features_principal_raw.parquet` por ejercicio, que no se versiona. La
comprobación de que siguen siendo ciertas es una línea de pandas y está en el docstring de
`series()`.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "reports", "03_evolucion_anual")

AZUL = "#1F4E79"
AZUL_SUAVE = "#9CB8D8"
TINTA = "#333333"
TINTA_SUAVE = "#6B6B6B"
ACENTO = "#B3472A"


def series():
    """
    Registros y megatoneladas por ejercicio.

    Reproducible con:

        d = pd.read_parquet('data/processed/mrv_features_principal_raw.parquet')
        d.groupby('reporting_year').agg(buques=('total_co2_emissions_m_tonnes', 'size'),
                                        mt=('total_co2_emissions_m_tonnes', 'sum'))
    """
    anios = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    buques = [12246, 12383, 12105, 12474, 13444, 12797, 14133, 16939]
    megatoneladas = [145.0, 146.7, 129.6, 126.7, 137.2, 128.4, 146.9, 150.1]
    return anios, buques, megatoneladas


def main() -> None:
    anios, buques, megatoneladas = series()

    mpl.rcParams.update({"font.size": 9, "axes.edgecolor": "#CCCCCC", "axes.linewidth": 0.8})
    figura, (izq, der) = plt.subplots(1, 2, figsize=(9.2, 3.5), sharex=True)

    colores = [AZUL_SUAVE] * (len(anios) - 1) + [AZUL]
    izq.bar(anios, buques, color=colores, width=0.66)
    izq.set_title("Buques-año en el registro", color=TINTA, fontsize=9.5, pad=8)
    izq.set_ylim(0, max(buques) * 1.30)
    izq.annotate(
        "desde 2025 entran la carga general\nde 400-5.000 GT y los offshore:\nel 70% del salto es regulatorio",
        xy=(2025, buques[-1]), xytext=(2021.4, max(buques) * 1.16),
        color=ACENTO, fontsize=8, ha="center", va="center", linespacing=1.35,
        arrowprops=dict(arrowstyle="-", color=ACENTO, lw=0.8, shrinkA=2, shrinkB=4),
    )
    for anio, valor in ((anios[0], buques[0]), (anios[-1], buques[-1])):
        izq.text(anio, valor * 1.02, f"{valor:,}".replace(",", "."), ha="center", va="bottom",
                 fontsize=8.2, color=TINTA_SUAVE)

    der.plot(anios, megatoneladas, "-o", color=AZUL, lw=2, ms=5)
    der.set_title("Emisiones totales declaradas (Mt de CO₂)", color=TINTA, fontsize=9.5, pad=8)
    der.set_ylim(0, max(megatoneladas) * 1.30)
    for anio, valor in ((anios[0], megatoneladas[0]), (2021, megatoneladas[3]), (anios[-1], megatoneladas[-1])):
        der.text(anio, valor + max(megatoneladas) * 0.05, f"{valor:.1f}".replace(".", ","),
                 ha="center", fontsize=8.2, color=TINTA_SUAVE)

    for eje in (izq, der):
        eje.set_xticks(anios)
        eje.set_xticklabels([str(a) for a in anios], fontsize=8.2)
        eje.grid(axis="y", color="#E6E6E6", lw=0.7)
        eje.set_axisbelow(True)
        eje.tick_params(colors=TINTA_SUAVE)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)

    figura.tight_layout()
    figura.savefig(SALIDA + ".png", dpi=150, bbox_inches="tight")
    figura.savefig(SALIDA + ".pdf", bbox_inches="tight")
    print("escrito:", SALIDA + ".png")


if __name__ == "__main__":
    main()
