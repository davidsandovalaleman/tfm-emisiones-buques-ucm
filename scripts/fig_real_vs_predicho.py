"""
FIGURA 4 de la memoria — predicho frente a observado sobre el conjunto de prueba.

La figura que producía el notebook 04 (`reports/04_real_vs_predicho_xgb.png`) dibuja las
emisiones reales y las predichas contra el índice de una submuestra de 800 filas. Sirve para
mirar el notebook por encima, pero no es lo que la memoria promete —«valores predichos frente a
observados, con la diagonal de referencia»— y sin diagonal no se puede leer si el modelo se queda
corto o se pasa.

Esta genera la figura de la memoria, con tres decisiones de diseño:

1. **Hexágonos de densidad y no puntos.** Son 20.034 filas: un diagrama de dispersión se
   convierte en una mancha y esconde justamente donde está la masa. El color codifica cuántos
   buques-año caen en cada celda, que es una magnitud, así que va en una rampa de un solo tono.
2. **Escala logarítmica en los dos ejes.** El objetivo va de decenas de toneladas a 315.478, y en
   escala lineal el 90% de los buques se amontona contra el origen. El precio es que hay que
   excluir las filas cuyo CO2 observado o predicho es cero (buques inactivos, y las predicciones
   que el recorte fisico deja en cero), que el logaritmo no admite; se dice en el pie.
3. **La diagonal, y los dos casos que la memoria cita.** Sin la identidad no hay figura; y los dos
   buques del apartado 6 se anotan para que el lector los sitúe.

Uso:  python scripts/fig_real_vs_predicho.py
Lee:  reports/05_test_predicciones.csv
Deja: reports/04_predicho_vs_observado.png  (+ .pdf)
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = os.path.join(RAIZ, "reports", "05_test_predicciones.csv")
SALIDA = os.path.join(RAIZ, "reports", "04_predicho_vs_observado")

# Misma familia de azules que el resto de figuras de la memoria; rampa de un solo tono porque
# lo que codifica es una magnitud (cuántos buques-año hay en la celda), no una identidad.
RAMPA = LinearSegmentedColormap.from_list(
    "azules_tfm", ["#E8EEF6", "#9CB8D8", "#4E7CB0", "#1F4E79"]
)
TINTA = "#333333"
TINTA_SUAVE = "#6B6B6B"
GRIS_LINEA = "#8C8C8C"
ACENTO = "#B3472A"

# Los dos casos que cita el apartado 6, con el criterio con el que se eligio cada uno. NO son "los
# dos peores errores del test": uno es el buque-ano que mas emitio y el otro el que peor se predice.
# Cada IMO aparece hasta ocho veces en el conjunto, asi que el criterio se aplica dentro del buque y
# no sobre la primera fila que salga. La posicion de cada etiqueta se fija a mano: con dos
# anotaciones tan proximas en la esquina superior derecha, el desplazamiento automatico las solapa.
CASOS = {
    9351488: ("CRUISE BARCELONA", "mayor emisor del test", "mayor_emisor", (1.6e4, 2.4e5), "left"),
    9265500: ("DUKHAN", "peor error absoluto", "peor_error", (1.1e4, 4.2e3), "left"),
}


def main() -> None:
    datos = pd.read_csv(ENTRADA)
    total = len(datos)
    activos = datos[(datos["co2_real"] > 0) & (datos["pred_xgboost"] > 0)].copy()
    excluidas = total - len(activos)

    residuo = datos["pred_xgboost"] - datos["co2_real"]
    ss_res = float((residuo**2).sum())
    ss_tot = float(((datos["co2_real"] - datos["co2_real"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    mae = float(residuo.abs().mean())

    mpl.rcParams.update({"font.size": 9, "axes.edgecolor": "#CCCCCC", "axes.linewidth": 0.8})
    figura, eje = plt.subplots(figsize=(7.2, 5.4))

    malla = eje.hexbin(
        activos["co2_real"], activos["pred_xgboost"],
        xscale="log", yscale="log", bins="log",
        gridsize=46, cmap=RAMPA, mincnt=1, linewidths=0.0,
    )

    limites = (30, 4e5)
    eje.plot(limites, limites, ls="--", lw=1.2, color=GRIS_LINEA, zorder=3)
    eje.annotate(
        "predicción perfecta", xy=(3.2e2, 3.2e2), xytext=(1.1e3, 9.5e1),
        color=GRIS_LINEA, fontsize=8.5, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=GRIS_LINEA, lw=0.8, shrinkA=2, shrinkB=2),
    )

    for imo, (nombre, papel, criterio, posicion, alineacion) in CASOS.items():
        filas = activos[activos["ship_imo_number"] == imo]
        if filas.empty:
            continue
        if criterio == "mayor_emisor":
            fila = filas.loc[filas["co2_real"].idxmax()]
        else:
            fila = filas.loc[(filas["pred_xgboost"] - filas["co2_real"]).idxmin()]
        x, y = float(fila["co2_real"]), float(fila["pred_xgboost"])
        print(f"  {nombre}: observado {x:,.0f} t | predicho {y:,.0f} t | error {y - x:,.0f} t"
              .replace(",", "."))
        eje.plot(x, y, "o", ms=8, mfc="none", mec=ACENTO, mew=1.7, zorder=5)
        eje.annotate(
            f"{nombre}\n({papel})", xy=(x, y), xytext=posicion,
            color=ACENTO, fontsize=8.2, ha=alineacion, va="center", linespacing=1.3,
            arrowprops=dict(arrowstyle="-", color=ACENTO, lw=0.8, alpha=0.85,
                            shrinkA=2, shrinkB=6), zorder=5,
        )

    eje.set_xlim(*limites)
    eje.set_ylim(*limites)
    eje.set_aspect("equal")
    eje.set_xlabel("CO₂ observado (t, escala logarítmica)", color=TINTA)
    eje.set_ylabel("CO₂ predicho (t, escala logarítmica)", color=TINTA)
    eje.set_title(
        f"Los buques que el modelo no vio entrenando: R² {r2:.4f}".replace(".", ","),
        color=TINTA, pad=10,
    )
    eje.grid(True, which="major", color="#E6E6E6", lw=0.7)
    eje.set_axisbelow(True)
    eje.tick_params(colors=TINTA_SUAVE)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)

    eje.text(
        0.03, 0.97,
        f"MAE {mae:,.0f} t".replace(",", ".") + f"\n{len(activos):,} buques-año".replace(",", "."),
        transform=eje.transAxes, va="top", ha="left", fontsize=8.5, color=TINTA_SUAVE,
        linespacing=1.5,
    )

    barra = figura.colorbar(malla, ax=eje, shrink=0.82, pad=0.02)
    barra.set_label("buques-año por celda (escala logarítmica)", color=TINTA_SUAVE, fontsize=8.5)
    barra.ax.tick_params(colors=TINTA_SUAVE, labelsize=8)
    barra.outline.set_visible(False)

    figura.tight_layout()
    figura.savefig(SALIDA + ".png", dpi=150, bbox_inches="tight")
    figura.savefig(SALIDA + ".pdf", bbox_inches="tight")
    print(f"R2 {r2:.4f} | MAE {mae:,.1f} t | {len(activos)} filas dibujadas, "
          f"{excluidas} excluidas por tener un cero en alguno de los dos ejes")
    print("escrito:", SALIDA + ".png")


if __name__ == "__main__":
    main()
