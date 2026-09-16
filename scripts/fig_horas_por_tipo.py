#!/usr/bin/env python3
"""
Figura 7 de la memoria (§6) — Toneladas de CO2 por hora navegada, por tipo de buque.

Lee reports/07_pendiente_horas_por_tipo.csv y escribe
reports/07_pendiente_horas_por_tipo.png (y .pdf, que es lo que conviene incrustar en Word
si se quiere que el texto de la figura no pixele al imprimir).

Uso:
    python scripts/fig_horas_por_tipo.py
    python scripts/fig_horas_por_tipo.py --csv otra/ruta.csv --salida reports/fig7
    python scripts/fig_horas_por_tipo.py --idioma en      # deja los nombres originales

No depende de seaborn ni de ningun estilo instalado: solo pandas y matplotlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: se ejecuta igual en un servidor o en un notebook
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parents[1]
CSV_POR_DEFECTO = RAIZ / "reports" / "07_pendiente_horas_por_tipo.csv"
SALIDA_POR_DEFECTO = RAIZ / "reports" / "07_pendiente_horas_por_tipo"

# Nombres en castellano para la memoria. Si un tipo no esta aqui, se deja tal cual.
TRADUCCION = {
    "LNG carrier": "Metanero (LNG)",
    "Passenger ship": "Buque de pasaje",
    "Ro-pax ship": "Ro-pax",
    "Container ship": "Portacontenedores",
    "Container/ro-ro cargo ship": "Contenedores / ro-ro",
    "Container/ro-ro": "Contenedores / ro-ro",
    "Refrigerated cargo carrier": "Carga refrigerada",
    "Refrigerated cargo": "Carga refrigerada",
    "Ro-ro ship": "Ro-ro",
    "Oil tanker": "Petrolero",
    "Vehicle carrier": "Transporte de vehículos",
    "Bulk carrier": "Granelero",
    "Gas carrier": "Gasero",
    "Chemical tanker": "Quimiquero",
    "Other ship types": "Otros tipos",
    "General cargo ship": "Carga general",
    "Combination carrier": "Buque combinado",
}

# Paleta sobria: una sola familia de color, intensidad proporcional al valor.
COLOR_ALTO = "#1f4e79"
COLOR_BAJO = "#a8c4de"
COLOR_TEXTO = "#333333"


# --------------------------------------------------------------------------------------
# Deteccion de columnas
# --------------------------------------------------------------------------------------

PISTAS_TIPO = ("tipo", "ship_type", "type", "categoria", "category")
PISTAS_VALOR = ("pendiente", "t_hora", "co2_hora", "por_hora", "slope", "toneladas", "valor", "value")


def _detectar_columnas(df: pd.DataFrame) -> tuple[str, str]:
    """Encuentra la columna de tipo de buque y la del valor, sin depender de un nombre exacto."""
    col_tipo = None
    col_valor = None

    for col in df.columns:
        nombre = str(col).strip().lower()
        if col_tipo is None and any(p in nombre for p in PISTAS_TIPO):
            col_tipo = col
        if col_valor is None and any(p in nombre for p in PISTAS_VALOR):
            col_valor = col

    # Plan B: la primera columna no numerica es el tipo; la primera numerica, el valor.
    if col_tipo is None:
        no_numericas = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        col_tipo = no_numericas[0] if no_numericas else df.columns[0]
    if col_valor is None:
        numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numericas:
            raise SystemExit(
                f"No hay ninguna columna numerica en el CSV. Columnas: {list(df.columns)}"
            )
        col_valor = numericas[0]

    return col_tipo, col_valor


# --------------------------------------------------------------------------------------
# Figura
# --------------------------------------------------------------------------------------

def construir_figura(df: pd.DataFrame, col_tipo: str, col_valor: str, idioma: str):
    datos = df[[col_tipo, col_valor]].dropna().copy()
    datos[col_valor] = pd.to_numeric(datos[col_valor], errors="coerce")
    datos = datos.dropna(subset=[col_valor])

    if idioma == "es":
        datos[col_tipo] = datos[col_tipo].astype(str).str.strip().map(
            lambda x: TRADUCCION.get(x, x)
        )

    # Orden ascendente porque barh dibuja de abajo arriba: el mayor queda arriba.
    datos = datos.sort_values(col_valor, ascending=True)

    valores = datos[col_valor].to_numpy()
    etiquetas = datos[col_tipo].astype(str).to_numpy()

    # Intensidad de color proporcional al valor, dentro de una sola familia.
    vmin, vmax = valores.min(), valores.max()
    rango = (vmax - vmin) or 1.0
    mapa = matplotlib.colors.LinearSegmentedColormap.from_list("azul", [COLOR_BAJO, COLOR_ALTO])
    colores = [mapa((v - vmin) / rango) for v in valores]

    alto = max(3.2, 0.34 * len(valores) + 1.1)
    fig, ax = plt.subplots(figsize=(7.2, alto), dpi=300)

    ax.barh(etiquetas, valores, color=colores, height=0.68, zorder=3)

    # Valor al final de cada barra: evita tener que leer el eje.
    for y, v in enumerate(valores):
        ax.text(
            v + rango * 0.02,
            y,
            f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
            va="center",
            ha="left",
            fontsize=8.5,
            color=COLOR_TEXTO,
        )

    ax.set_xlim(0, vmax * 1.14)
    ax.set_xlabel("Toneladas de CO$_2$ por hora navegada", fontsize=9.5, color=COLOR_TEXTO)
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0, labelsize=9.5, colors=COLOR_TEXTO)
    ax.tick_params(axis="x", labelsize=8.5, colors=COLOR_TEXTO)

    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")

    fig.tight_layout()
    return fig, datos


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=CSV_POR_DEFECTO)
    p.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO, help="ruta sin extension")
    p.add_argument("--idioma", choices=("es", "en"), default="es")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"ERROR: no encuentro {args.csv}", file=sys.stderr)
        print("Pasa la ruta con --csv si el archivo esta en otro sitio.", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    col_tipo, col_valor = _detectar_columnas(df)
    print(f"Columnas detectadas -> tipo: '{col_tipo}' | valor: '{col_valor}'")

    fig, datos = construir_figura(df, col_tipo, col_valor, args.idioma)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    png = args.salida.with_suffix(".png")
    pdf = args.salida.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    alto = datos.iloc[-1]
    bajo = datos.iloc[0]
    ratio = alto[col_valor] / bajo[col_valor] if bajo[col_valor] else float("nan")
    print(f"Escrito: {png}")
    print(f"Escrito: {pdf}")
    print(
        f"Comprobación: {alto[col_tipo]} {alto[col_valor]:.2f} t/h frente a "
        f"{bajo[col_tipo]} {bajo[col_valor]:.2f} t/h (x{ratio:.2f})"
    )
    print("El texto de la memoria dice 6,48 / 3,02 y x2,15: si no coincide, revisa el CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
