"""Prepara los extractos que alimentan el tablero de Tableau .

    python scripts/exportar_tableau.py

**Por qué un script y no un `.twbx` a mano.** El tablero se construye en Tableau, pero lo que decide
si el tablero es correcto son los datos que lee. Dejarlos hechos a mano dentro del `.twbx` los
convertiría en artefactos que nadie puede regenerar. Aquí las cuatro fuentes salen de los informes ya versionados, con el mismo criterio
de población que el resto del proyecto, y el `.twbx` solo las conecta y las pinta.

**Las cuatro fuentes, y qué hoja alimenta cada una:**

| Fichero | Hoja del tablero | Grano |
|---|---|---|
| `pabellon_mapa.csv` | mapa de riesgo por pabellón | país de registro |
| `cii_trayectoria.csv` | serie 2025-2030 con filtro de año | buque × año (formato largo) |
| `buques.csv` | dispersión y tabla de detalle | buque |
| `navieras.csv` | ranking de exposición | naviera |

El formato largo de `cii_trayectoria.csv` es deliberado: Tableau anima y filtra por año sin pivotar
nada, que es justo lo que la tabla ancha de `18_pasaporte_buques.csv` no deja hacer.
"""
from __future__ import annotations

import os

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFORMES = os.path.join(RAIZ, "reports")
SALIDA = os.path.join(INFORMES, "tableau")
ANIOS = range(2025, 2031)

TRADUCCION_BANDA = {"A": "A · muy por encima", "B": "B · por encima",
                    "C": "C · cumple", "D": "D · por debajo", "E": "E · muy por debajo"}


def main() -> None:
    os.makedirs(SALIDA, exist_ok=True)
    ficha = pd.read_csv(f"{INFORMES}/18_pasaporte_buques.csv")

    # --- 1. mapa por pabellón: ya viene con ISO-3, que es lo que Tableau geocodifica solo
    mapa = pd.read_csv(f"{INFORMES}/19_pabellon_mapa.csv")
    mapa = mapa.rename(columns={
        "pais": "País de registro", "iso3": "ISO3", "bloque": "Bloque",
        "buques": "Buques", "co2_Mt": "CO2 (Mt)", "factura_MEUR": "Factura 2026 (M EUR)",
        "ahorro_Mt": "Ahorro alcanzable (Mt)", "pct_de_2025": "% en D/E 2025",
        "pct_de_2030": "% en D/E 2030", "pct_registro_abierto": "% registro abierto (ITF)",
        "cuota_pct": "% de la factura"})
    mapa["Deterioro 2025-2030 (puntos)"] = (mapa["% en D/E 2030"] - mapa["% en D/E 2025"]).round(1)
    mapa.to_csv(f"{SALIDA}/pabellon_mapa.csv", index=False, encoding="utf-8-sig")

    # --- 2. trayectoria CII en formato largo
    columnas = ["ship_imo_number", "ship_name", "company_name", "ship_type_agrupado",
                "cii_tipo", "capacidad_estimada", "ets_2026_eur"]
    largo = ficha.melt(id_vars=columnas, value_vars=[f"banda_{a}" for a in ANIOS],
                       var_name="anio", value_name="banda")
    largo["anio"] = largo["anio"].str.removeprefix("banda_").astype(int)
    largo = largo.dropna(subset=["banda"])
    largo["banda_etiqueta"] = largo["banda"].map(TRADUCCION_BANDA)
    largo["incumple"] = largo["banda"].isin(["D", "E"])
    largo = largo.rename(columns={
        "ship_imo_number": "IMO", "ship_name": "Buque", "company_name": "Naviera",
        "ship_type_agrupado": "Tipo", "cii_tipo": "Tipo CII",
        "capacidad_estimada": "Capacidad (t)", "ets_2026_eur": "Factura ETS 2026 (EUR)",
        "anio": "Año", "banda": "Banda", "banda_etiqueta": "Banda (etiqueta)",
        "incumple": "En D/E"})
    largo.to_csv(f"{SALIDA}/cii_trayectoria.csv", index=False, encoding="utf-8-sig")

    # --- 3. detalle por buque
    buques = ficha.rename(columns={
        "ship_imo_number": "IMO", "ship_name": "Buque", "company_name": "Naviera",
        "ship_type_agrupado": "Tipo", "capacidad_estimada": "Capacidad (t)",
        "total_co2_emissions_m_tonnes": "CO2 declarado (t)", "pred_honesta": "CO2 predicho (t)",
        "residuo_rel": "Desvío sobre sus pares", "grupo": "Grupo de comparación",
        "ahorro_actividad_constante_t": "Ahorro a actividad constante (t)",
        "ahorro_eur": "Valor del ahorro (EUR)", "ets_2026_eur": "Factura ETS 2026 (EUR)",
        "penalizacion_fueleu_eur": "Multa FuelEU (EUR)", "cumple_fueleu": "Cumple FuelEU",
        "ghgie_wtw": "Intensidad GEI (gCO2eq/MJ)", "banda_2025": "Banda CII 2025",
        "anio_caida": "Año en que cae a D/E", "time_spent_at_sea_hours": "Horas de mar",
        "distancia_nm": "Distancia (mn)", "velocidad_nudos": "Velocidad (nudos)"})
    buques["Banda CII 2025 (etiqueta)"] = buques["Banda CII 2025"].map(TRADUCCION_BANDA)
    buques["Factura total 2026 (EUR)"] = (buques["Factura ETS 2026 (EUR)"]
                                          + buques["Multa FuelEU (EUR)"].fillna(0))
    buques.to_csv(f"{SALIDA}/buques.csv", index=False, encoding="utf-8-sig")

    # --- 4. ranking de navieras
    nav = pd.read_csv(f"{INFORMES}/16_navieras_exposicion.csv")
    nav = nav.rename(columns={
        "nombre": "Naviera", "buques": "Buques", "co2_t": "CO2 (t)",
        "ets_eur": "ETS 2026 (EUR)", "fueleu_eur": "FuelEU (EUR)",
        "factura_eur": "Factura total (EUR)", "ahorro_t": "Ahorro alcanzable (t)",
        "cuota_pct": "% de la factura", "cuota_acumulada_pct": "% acumulado"})
    nav.head(300).to_csv(f"{SALIDA}/navieras.csv", index=False, encoding="utf-8-sig")

    for nombre in ("pabellon_mapa", "cii_trayectoria", "buques", "navieras"):
        d = pd.read_csv(f"{SALIDA}/{nombre}.csv")
        print(f"  tableau/{nombre}.csv: {len(d):,} filas, {d.shape[1]} columnas".replace(",", "."))
    print(f"\nCuatro extractos escritos en {SALIDA}.")


if __name__ == "__main__":
    main()
