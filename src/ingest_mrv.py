"""
Ingesta de los informes públicos anuales de THETIS-MRV (EU MRV, Reglamento (UE) 2015/757).

Cada fichero .xlsx descargado del portal (https://mrv.emsa.europa.eu/#public/emission-report)
tiene una cabecera de 3 filas (grupo de variable / subgrupo / nombre de variable) con celdas
combinadas, y los datos empiezan en la fila 4. A partir de 2024 se añaden dos hojas por año
("... Full ERs" y "... Partial ERs", informes completos y parciales del Art. 11(2)) y se añaden
columnas de CH4 y N2O que no existen en años anteriores.

Este módulo aplana esa cabecera a nombres de columna únicos y homogéneos entre años, y
consolida todos los años en un único DataFrame.
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path

import openpyxl
import pandas as pd

# Año -> nombre de fichero (dentro de data/raw) y hojas a leer (hoja, cobertura)
YEAR_FILES = {
    2018: ("2018-v275-EU_MRV_Publication_of_information.xlsx", [("2018", "Full")]),
    2019: ("2019-v228-EU_MRV_Publication_of_information.xlsx", [("2019", "Full")]),
    2020: ("2020-v209-EU_MRV_Publication_of_information.xlsx", [("2020", "Full")]),
    2021: ("2021-v218-EU_MRV_Publication_of_information.xlsx", [("2021", "Full")]),
    2022: ("2022-v241-EU_MRV_Publication_of_information.xlsx", [("2022", "Full")]),
    2023: ("2023-v91-EU_MRV_Publication_of_information.xlsx", [("2023", "Full")]),
    2024: (
        "2024-v235-EU_MRV_Publication_of_information.xlsx",
        [("2024 Full ERs", "Full"), ("2024 Partial ERs", "Partial")],
    ),
    2025: (
        "2025-v37-EU_MRV_Publication_of_information.xlsx",
        [("2025 Full ERs", "Full"), ("2025 Partial ERs", "Partial")],
    ),
}

HEADER_ROWS = 3  # filas 1-3 = cabecera (grupo, subgrupo, nombre); los datos empiezan en la fila 4


def _slugify(text: str) -> str:
    """Convierte una etiqueta legible en un identificador snake_case ASCII válido para pandas."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _ffill_row(row):
    """Rellena hacia adelante los valores de una fila de cabecera (simula las celdas combinadas)."""
    out, last = [], None
    for v in row:
        if v is not None and str(v).strip() != "":
            last = v
        out.append(last)
    return out


def _read_header_rows(raw_dir: Path, filename: str, sheet_name: str):
    """Lee solo las 3 filas de cabecera de una hoja (rápido, sin cargar los datos)."""
    path = Path(raw_dir) / filename
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(min_row=1, max_row=HEADER_ROWS, values_only=True)
    row1, row2, row3 = next(rows_iter), next(rows_iter), next(rows_iter)
    wb.close()
    return row1, row2, row3


def _global_ambiguous_names(raw_dir: Path) -> set:
    """
    Recorre TODAS las hojas de TODOS los años y determina qué nombres de columna (row3)
    aparecen duplicados en al menos una hoja (p.ej. 'IMO Number' y 'Name' se repiten para
    Ship y para Company solo en las hojas de 2024 en adelante).

    Se usa para aplicar la MISMA regla de desambiguación (prefijo de grupo) en todos los años,
    aunque en un año concreto esa columna no esté duplicada — así 'imo_number' identifica
    siempre al buque, en todos los años, en vez de llamarse distinto según el año.
    """
    ambiguous = set()
    for year, (filename, sheets) in YEAR_FILES.items():
        for sheet_name, _coverage in sheets:
            _row1, _row2, row3 = _read_header_rows(raw_dir, filename, sheet_name)
            names = [str(v).strip() for v in row3 if v is not None]
            counts = Counter(names)
            ambiguous.update(name for name, n in counts.items() if n > 1)
    return ambiguous


def _flatten_header(row1, row2, row3, ambiguous_names: set):
    """
    Construye nombres de columna únicos y estables entre años a partir de las 3 filas de cabecera.

    `ambiguous_names` es el conjunto (precalculado con `_global_ambiguous_names`) de nombres
    que hay que desambiguar prefijando el grupo (columna 'Ship'/'Company', etc.), para que la
    misma variable reciba siempre el mismo nombre de columna independientemente del año.

    Devuelve dos listas paralelas:
      - column_names: nombres snake_case usados en el DataFrame
      - column_labels: etiquetas legibles (para el diccionario de variables)
    """
    group1 = _ffill_row(row1)
    raw_names = [str(v).strip() if v is not None else f"col_{i}" for i, v in enumerate(row3)]

    labels = []
    seen_group_name = Counter()
    for g1, name in zip(group1, raw_names):
        # Los nombres de una sola letra ('A', 'B', 'C', 'D' — las opciones de método de
        # monitorización del Reglamento MRV) no son ambiguos dentro de una misma hoja, pero
        # tampoco dicen nada por sí solos fuera de contexto: se prefijan siempre con el grupo,
        # igual que los que sí están duplicados.
        needs_group_prefix = (name in ambiguous_names) or (len(name) <= 1)
        if not needs_group_prefix:
            label = name
        else:
            # Nombre ambiguo en el conjunto global: se prefija con el grupo para desambiguar,
            # y si aun así se repite dentro del mismo grupo (p.ej. 'D' y 'D' en Monitoring
            # methods), se añade un contador posicional.
            key = (g1, name)
            seen_group_name[key] += 1
            occurrence = seen_group_name[key]
            base = f"{g1} {name}".strip() if g1 else name
            label = base if occurrence == 1 else f"{base} {occurrence}"
        labels.append(label)

    # Por si dos labels acaban coincidiendo igualmente, se añade un sufijo numérico de seguridad.
    seen_label = Counter()
    unique_labels = []
    for label in labels:
        seen_label[label] += 1
        unique_labels.append(label if seen_label[label] == 1 else f"{label} ({seen_label[label]})")

    column_names = [_slugify(lbl) for lbl in unique_labels]
    return column_names, unique_labels


def load_year_sheet(
    raw_dir: Path, filename: str, sheet_name: str, year: int, coverage: str, ambiguous_names: set
) -> pd.DataFrame:
    """Carga una hoja (un año, o un año+cobertura parcial) y devuelve un DataFrame limpio."""
    path = Path(raw_dir) / filename
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name]

    rows_iter = ws.iter_rows(values_only=True)
    row1 = next(rows_iter)
    row2 = next(rows_iter)
    row3 = next(rows_iter)
    column_names, column_labels = _flatten_header(row1, row2, row3, ambiguous_names)

    data = list(rows_iter)
    df = pd.DataFrame(data, columns=column_names)

    df.insert(0, "reporting_year", year)
    df.insert(1, "report_coverage", coverage)
    df.insert(2, "source_file", filename)

    wb.close()
    return df, dict(zip(column_names, column_labels))


# Columnas que son identificadores o texto y NUNCA deben convertirse a numérico,
# aunque su contenido parezca mayoritariamente numérico (p.ej. IMO Number).
NON_NUMERIC_COLUMNS = {
    "reporting_year",
    "report_coverage",
    "source_file",
    "imo_number",
    "ship_imo_number",
    "company_imo_number",
    "name",
    "ship_name",
    "company_name",
    "ship_type",
    "reporting_period",
    "technical_efficiency",
    "port_of_registry",
    "home_port",
    "ice_class",
    "doc_issue_date",
    "doc_expiry_date",
    "verifier_number",
    "verifier_name",
    "verifier_nab",
    "verifier_address",
    "verifier_city",
    "verifier_accreditation_number",
    "verifier_country",
    # Campo de texto libre (comentarios del armador/verificador), nunca debe interpretarse
    # como numérico aunque, por casualidad, algún comentario suelto contenga solo un número.
    "additional_information_to_facilitate_the_understanding_of_the_reported_average_operational_energy_efficiency_indicators",
}

# Valores centinela usados en el fichero de origen para "no aplica" / "sin dato", que hay que
# tratar como missing (NaN) antes de intentar convertir una columna a numérico.
SENTINEL_VALUES = {"n/a", "na", "not applicable", "-", "", "nd", "n.d."}


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte a numérico las columnas que son mayoritariamente numéricas, tratando los
    valores centinela ('N/A', 'Not Applicable', ...) como missing. Las columnas identificativas
    o de texto (NON_NUMERIC_COLUMNS, o las de un solo carácter A/B/C/D de métodos de monitorización,
    que son indicadores 'Yes'/vacío) se dejan tal cual.
    """
    for col in df.columns:
        if col in NON_NUMERIC_COLUMNS:
            continue
        if df[col].dtype != object:
            continue

        cleaned = df[col].apply(
            lambda v: None if isinstance(v, str) and v.strip().lower() in SENTINEL_VALUES else v
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")

        non_null = cleaned.notna().sum()
        if non_null == 0:
            continue
        # Si la MAYORÍA de los valores no-nulos se han podido convertir a número, la columna es
        # numérica por naturaleza (p.ej. ratios de eficiencia, donde hasta un ~15% de las filas
        # pueden venir como texto de error tipo 'Division by zero!' cuando el denominador es 0):
        # se adopta la versión numérica y ese texto residual pasa a NaN, igual que un missing.
        # Si la mayoría NO es convertible (columnas de texto genuinas: indicadores Yes/No de
        # métodos de monitorización, o texto libre con algún número suelto), se deja el texto.
        # El umbral es 0.5 (mayoría) en vez de exigir casi el 100%, precisamente para no perder
        # columnas numéricas legítimas con muchos 'Division by zero!'.
        if numeric.notna().sum() / non_null >= 0.5:
            df[col] = numeric
        else:
            df[col] = cleaned
    return df


# Pares (columna_canónica, columna_variante) que representan el MISMO concepto de negocio pero
# terminaron con nombres distintos porque el rótulo del grupo de cabecera del Excel cambió entre
# rangos de años (el propio portal ha ido renombrando secciones). Se detectó porque, para el
# mismo buque-año, ambas columnas nunca coexisten (una solo tiene datos en 2018-2019, la otra en
# 2020 en adelante) y, en los años en los que SÍ coexisten dos variantes del mismo concepto
# (2020-2023), sus valores son idénticos en el 100% de los casos comprobados (ver notebook 02,
# sección de EDA sobre "tiempo en mar"). Sin esta fusión, cualquier análisis que usara solo la
# columna canónica perdería ~77% de los datos de la variable, cuando en realidad sí están
# disponibles bajo el otro nombre.
_CONCEPTOS_DIVIDIDOS = [
    # (columna canónica que se rellena, columna variante que la alimenta cuando falta)
    ("time_spent_at_sea_hours", "total_time_spent_at_sea_hours"),
    ("annual_time_spent_at_sea_hours", "annual_total_time_spent_at_sea_hours"),
    # Mismo patrón que las dos parejas de arriba, detectado con un barrido sistemático de todas
    # las columnas (ver cuaderno 01, secciones 9-10). El grupo "navegación en hielo" del portal también cambió de
    # rótulo entre 2018-2023 ("Total time spent at sea through ice" / "Through ice") y 2024-2025
    # ("Time spent at sea through ice" / "Distance through ice"). Verificado: solape 0 filas entre
    # cada pareja (nunca coexisten) y mismas unidades/etiqueta de negocio.
    ("time_spent_at_sea_through_ice_hours", "total_time_spent_at_sea_through_ice_hours"),
    ("distance_through_ice_n_miles", "through_ice_n_miles"),
]


def _coalesce_split_concepts(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena la columna canónica de cada concepto dividido con su variante, sin perder ninguna."""
    for canonica, variante in _CONCEPTOS_DIVIDIDOS:
        if canonica in df.columns and variante in df.columns:
            # Comprobación defensiva: si en algún momento ambas columnas tienen dato para la
            # misma fila, deben coincidir (si no, no son realmente el mismo concepto y no se
            # deben fusionar a ciegas). Se comprueba en cada ejecución para que un cambio
            # futuro del portal no pase desapercibido.
            ambas = df[[canonica, variante]].dropna()
            if len(ambas) > 0:
                iguales = (ambas[canonica] == ambas[variante]).mean()
                if iguales < 0.99:
                    raise ValueError(
                        f"'{canonica}' y '{variante}' se suponen el mismo concepto pero solo "
                        f"coinciden en el {iguales:.1%} de las {len(ambas)} filas donde ambas "
                        "tienen dato. Revisar antes de fusionar (puede que ya no sean el mismo "
                        "concepto, p.ej. si el portal cambia de definición)."
                    )
            df[canonica] = df[canonica].fillna(df[variante])
    return df


def _detectar_conceptos_partidos_sin_resolver(df: pd.DataFrame, cols_ya_resueltas: set) -> list:
    """
    Barrido defensivo: busca pares de columnas cuyo nombre difiere en un único
    token (p.ej. 'total_x' vs 'x') y cuya cobertura de datos es prácticamente complementaria
    (casi nunca tienen dato ambas a la vez) — la firma de un "concepto partido" como los
    ya resueltos. Se ejecuta después de `_coalesce_split_concepts` para avisar si queda
    algún par sin resolver en `_CONCEPTOS_DIVIDIDOS`, en vez de depender de detectarlo a mano
    explorando el EDA.
    """
    numeric_cols = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c not in NON_NUMERIC_COLUMNS and c not in cols_ya_resueltas
    ]
    n = len(df)
    sospechosos = []
    for c1 in numeric_cols:
        tokens1 = c1.split("_")
        for i in range(len(tokens1)):
            c2 = "_".join(tokens1[:i] + tokens1[i + 1 :])
            if c2 not in numeric_cols or c2 <= c1:
                continue
            nn1, nn2 = df[c1].notna(), df[c2].notna()
            cov1, cov2 = nn1.mean(), nn2.mean()
            if cov1 < 0.01 or cov2 < 0.01:
                continue
            overlap = (nn1 & nn2).sum()
            if overlap <= max(5, 0.001 * n):
                sospechosos.append((c1, c2, cov1, cov2, overlap))
    return sospechosos


def load_all_years(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga y consolida todos los años definidos en YEAR_FILES.

    Devuelve:
      - df: DataFrame consolidado (una fila por buque, año y cobertura)
      - diccionario_variables: DataFrame con columna técnica -> etiqueta original del portal
    """
    ambiguous_names = _global_ambiguous_names(raw_dir)

    frames = []
    # Columnas de metadatos añadidas por este script (no vienen del Excel de origen), documentadas
    # aquí para que el diccionario de variables cubra el 100% de las columnas del dataset final.
    dictionary = {
        "reporting_year": "Año del informe (añadido en la ingesta, a partir del fichero de origen)",
        "report_coverage": "Cobertura del informe: 'Full' (año completo) o 'Partial' "
        "(informe parcial Art. 11(2), p.ej. por cambio de compañía a mitad de año) "
        "(añadido en la ingesta)",
        "source_file": "Fichero .xlsx de data/raw/ del que procede la fila (añadido en la ingesta)",
    }
    for year, (filename, sheets) in sorted(YEAR_FILES.items()):
        for sheet_name, coverage in sheets:
            df_year, labels = load_year_sheet(raw_dir, filename, sheet_name, year, coverage, ambiguous_names)
            frames.append(df_year)
            dictionary.update(labels)
            print(f"  {year} ({coverage:7s}) <- '{sheet_name}': {len(df_year):>6d} filas, {df_year.shape[1]} columnas")

    df = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # 'reporting_period' llega mixto entre hojas: año como float (informes completos, p.ej.
    # 2018.0) o como rango de fechas en texto (informes parciales, p.ej. '2024 (1/1 - 21/10)').
    # Se normaliza todo a texto para poder guardar en parquet sin ambigüedad de tipos.
    if "reporting_period" in df.columns:
        def _normalize_period(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            if isinstance(v, (int, float)):
                return str(int(v))
            return str(v).strip()

        df["reporting_period"] = df["reporting_period"].apply(_normalize_period)

    assert df.columns.is_unique, (
        "Nombres de columna duplicados tras el aplanado de cabecera — el slugify de dos "
        "etiquetas distintas ha colisionado en el mismo nombre final. Columnas duplicadas: "
        f"{df.columns[df.columns.duplicated()].tolist()}"
    )

    df = _clean_numeric_columns(df)
    df = _coalesce_split_concepts(df)

    cols_resueltas = {c for par in _CONCEPTOS_DIVIDIDOS for c in par}
    sospechosos = _detectar_conceptos_partidos_sin_resolver(df, cols_resueltas)
    if sospechosos:
        print(
            f"\n⚠️  AVISO: se han detectado {len(sospechosos)} posible(s) par(es) de columnas con "
            "el mismo patrón de 'concepto partido' (ver secciones 9-10 del cuaderno 01) "
            "que NO están en _CONCEPTOS_DIVIDIDOS. Revisar antes de dar la ingesta por buena:"
        )
        for c1, c2, cov1, cov2, overlap in sospechosos:
            print(f"   - {c1} <-> {c2}  (cobertura {cov1:.1%} / {cov2:.1%}, solape={overlap} filas)")

    # Anotar en el diccionario que las columnas "variante" ya han sido volcadas en su canónica
    # (se mantienen en el dataset por transparencia, pero están redundantes para el análisis).
    for canonica, variante in _CONCEPTOS_DIVIDIDOS:
        if variante in dictionary:
            dictionary[variante] += (
                f" [REDUNDANTE: mismo concepto que '{canonica}' bajo un rótulo de grupo distinto "
                f"según el año; sus valores ya se han volcado en '{canonica}' cuando esta estaba "
                "vacía — usar la columna canónica para el análisis]"
            )

    dict_df = (
        pd.DataFrame(sorted(dictionary.items()), columns=["columna", "etiqueta_original_portal"])
        .sort_values("columna")
        .reset_index(drop=True)
    )
    return df, dict_df


if __name__ == "__main__":
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando y consolidando los 8 años de THETIS-MRV...")
    df, dict_df = load_all_years(raw_dir)

    print(f"\nDataset consolidado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(df["reporting_year"].value_counts().sort_index())

    out_parquet = processed_dir / "mrv_consolidado.parquet"
    out_csv_dict = processed_dir / "diccionario_variables.csv"
    df.to_parquet(out_parquet, index=False)
    dict_df.to_csv(out_csv_dict, index=False)
    print(f"\nGuardado: {out_parquet}")
    print(f"Guardado: {out_csv_dict}")
