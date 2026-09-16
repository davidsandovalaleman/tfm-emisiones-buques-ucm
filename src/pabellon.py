"""Pais de registro de cada buque, a partir del puerto de registro del MRV.

El registro publico trae una columna, `port_of_registry`, que cubre el
**96,4% de las emisiones de 2025**. Es la unica dimension geografica del dataset: no
hay coordenadas, ni puertos de escala, ni rutas. Convertirla en pais abre dos
cosas -- un mapa y la pregunta de **quien paga la factura europea desde fuera de Europa**.

Tres decisiones, y por que:

1. **El puerto viene sucio y hay que normalizarlo.** Mismo puerto escrito de seis maneras
   (`VALLETTA`, `VALETTA`, `VALLETA`; `COPENHAGEN`, `KOBENHAVN`, `KØBENHAVN`; cinco grafias de
   `ST. JOHN'S`). Se normaliza a mayusculas sin acentos, se colapsan los espacios, se quitan los
   sufijos de pais (`MAJURO, MARSHALL ISLANDS`) y se aplica una tabla de sinonimos.
2. **La tabla puerto -> pais es manual y parcial, y eso se declara.** Cubre los puertos que suman
   el ~96% de las emisiones de 2025; el resto queda como `No identificado`. Es preferible a un
   geocodificador: son 500 nombres, la cola es irrelevante y una tabla escrita a mano se puede
   revisar linea a linea.
3. **`registro_abierto` no es un juicio propio.** Marca los registros que la Federacion
   Internacional de los Trabajadores del Transporte (ITF) clasifica como pabellones de conveniencia.
   Se cita como lo que es: una lista publicada por un tercero, no una opinion del trabajo. El eje
   principal del analisis es otro y es puramente factual: **dentro o fuera del EEE**.

Limitacion que hay que decir en voz alta: el puerto de registro **no es exactamente el pabellon**.
Coinciden en los grandes registros (Monrovia-Liberia, Majuro-Islas Marshall, Panama, Nassau-Bahamas)
y pueden no coincidir en casos sueltos. Para lo que se usa aqui -- ordenar paises por emisiones y
por factura -- la aproximacion es buena; para afirmar el pabellon de un buque concreto, no.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# ── Espacio Economico Europeo: UE-27 mas Islandia, Liechtenstein y Noruega ──────────────────────
EEE = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN",
    "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP",
    "SWE", "ISL", "LIE", "NOR",
}

# ── Registros que la ITF clasifica como pabellones de conveniencia (lista publicada) ────────────
# Fuente: International Transport Workers' Federation, "Flags of Convenience" (lista vigente en
# 2026). Se marca a nivel de puerto porque un mismo pais puede tener un registro internacional
# separado: Madeira (MAR) esta en la lista y Lisboa no.
REGISTRO_ABIERTO_PUERTOS = {
    "MONROVIA", "MAJURO", "PANAMA", "NASSAU", "VALLETTA", "LIMASSOL", "HAMILTON", "DOUGLAS",
    "GIBRALTAR", "SAINT JOHNS", "BRIDGETOWN", "BASSETERRE", "FREETOWN", "WILLEMSTAD", "MADEIRA",
    "TORSHAVN", "GEORGE TOWN", "MALAKAL HARBOR", "MATA UTU",
    "FUNAFUTI", "KINGSTOWN", "PORT VILA", "BELIZE CITY", "PORT VICTORIA", "LIBERIA",
    "MALTA", "CYPRUS", "BAHAMAS", "MARSHALL ISLANDS",
}

# ── Sinonimos: variantes del mismo puerto ───────────────────────────────────────────────────────
SINONIMOS = {
    "VALETTA": "VALLETTA", "VALLETA": "VALLETTA",
    "GENOVA": "GENOA", "NAPLES": "NAPOLI",
    "KOBENHAVN": "COPENHAGEN", "COPENHAGUE": "COPENHAGEN",
    "HELSINGFORS": "HELSINKI",
    "GOTEBORG": "GOTHENBURG",
    "HONGKONG": "HONG KONG",
    "LIMASOL": "LIMASSOL",
    "ANTWERP": "ANTWERPEN",
    "DRAGOER": "DRAGOR",
    "ST JOHNS": "SAINT JOHNS", "ST. JOHNS": "SAINT JOHNS", "ST.JOHNS": "SAINT JOHNS",
    "SAINT-JOHNS": "SAINT JOHNS", "STJOHNS": "SAINT JOHNS",
    "LAS PALMAS DE GRAN CANARIA": "LAS PALMAS",
    "YANGPU CHINA": "YANG PU", "YANG SHAN CHINA": "YANG SHAN", "YANGPU": "YANG PU",
    "SAINTJOHNS": "SAINT JOHNS",
    "ROSTCK": "ROSTOCK", "MARSELLE": "MARSEILLE", "FOSNAVAAG": "FOSNAVAG",
    "HAI KOU": "HAIKOU", "PANAMA CITY": "PANAMA", "TANGIER": "TANGER",
    "FUNCHAL": "MADEIRA", "ALESUND": "AALESUND", "VENEZIA": "VENICE",
}

# ── Puerto normalizado -> (pais, ISO-3) ─────────────────────────────────────────────────────────
PUERTO_PAIS = {
    # registros grandes fuera del EEE
    "MONROVIA": ("Liberia", "LBR"), "MAJURO": ("Islas Marshall", "MHL"),
    "PANAMA": ("Panamá", "PAN"), "NASSAU": ("Bahamas", "BHS"),
    "SINGAPORE": ("Singapur", "SGP"), "HONG KONG": ("Hong Kong", "HKG"),
    "HAMILTON": ("Bermudas", "BMU"), "DOUGLAS": ("Isla de Man", "IMN"),
    "SAINT JOHNS": ("Antigua y Barbuda", "ATG"), "BRIDGETOWN": ("Barbados", "BRB"),
    "GEORGE TOWN": ("Islas Caimán", "CYM"), "GIBRALTAR": ("Gibraltar", "GIB"),
    "BASSETERRE": ("San Cristóbal y Nieves", "KNA"), "FREETOWN": ("Sierra Leona", "SLE"),
    "WILLEMSTAD": ("Curazao", "CUW"), "MALAKAL HARBOR": ("Palaos", "PLW"),
    "MATA UTU": ("Wallis y Futuna", "WLF"), "TORSHAVN": ("Islas Feroe", "FRO"),
    # EEE
    "VALLETTA": ("Malta", "MLT"), "LIMASSOL": ("Chipre", "CYP"), "PIKIS": ("Chipre", "CYP"),
    "MADEIRA": ("Portugal", "PRT"), "PORTUGAL": ("Portugal", "PRT"), "LISBOA": ("Portugal", "PRT"),
    "PIRAEUS": ("Grecia", "GRC"), "CHIOS": ("Grecia", "GRC"), "ANDROS": ("Grecia", "GRC"),
    "HERAKLION": ("Grecia", "GRC"), "CHANIA": ("Grecia", "GRC"), "SYROS": ("Grecia", "GRC"),
    "THESSALONIKI": ("Grecia", "GRC"), "PATRAS": ("Grecia", "GRC"), "KALYMNOS": ("Grecia", "GRC"),
    "PALERMO": ("Italia", "ITA"), "GENOA": ("Italia", "ITA"), "NAPOLI": ("Italia", "ITA"),
    "CATANIA": ("Italia", "ITA"), "BARI": ("Italia", "ITA"), "VENICE": ("Italia", "ITA"),
    "TRIESTE": ("Italia", "ITA"), "CAGLIARI": ("Italia", "ITA"),
    "HAMBURG": ("Alemania", "DEU"), "BREMEN": ("Alemania", "DEU"), "LUBECK": ("Alemania", "DEU"),
    "ROSTOCK": ("Alemania", "DEU"),
    "COPENHAGEN": ("Dinamarca", "DNK"), "ESBJERG": ("Dinamarca", "DNK"),
    "HELLERUP": ("Dinamarca", "DNK"), "DRAGOR": ("Dinamarca", "DNK"),
    "SVENDBORG": ("Dinamarca", "DNK"), "AARHUS": ("Dinamarca", "DNK"),
    "AALBORG": ("Dinamarca", "DNK"), "MARSTAL": ("Dinamarca", "DNK"),
    "ROENNE": ("Dinamarca", "DNK"), "SONDERBORG": ("Dinamarca", "DNK"),
    "HUMLEBAEK": ("Dinamarca", "DNK"), "SUNDBY": ("Dinamarca", "DNK"),
    "HIRTSHALS": ("Dinamarca", "DNK"), "SKAGEN": ("Dinamarca", "DNK"),
    "ROSKILDE": ("Dinamarca", "DNK"), "TAARBAEK": ("Dinamarca", "DNK"),
    "GRENAA": ("Dinamarca", "DNK"), "MARIBO": ("Dinamarca", "DNK"),
    "RINGKOBING": ("Dinamarca", "DNK"), "MUNKEBO": ("Dinamarca", "DNK"),
    "MOGELTONDER": ("Dinamarca", "DNK"),
    "MARSEILLE": ("Francia", "FRA"), "MORLAIX": ("Francia", "FRA"), "BASTIA": ("Francia", "FRA"),
    "AJACCIO": ("Francia", "FRA"), "LE HAVRE": ("Francia", "FRA"), "CAEN": ("Francia", "FRA"),
    "CHERBOURG": ("Francia", "FRA"),
    "SANTA CRUZ DE TENERIFE": ("España", "ESP"), "LAS PALMAS": ("España", "ESP"),
    "BERGEN": ("Noruega", "NOR"), "OSLO": ("Noruega", "NOR"), "HAUGESUND": ("Noruega", "NOR"),
    "TROMSO": ("Noruega", "NOR"), "STAVANGER": ("Noruega", "NOR"),
    "KRISTIANSAND": ("Noruega", "NOR"), "FOSNAVAG": ("Noruega", "NOR"),
    "SANDNES": ("Noruega", "NOR"), "ARENDAL": ("Noruega", "NOR"),
    "TRONDHEIM": ("Noruega", "NOR"), "HAMMERFEST": ("Noruega", "NOR"),
    "AKREHAMN": ("Noruega", "NOR"),
    "AMSTERDAM": ("Países Bajos", "NLD"), "DELFZIJL": ("Países Bajos", "NLD"),
    "ROTTERDAM": ("Países Bajos", "NLD"), "HARLINGEN": ("Países Bajos", "NLD"),
    "HOEK VAN HOLLAND": ("Países Bajos", "NLD"), "GRONINGEN": ("Países Bajos", "NLD"),
    "SNEEK": ("Países Bajos", "NLD"), "DORDRECHT": ("Países Bajos", "NLD"),
    "HEERENVEEN": ("Países Bajos", "NLD"), "VLISSINGEN": ("Países Bajos", "NLD"),
    "MARIEHAMN": ("Finlandia", "FIN"), "HELSINKI": ("Finlandia", "FIN"),
    "ECKERO": ("Finlandia", "FIN"), "PORVOO": ("Finlandia", "FIN"),
    "GOTHENBURG": ("Suecia", "SWE"), "STOCKHOLM": ("Suecia", "SWE"), "MALMO": ("Suecia", "SWE"),
    "VISBY": ("Suecia", "SWE"), "TRELLEBORG": ("Suecia", "SWE"), "DONSO": ("Suecia", "SWE"),
    "SUNDSVALL": ("Suecia", "SWE"),
    "ANTWERPEN": ("Bélgica", "BEL"), "LUXEMBOURG": ("Luxemburgo", "LUX"),
    "KLAIPEDA": ("Lituania", "LTU"), "TALLINN": ("Estonia", "EST"), "RIGA": ("Letonia", "LVA"),
    "ZADAR": ("Croacia", "HRV"), "RIJEKA": ("Croacia", "HRV"), "ARKLOW": ("Irlanda", "IRL"),
    # resto del mundo
    "LONDON": ("Reino Unido", "GBR"), "DOVER": ("Reino Unido", "GBR"),
    "HARWICH": ("Reino Unido", "GBR"),
    "ISTANBUL": ("Turquía", "TUR"), "IZMIR": ("Turquía", "TUR"), "TEKIRDAG": ("Turquía", "TUR"),
    "TOKYO": ("Japón", "JPN"), "KOBE": ("Japón", "JPN"), "JEJU": ("Corea del Sur", "KOR"),
    "NORFOLK": ("Estados Unidos", "USA"), "WILMINGTON, DE": ("Estados Unidos", "USA"),
    "YANG SHAN": ("China", "CHN"), "GUANGZHOU": ("China", "CHN"), "SHANGHAI": ("China", "CHN"),
    "YANG PU": ("China", "CHN"), "KEELUNG": ("Taiwán", "TWN"),
    "DAMMAM": ("Arabia Saudí", "SAU"), "KUWAIT": ("Kuwait", "KWT"),
    "LA GOULETTE": ("Túnez", "TUN"), "ALGIERS": ("Argelia", "DZA"), "ORAN": ("Argelia", "DZA"),
    "ALEXANDRIA": ("Egipto", "EGY"), "MANILA": ("Filipinas", "PHL"), "MUMBAI": ("India", "IND"),
    "CHATTOGRAM": ("Bangladés", "BGD"), "RIO DE JANEIRO": ("Brasil", "BRA"),
    "SAN MARINO": ("San Marino", "SMR"),
    # el propio nombre del pais aparece como puerto de registro en algunas declaraciones
    "MALTA": ("Malta", "MLT"), "LIBERIA": ("Liberia", "LBR"), "CYPRUS": ("Chipre", "CYP"),
    "BAHAMAS": ("Bahamas", "BHS"), "MARSHALL ISLANDS": ("Islas Marshall", "MHL"),
    "CHINA": ("China", "CHN"), "DENMARK": ("Dinamarca", "DNK"), "NORWAY": ("Noruega", "NOR"),
    "GREECE": ("Grecia", "GRC"), "ITALY": ("Italia", "ITA"), "NETHERLANDS": ("Países Bajos", "NLD"),
    # cola: puertos con mas de 25 kt de CO2 en 2025
    "FREDERIKSHAVN": ("Dinamarca", "DNK"), "ROMO": ("Dinamarca", "DNK"),
    "HERNING": ("Dinamarca", "DNK"), "RUNGSTED": ("Dinamarca", "DNK"),
    "SAKSKOBING": ("Dinamarca", "DNK"), "RODBYHAVN": ("Dinamarca", "DNK"),
    "SKOVSHOVED": ("Dinamarca", "DNK"), "ODENSE": ("Dinamarca", "DNK"),
    "VEJLE": ("Dinamarca", "DNK"), "NAKSKOV": ("Dinamarca", "DNK"),
    "EMDEN": ("Alemania", "DEU"), "PUTTGARDEN": ("Alemania", "DEU"),
    "LIVORNO": ("Italia", "ITA"), "VENICE": ("Italia", "ITA"),
    "FLORO": ("Noruega", "NOR"), "AALESUND": ("Noruega", "NOR"),
    "ABERDEEN": ("Reino Unido", "GBR"), "CARDIFF": ("Reino Unido", "GBR"),
    "BURGAS": ("Bulgaria", "BGR"),
    "FUNAFUTI": ("Tuvalu", "TUV"), "KINGSTOWN": ("San Vicente y las Granadinas", "VCT"),
    "PORT VILA": ("Vanuatu", "VUT"), "BELIZE CITY": ("Belice", "BLZ"),
    "PORT VICTORIA": ("Seychelles", "SYC"),
    "TANGER": ("Marruecos", "MAR"), "HAIFA": ("Israel", "ISR"),
    "BANGKOK": ("Tailandia", "THA"), "JAKARTA": ("Indonesia", "IDN"),
    "PORT KELANG": ("Malasia", "MYS"), "OITA": ("Japón", "JPN"), "HAIKOU": ("China", "CHN"),
}

NO_IDENTIFICADO = "No identificado"
SIN_DECLARAR = "Sin declarar"

_SUFIJOS = re.compile(
    r",\s*(LIBERIA|MARSHALL ISLANDS|CHINA|FAROE ISLANDS|MALTA|CYPRUS|PANAMA|BAHAMAS"
    r"|PORTUGAL|MADEIRA|DENMARK|NORWAY|GREECE|ITALY|SPAIN|FRANCE|GERMANY)$")


def normalizar(serie: pd.Series) -> pd.Series:
    """Deja el puerto en una forma comparable: mayusculas, sin acentos, sin sufijo de pais."""
    s = serie.astype("string").str.strip()
    s = s.mask(s.str.upper().isin(["NONE", "NAN", "N/A", "NA", "-", ""]))
    s = s.str.upper()
    # Ø, Æ, Å y las comillas tipograficas no se descomponen con NFKD: van a mano.
    s = (s.str.replace("Ø", "O", regex=False).str.replace("Æ", "AE", regex=False)
          .str.replace("Å", "A", regex=False)
          .str.replace("´", "'", regex=False).str.replace("`", "'", regex=False)
          .str.replace("’", "'", regex=False).str.replace("，", ",", regex=False))
    s = s.map(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()
              if pd.notna(x) else x)
    s = s.str.replace(r"[.'\-]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.replace(_SUFIJOS, "", regex=True).str.strip()
    # Amarres numerados del mismo puerto: "CHANIA 29", "CHANIA 30", "VENEZIA RI 042".
    s = s.str.replace(r"^(CHANIA|VENEZIA|PIRAEUS|SYROS)\b.*$", r"\1", regex=True)
    s = s.mask(s.eq(""))
    return s.replace(SINONIMOS)


def clasificar(serie: pd.Series) -> pd.DataFrame:
    """Devuelve `puerto_norm`, `pais`, `iso3`, `eee` y `registro_abierto` para cada fila."""
    p = normalizar(serie)
    pais = p.map(lambda x: PUERTO_PAIS.get(x, (None, None))[0] if pd.notna(x) else None)
    iso3 = p.map(lambda x: PUERTO_PAIS.get(x, (None, None))[1] if pd.notna(x) else None)
    pais = pais.fillna(pd.Series(NO_IDENTIFICADO, index=p.index).where(p.notna(), SIN_DECLARAR))
    return pd.DataFrame({
        "puerto_norm": p,
        "pais": pais,
        "iso3": iso3,
        "eee": iso3.isin(EEE),
        "registro_abierto": p.isin(REGISTRO_ABIERTO_PUERTOS),
    }, index=serie.index)


def cobertura(df: pd.DataFrame, col_co2: str = "total_co2_emissions_m_tonnes") -> dict:
    """Cuanto del universo queda identificado. Se publica siempre junto a los resultados."""
    c = clasificar(df["port_of_registry"])
    ok = c["iso3"].notna()
    total = float(df[col_co2].sum())
    return {
        "filas": int(len(df)),
        "puertos_distintos": int(c["puerto_norm"].nunique()),
        "filas_con_pais": int(ok.sum()),
        "pct_filas_con_pais": round(100 * ok.mean(), 2),
        "pct_emisiones_con_pais": round(100 * float(df.loc[ok, col_co2].sum()) / total, 2),
        "pct_emisiones_sin_declarar": round(
            100 * float(df.loc[c["pais"] == SIN_DECLARAR, col_co2].sum()) / total, 2),
    }
