"""
Funciones de apoyo para el EDA y la depuración de datos (atípicos, missings, imputación,
asociación con la variable objetivo).

Siguen el mismo criterio de depuración que `src/FuncionesMineria.py`, con ajustes de robustez
(evitar `np.isnan` sobre arrays con NaN de pandas nullable, fijar la semilla aleatoria de forma
opcional para reproducibilidad) y adaptadas a los nombres de columna del dataset de THETIS-MRV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler


def analizar_variables_categoricas(datos: pd.DataFrame) -> dict:
    """Frecuencias absolutas y relativas de cada variable categórica/texto de un DataFrame."""
    variables = list(datos.columns)
    numericas = datos.select_dtypes(include=["int", "int32", "int64", "float", "float32", "float64"]).columns
    categoricas = [v for v in variables if v not in numericas]

    resultados = {}
    for categoria in categoricas:
        resumen = pd.DataFrame(
            {
                "n": datos[categoria].value_counts(),
                "%": datos[categoria].value_counts(normalize=True),
            }
        )
        resultados[categoria] = resumen
    return resultados


def cuentaDistintos(datos: pd.DataFrame) -> pd.DataFrame:
    """Nº de valores distintos en cada variable numérica de un DataFrame."""
    numericas = datos.select_dtypes(include=["int", "int32", "int64", "float", "float32", "float64"])
    resultados = numericas.apply(lambda x: x.nunique())
    return pd.DataFrame({"Columna": resultados.index, "Distintos": resultados.values})


def atipicosAmissing(varaux: pd.Series) -> list:
    """
    Identifica valores atípicos en una serie numérica y los sustituye por NaN.

    Combina dos criterios:
      - Distancia a la media/mediana en desviaciones típicas (o MAD si la distribución es
        asimétrica), criterio 1.
      - Rango intercuartílico amplio (Q1 - 3·IQR, Q3 + 3·IQR), criterio 2.
    Solo se marca como atípico lo que cumple AMBOS criterios a la vez.

    Devuelve [serie_sin_atipicos, nº_de_atipicos_encontrados].
    """
    varaux = varaux.astype("float64")

    if abs(varaux.skew()) < 1:
        criterio1 = abs((varaux - varaux.mean()) / varaux.std()) > 3
    else:
        mad = sm.robust.mad(varaux.dropna(), axis=0)
        criterio1 = abs((varaux - varaux.median()) / mad) > 8

    qnt = varaux.quantile([0.25, 0.75]).dropna()
    Q1, Q3 = qnt.iloc[0], qnt.iloc[1]
    H = 3 * (Q3 - Q1)
    criterio2 = (varaux < (Q1 - H)) | (varaux > (Q3 + H))

    var = varaux.copy()
    atipicos = (criterio1 & criterio2).fillna(False)
    var[atipicos] = np.nan
    return [var, int(atipicos.sum())]


def patron_perdidos(datos_input: pd.DataFrame, figsize=(8, 6)):
    """Mapa de calor de correlación de valores ausentes entre las columnas que tienen algún missing."""
    cols_con_missing = datos_input.columns[datos_input.isna().sum() > 0]
    if len(cols_con_missing) < 2:
        print("Menos de 2 columnas con missings: no hay correlación de missings que mostrar.")
        return
    correlation_matrix = datos_input[cols_con_missing].isna().corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.figure(figsize=figsize)
    sns.set_theme(font_scale=1.0)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=False, mask=mask)
    plt.title("Matriz de correlación de valores ausentes")
    plt.tight_layout()
    # Nota: no se llama a plt.show() aquí a propósito. Bajo el backend inline de Jupyter,
    # plt.show() renderiza Y limpia la figura activa; si el notebook hace plt.savefig()
    # justo después de llamar a esta función (patrón usado en notebook 02), el savefig()
    # se queda sin nada que guardar y produce un PNG en blanco. Se deja la figura activa:
    # Jupyter la muestra igualmente sola al terminar la celda, y savefig() sí la captura.


def ImputacionCuant(var: pd.Series, tipo: str) -> pd.Series:
    """Imputa missings de una variable cuantitativa: 'media', 'mediana' o 'aleatorio' (según distribución empírica)."""
    vv = var.copy().astype("float64")
    faltan = vv.isna()

    if tipo == "media":
        vv[faltan] = round(vv.mean(), 4)
    elif tipo == "mediana":
        vv[faltan] = round(vv.median(), 4)
    elif tipo == "aleatorio":
        x = vv[~faltan]
        frec = x.value_counts(normalize=True).reset_index()
        frec.columns = ["Valor", "Frec"]
        frec = frec.sort_values(by="Valor")
        frec["FrecAcum"] = frec["Frec"].cumsum()
        random_values = np.random.uniform(min(frec["FrecAcum"]), 1, int(faltan.sum()))
        imputados = [round(list(frec["Valor"][frec["FrecAcum"] <= r])[-1], 4) for r in random_values]
        vv[faltan] = imputados
    else:
        raise ValueError("tipo debe ser 'media', 'mediana' o 'aleatorio'")
    return vv


def ImputacionCuali(var: pd.Series, tipo: str) -> pd.Series:
    """Imputa missings de una variable cualitativa: 'moda' o 'aleatorio' (a partir de la distribución empírica)."""
    vv = var.copy()
    faltan = vv.isna()

    if tipo == "moda":
        moda = vv[~faltan].value_counts().idxmax()
        vv[faltan] = moda
    elif tipo == "aleatorio":
        vv[faltan] = np.random.choice(vv[~faltan], size=int(faltan.sum()), replace=True)
    else:
        raise ValueError("tipo debe ser 'moda' o 'aleatorio'")
    return vv


def Vcramer(v: pd.Series, target: pd.Series) -> float:
    """V de Cramer entre dos variables (discretiza en quintiles las que sean numéricas)."""
    v = v.copy()
    target = target.copy()

    if pd.api.types.is_numeric_dtype(v):
        p = sorted(set(v.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).dropna()))
        v = pd.cut(v, bins=p, include_lowest=True)
        v = v.astype("object").fillna("missing")
    if pd.api.types.is_numeric_dtype(target):
        p = sorted(set(target.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).dropna()))
        target = pd.cut(target, bins=p, include_lowest=True)
        target = target.astype("object").fillna("missing")

    v = v.reset_index(drop=True)
    target = target.reset_index(drop=True)
    tabla_cruzada = pd.crosstab(v, target)
    chi2 = chi2_contingency(tabla_cruzada)[0]
    n = tabla_cruzada.sum().sum()
    return float(np.sqrt(chi2 / (n * (min(tabla_cruzada.shape) - 1))))


def graficoVcramer(matriz: pd.DataFrame, target: pd.Series, figsize=(10, 6)):
    """Gráfico de barras con la V de Cramer de cada columna de `matriz` frente a `target`."""
    salida = {x: Vcramer(matriz[x], target) for x in matriz.columns}
    ordenado = dict(sorted(salida.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=figsize)
    plt.barh(list(ordenado.keys()), list(ordenado.values()), color="#3B6E8F")
    plt.xlabel("V de Cramer")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # Ver nota en patron_perdidos(): se omite plt.show() a propósito para que el
    # plt.savefig() posterior en el notebook no se encuentre la figura ya vaciada.
    return ordenado


def mejorTransfCorr(vv: pd.Series, target: pd.Series):
    """
    Busca, entre un catálogo de transformaciones habituales (log, exp, cuadrado, raíz...), la que
    maximiza la correlación de Pearson de `vv` con una `target` continua.

    Devuelve [nombre_transformación, serie_transformada].
    """
    vv_arr = StandardScaler().fit_transform([[x] for x in list(vv)])
    vv_arr = vv_arr + abs(np.min(vv_arr)) * 1.0001
    vv_arr = [x[0] for x in vv_arr]

    posibles_transf = pd.DataFrame(
        {
            "x": vv_arr,
            "logx": np.log(vv_arr),
            "expx": np.exp(np.clip(vv_arr, None, 20)),  # evita overflow con exp() de valores grandes
            "sqrx": [x**2 for x in vv_arr],
            "sqrtx": np.sqrt(vv_arr),
            "raiz4": [x ** (1 / 4) for x in vv_arr],
        }
    )
    cor_values = posibles_transf.apply(lambda col: abs(np.corrcoef(target, col)[0, 1]), axis=0)
    mejor = cor_values.idxmax()
    return [mejor, posibles_transf[mejor]]
