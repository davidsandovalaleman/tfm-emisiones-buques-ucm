"""
Funciones de modelización ML clásico (notebook 04).

La selección de variables y el modelo de regresión lineal se apoyan en src/FuncionesMineria.py,
que no se duplica aquí.

Este módulo cubre la comparativa con otras técnicas (regresión regularizada y ensembles de
árboles) sobre los datasets ya codificados en one-hot
(mrv_features_{principal,control}_ml.parquet), usando la misma partición es_train calculada en el
notebook 03 de feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import RANDOM_STATE

ID_COLS = ["reporting_year", "ship_imo_number", "ship_name", "company_name"]
TARGET_COL = "total_co2_emissions_m_tonnes"
COLS_NO_PREDICTORAS = ID_COLS + [TARGET_COL, "es_train", "target_reconstruido"]

# Variables continuas / categóricas del modelo principal, en la codificación "raw" (sin one-hot)
# que esperan las funciones de FuncionesMineria.py (lm, lm_stepwise, crear_data_modelo...), que
# hacen ellas mismas el pd.get_dummies internamente.
#
# es_subcategoria_nueva se trata como continua (ya es un flag 0/1: tratarla como categórica de
# 2 niveles produce la misma columna dummy única) -- igual que se dejó sin expandir en la versión
# _ml del notebook 03. prop_missings sí se trata como categórica, siguiendo la misma decisión
# tomada en el notebook 02 (EDA).
VAR_CONT_PRINCIPAL = ["time_spent_at_sea_hours", "te_valor", "es_subcategoria_nueva"]
VAR_CATEG_PRINCIPAL = ["ship_type_agrupado", "ice_class", "te_metodo", "prop_missings"]

VAR_CONT_CONTROL = VAR_CONT_PRINCIPAL + ["total_fuel_consumption_m_tonnes"]
VAR_CATEG_CONTROL = VAR_CATEG_PRINCIPAL


def separar_train_test_ml(df_ml: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa un dataset ya codificado en one-hot (mrv_features_{principal,control}_ml.parquet) en
    X/y de train y test, usando la columna es_train ya calculada (partición 80/20 agrupada por
    buque, calculada en el notebook 03). Descarta las columnas identificadoras y las de
    control (es_train, target_reconstruido), que no son predictoras.

    Returns:
        x_train, x_test, y_train, y_test
    """
    predictoras = [c for c in df_ml.columns if c not in COLS_NO_PREDICTORAS]

    train = df_ml[df_ml["es_train"]]
    test = df_ml[~df_ml["es_train"]]

    return train[predictoras], test[predictoras], train[TARGET_COL], test[TARGET_COL]


def predecir(modelo, x) -> np.ndarray:
    """
    Predicción con la restricción física del dominio: las emisiones de CO2 no pueden ser negativas.

    Ningún regresor de los que se comparan aquí (lineal, regularizado o de árboles) conoce esa
    restricción, así que todos producen alguna predicción por debajo de cero en las filas de
    actividad muy baja. Recortar a 0 es la proyección del pronóstico sobre el conjunto de valores
    físicamente posibles: nunca puede empeorar el error (el valor real siempre es >= 0, así que 0
    está siempre al menos tan cerca como cualquier negativo) y evita que el simulador y la API
    sirvan un modelo capaz de predecir emisiones negativas.

    Se aplica de forma uniforme a todas las técnicas comparadas, para que la comparación siga
    siendo justa.
    """
    return np.clip(np.asarray(modelo.predict(x), dtype=float), 0.0, None)


def calcular_metricas(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Métricas estándar de regresión: RMSE y MAE en las unidades originales (toneladas de CO2),
    y R^2. MAE se añade porque, con una distribución del target tan asimétrica (ver notebook 02),
    es menos sensible que RMSE a los buques de mayor tamaño y complementa la lectura.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def entrenar_comparativa_sklearn(df_ml: pd.DataFrame, modelos: dict, nombre_dataset: str) -> pd.DataFrame:
    """
    Entrena cada modelo de `modelos` (dict nombre -> estimador sklearn sin ajustar) sobre la
    partición train de df_ml y evalúa en train y test. Pensada para comparar, sobre un mismo
    dataset ya preparado (principal o control), varias técnicas de Machine Learning
    (regularización, Random Forest, Gradient Boosting) frente al modelo de referencia
    de regresión lineal con selección de variables (que se evalúa aparte, con las funciones de
    FuncionesMineria.py, porque necesita el dataset _raw, no el _ml).

    Returns:
        DataFrame con una fila por modelo y columnas de métricas train/test + nº de parámetros.
    """
    x_train, x_test, y_train, y_test = separar_train_test_ml(df_ml)

    filas = []
    for nombre, modelo in modelos.items():
        modelo.fit(x_train, y_train)

        pred_train = predecir(modelo, x_train)
        pred_test = predecir(modelo, x_test)

        m_train = calcular_metricas(y_train, pred_train)
        m_test = calcular_metricas(y_test, pred_test)

        n_params = x_train.shape[1]
        if hasattr(modelo, "n_features_in_") and hasattr(modelo, "coef_"):
            n_params = int(np.sum(np.abs(np.ravel(modelo.coef_)) > 1e-8)) + 1  # + intercepto

        filas.append(
            {
                "dataset": nombre_dataset,
                "modelo": nombre,
                "RMSE_train": m_train["RMSE"],
                "RMSE_test": m_test["RMSE"],
                "MAE_train": m_train["MAE"],
                "MAE_test": m_test["MAE"],
                "R2_train": m_train["R2"],
                "R2_test": m_test["R2"],
                "n_variables_no_nulas": n_params,
            }
        )

    return pd.DataFrame(filas)
