"""
Funciones de deep learning (notebook 05).

Mantiene el mismo contrato que `modelizacion_ml.py` (notebook 04) para que la comparación entre
familias de modelos sea directa:

- misma partición train/test agrupada por buque (columna `es_train` calculada en el notebook 03),
- misma validación cruzada `GroupKFold(3)` por buque para tomar decisiones,
- misma restricción física de no negatividad sobre la predicción (`np.clip(..., 0, None)`),
- mismas métricas (`calcular_metricas`).

Lo específico de deep learning que sí vive aquí:

1. **Escalado**. Una red no es invariante a la escala: hay que estandarizar los predictores y el
   objetivo. El objetivo se estandariza porque el CO₂ está en decenas de miles de toneladas y
   `mse` sobre esa escala produce gradientes iniciales enormes. Los estadísticos de escalado se
   calculan **solo con las filas de entrenamiento de cada fold**, nunca con las de validación.
2. **Validación interna agrupada**. `model.fit(..., validation_split=0.15)` de Keras separa el
   último 15% de las filas *tal cual llegan*, sin agrupar: el mismo buque acabaría a la vez en
   entrenamiento y en la validación que decide el early stopping. Aquí la validación interna se
   construye a mano con `GroupShuffleSplit` por buque, igual que el resto del proyecto.

Las redes se construyen con la API secuencial y funcional de Keras, con `Dropout`,
`kernel_regularizer=l2` y `EarlyStopping` con `restore_best_weights`.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from feature_engineering import RANDOM_STATE
from modelizacion_ml import COLS_NO_PREDICTORAS, TARGET_COL, calcular_metricas

# Variables en la codificación "raw" (sin one-hot), para la variante con embeddings.
VAR_NUM_RAW = ["time_spent_at_sea_hours", "te_valor", "es_subcategoria_nueva"]
VAR_CAT_RAW = ["ship_type_agrupado", "ice_class", "te_metodo", "prop_missings"]

COL_GRUPO = "ship_imo_number"


def fijar_semilla(semilla: int = RANDOM_STATE) -> None:
    """Semilla única para numpy, python y TensorFlow."""
    keras.utils.set_random_seed(int(semilla) % (2**31 - 1))


def predictoras_ml(df_ml: pd.DataFrame) -> list[str]:
    """Columnas predictoras de un dataset ya codificado en one-hot (mismo criterio que el notebook 04)."""
    return [c for c in df_ml.columns if c not in COLS_NO_PREDICTORAS]


class EscaladoObjetivo:
    """
    Estandarización del objetivo con la inversa que devuelve toneladas de CO₂.

    `invertir` aplica además la restricción física de no negatividad, exactamente igual que
    `modelizacion_ml.predecir` hace con los modelos clásicos, para que la comparación entre
    familias sea homogénea.
    """

    def __init__(self, y_fit: np.ndarray):
        self.media = float(np.mean(y_fit))
        self.desv = float(np.std(y_fit))

    def transformar(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype="float64") - self.media) / self.desv

    def invertir(self, y_esc: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(y_esc, dtype="float64").ravel() * self.desv + self.media, 0.0, None)


def particion_interna_agrupada(grupos: np.ndarray, proporcion_val: float = 0.15,
                               semilla: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """
    Índices (ajuste, validación) agrupados por buque para el early stopping.

    Sustituye al `validation_split` de Keras, que parte sin agrupar y dejaría el mismo buque en
    ambos lados.
    """
    n = len(grupos)
    separador = GroupShuffleSplit(n_splits=1, test_size=proporcion_val, random_state=semilla)
    return next(separador.split(np.zeros(n), np.zeros(n), groups=grupos))


def construir_red_densa(n_entradas: int, unidades: tuple[int, ...] = (256, 128, 64),
                        dropout: float | tuple[float, ...] = 0.0, l2: float = 0.0,
                        learning_rate: float = 1e-3) -> keras.Model:
    """
    Red densa de regresión (API secuencial de Keras).

    - Activación `relu` en las capas ocultas y **lineal en la salida**: la salida `relu` satura a
      cero en cuanto un buque cae en la zona negativa y el gradiente deja de circular (medido en la
      sección 2.2 del notebook 05: R² 0,639 frente a 0,817). La no negatividad se impone después, al
      invertir el escalado.
    - `Dropout` y `kernel_regularizer=l2` son las dos técnicas de regularización.
    """
    if np.isscalar(dropout):
        dropout = tuple([float(dropout)] * len(unidades))

    modelo = keras.Sequential(name="red_densa")
    modelo.add(keras.Input(shape=(n_entradas,)))
    for u, d in zip(unidades, dropout):
        modelo.add(layers.Dense(u, activation="relu",
                                kernel_regularizer=keras.regularizers.l2(l2) if l2 else None))
        if d:
            modelo.add(layers.Dropout(d))
    modelo.add(layers.Dense(1, activation="linear"))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                   loss="mse", metrics=["mae"])
    return modelo


def construir_red_embeddings(cardinalidades: dict[str, int], n_numericas: int,
                             unidades: tuple[int, ...] = (256, 128, 64), dropout: float = 0.0,
                             dim_max: int = 8, learning_rate: float = 1e-3) -> keras.Model:
    """
    Variante con *embeddings* para las categóricas (API funcional de Keras).

    En lugar de una columna 0/1 por nivel (one-hot), cada nivel de cada variable categórica se
    representa por un vector denso aprendido durante el entrenamiento. Es la alternativa habitual
    en datos tabulares cuando las categóricas tienen muchos niveles.
    """
    entradas, ramas = [], []

    entrada_num = keras.Input(shape=(n_numericas,), name="numericas")
    entradas.append(entrada_num)
    ramas.append(entrada_num)

    for col, n_niveles in cardinalidades.items():
        entrada = keras.Input(shape=(1,), dtype="int32", name=col)
        entradas.append(entrada)
        dim = int(min(dim_max, (n_niveles + 1) // 2))
        ramas.append(layers.Flatten()(layers.Embedding(n_niveles, dim, name=f"emb_{col}")(entrada)))

    x = layers.Concatenate(name="concatenado")(ramas)
    for u in unidades:
        x = layers.Dense(u, activation="relu")(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    salida = layers.Dense(1, activation="linear")(x)

    modelo = keras.Model(entradas, salida, name="red_embeddings")
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                   loss="mse", metrics=["mae"])
    return modelo


def callbacks_estandar(paciencia: int = 15, reducir_lr: bool = True) -> list[keras.callbacks.Callback]:
    """Early stopping con `restore_best_weights` (+ reducción de learning rate en meseta)."""
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia,
                                         restore_best_weights=True)]
    if reducir_lr:
        cbs.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                     patience=max(3, paciencia // 3), min_lr=1e-5))
    return cbs


def entrenar_red(constructor, x_ajuste, y_ajuste, x_val, y_val, epocas: int = 300,
                 batch_size: int = 256, paciencia: int = 15, reducir_lr: bool = True,
                 semilla: int = RANDOM_STATE, verbose: int = 0):
    """Entrena una red construida por `constructor()` con validación interna explícita."""
    fijar_semilla(semilla)
    modelo = constructor()
    historia = modelo.fit(x_ajuste, y_ajuste, validation_data=(x_val, y_val), epochs=epocas,
                          batch_size=batch_size, callbacks=callbacks_estandar(paciencia, reducir_lr),
                          verbose=verbose)
    return modelo, historia


def cv_agrupada_red(constructor_por_dim, x: np.ndarray, y: np.ndarray, grupos: np.ndarray,
                    n_splits: int = 3, semilla: int = RANDOM_STATE, nombre: str = "red",
                    devolver_predicciones: bool = False, fraccion_train: float = 1.0,
                    **kwargs_entrenamiento):
    """
    Validación cruzada `GroupKFold` por buque para una red, con el mismo protocolo que el notebook 04.

    Dentro de cada fold: se estandarizan predictores y objetivo con el train del fold, se separa
    una validación interna agrupada para el early stopping, y se evalúa sobre el fold retenido en
    toneladas de CO₂ (con la restricción de no negatividad aplicada).

    `constructor_por_dim` recibe el número de columnas de entrada y devuelve un modelo compilado.

    `fraccion_train` < 1 entrena con una submuestra del train de cada fold **agrupada por buque**
    (se sortean buques enteros, no filas sueltas) y evalúa igualmente sobre el fold retenido
    completo. Es lo que necesita la curva de aprendizaje de la sección 9 del notebook 05: la única
    forma honesta de responder a "¿le faltan datos a la red?" es medir cómo cambia el resultado
    al variar la cantidad de datos, manteniendo todo lo demás fijo.
    """
    filas = []
    oof = np.full(len(y), np.nan)
    for i, (idx_tr, idx_va) in enumerate(GroupKFold(n_splits=n_splits).split(x, y, groups=grupos)):
        t0 = time.time()

        if fraccion_train < 1.0:
            conservados, _ = next(GroupShuffleSplit(n_splits=1, train_size=fraccion_train,
                                                    random_state=semilla)
                                  .split(np.zeros(len(idx_tr)), np.zeros(len(idx_tr)),
                                         groups=grupos[idx_tr]))
            idx_tr = idx_tr[conservados]

        escalador_x = StandardScaler().fit(x[idx_tr])
        escalador_y = EscaladoObjetivo(y[idx_tr])

        sub_aj, sub_val = particion_interna_agrupada(grupos[idx_tr], semilla=semilla)
        x_aj = escalador_x.transform(x[idx_tr][sub_aj]).astype("float32")
        x_val = escalador_x.transform(x[idx_tr][sub_val]).astype("float32")
        x_ev = escalador_x.transform(x[idx_va]).astype("float32")

        modelo, historia = entrenar_red(
            lambda: constructor_por_dim(x.shape[1]),
            x_aj, escalador_y.transformar(y[idx_tr][sub_aj]),
            x_val, escalador_y.transformar(y[idx_tr][sub_val]),
            semilla=semilla, **kwargs_entrenamiento,
        )

        pred = escalador_y.invertir(modelo.predict(x_ev, verbose=0))
        oof[idx_va] = pred
        metricas = calcular_metricas(y[idx_va], pred)
        # El tamaño del train solo se anota cuando se está submuestreando (curva de aprendizaje);
        # así las columnas de las validaciones cruzadas normales no cambian.
        tamano = ({} if fraccion_train >= 1.0 else
                  {"n_filas_train": len(idx_tr),
                   "n_buques_train": int(len(np.unique(grupos[idx_tr])))})
        filas.append({"modelo": nombre, "fold": i, **metricas, **tamano,
                      "epocas": len(historia.history["loss"]),
                      "segundos": round(time.time() - t0, 1)})
        keras.backend.clear_session()

    resultados = pd.DataFrame(filas)
    return (resultados, oof) if devolver_predicciones else resultados


def resumen_cv(df_folds: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación típica por modelo de una tabla de folds devuelta por `cv_agrupada_red`."""
    agregado = (df_folds.groupby("modelo", sort=False)
                .agg(R2_CV=("R2", "mean"), desv_R2=("R2", "std"),
                     MAE_CV=("MAE", "mean"), RMSE_CV=("RMSE", "mean"),
                     epocas_medias=("epocas", "mean"), segundos_total=("segundos", "sum"))
                .reset_index())
    return agregado


def codificar_categoricas(df_ajuste: pd.DataFrame, df_aplicar: pd.DataFrame,
                          columnas: list[str]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """
    Codifica las categóricas como enteros para la capa `Embedding`, aprendiendo el diccionario
    **solo con las filas de ajuste**. Los niveles no vistos se mandan a un índice reservado
    (el último), que es lo que hace `Embedding` con un vocabulario ampliado en uno.
    """
    codificado, cardinalidades = {}, {}
    for col in columnas:
        niveles = sorted(df_ajuste[col].astype(str).unique())
        mapa = {v: i for i, v in enumerate(niveles)}
        codificado[col] = (df_aplicar[col].astype(str).map(mapa)
                           .fillna(len(mapa)).astype("int32").values)
        cardinalidades[col] = len(mapa) + 1
    return codificado, cardinalidades


def cv_agrupada_embeddings(df_raw: pd.DataFrame, constructor_por_cardinalidades,
                           n_splits: int = 3, semilla: int = RANDOM_STATE,
                           nombre: str = "red con embeddings",
                           **kwargs_entrenamiento) -> pd.DataFrame:
    """Equivalente a `cv_agrupada_red` para la variante con embeddings (dataset _raw)."""
    df = df_raw.reset_index(drop=True)
    y = df[TARGET_COL].values.astype("float64")
    grupos = df[COL_GRUPO].values

    filas = []
    for i, (idx_tr, idx_va) in enumerate(GroupKFold(n_splits=n_splits).split(df, y, groups=grupos)):
        t0 = time.time()
        escalador_x = StandardScaler().fit(df.loc[idx_tr, VAR_NUM_RAW])
        escalador_y = EscaladoObjetivo(y[idx_tr])
        sub_aj, sub_val = particion_interna_agrupada(grupos[idx_tr], semilla=semilla)
        i_aj, i_val = idx_tr[sub_aj], idx_tr[sub_val]

        def entradas(indices):
            cod, _ = codificar_categoricas(df.loc[i_aj], df.loc[indices], VAR_CAT_RAW)
            return {"numericas": escalador_x.transform(df.loc[indices, VAR_NUM_RAW]).astype("float32"),
                    **cod}

        _, cardinalidades = codificar_categoricas(df.loc[i_aj], df.loc[i_aj], VAR_CAT_RAW)

        modelo, historia = entrenar_red(
            lambda: constructor_por_cardinalidades(cardinalidades, len(VAR_NUM_RAW)),
            entradas(i_aj), escalador_y.transformar(y[i_aj]),
            entradas(i_val), escalador_y.transformar(y[i_val]),
            semilla=semilla, **kwargs_entrenamiento,
        )
        pred = escalador_y.invertir(modelo.predict(entradas(idx_va), verbose=0))
        filas.append({"modelo": nombre, "fold": i, **calcular_metricas(y[idx_va], pred),
                      "epocas": len(historia.history["loss"]),
                      "segundos": round(time.time() - t0, 1)})
        keras.backend.clear_session()

    return pd.DataFrame(filas)


def graficar_historia(historia, titulo: str = "") -> None:
    """
    Curvas de pérdida y MAE de entrenamiento y validación.

    El título es un parámetro para poder comparar varias redes.
    """
    import matplotlib.pyplot as plt

    hist = pd.DataFrame(historia.history)
    hist["epoca"] = historia.epoch
    tiene_mae = "mae" in hist.columns

    fig, ejes = plt.subplots(1, 2 if tiene_mae else 1, figsize=(13, 4.5))
    ejes = np.atleast_1d(ejes)

    ejes[0].plot(hist["epoca"], hist["loss"], label="Entrenamiento")
    ejes[0].plot(hist["epoca"], hist["val_loss"], label="Validación")
    ejes[0].set_xlabel("Época"); ejes[0].set_ylabel("Pérdida (MSE, objetivo estandarizado)")
    ejes[0].legend(); ejes[0].grid(alpha=0.3)

    if tiene_mae:
        ejes[1].plot(hist["epoca"], hist["mae"], label="Entrenamiento")
        ejes[1].plot(hist["epoca"], hist["val_mae"], label="Validación")
        ejes[1].set_xlabel("Época"); ejes[1].set_ylabel("MAE (objetivo estandarizado)")
        ejes[1].legend(); ejes[1].grid(alpha=0.3)

    if titulo:
        fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()
