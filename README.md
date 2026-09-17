# TFM — Predicción de emisiones de CO₂ de buques

Trabajo Fin de Máster — UCM Online, Big Data, Data Science e Inteligencia Artificial.
Autor: David Sandoval.

Sistema de aprendizaje automático que predice las emisiones anuales de CO₂ de un buque a partir de
datos públicos, explica qué factores las determinan y permite simular escenarios operativos de
reducción sobre la flota del Espacio Económico Europeo.

## Objetivos, y dónde se resuelve cada uno

| Objetivo | Dónde | Resultado principal |
|---|---|---|
| **1. Predecir** | `notebooks/04`, `05` | XGBoost, **R² 0,8337** en test con lo que el registro publica de cualquier buque (MAE 2.686 t, un 70% menos de error que predecir la media). Reconstruyendo del propio registro el tamaño del buque sube a **0,8999**, y con la velocidad además a **0,9484** |
| **2. Explicar** | `notebooks/07` | Tipo de buque y horas en el mar suman ≈80% de la explicación. **Cada hora de navegación cuesta entre 3,02 t (carga general) y 6,48 t (metanero)** de CO₂. Al medir velocidad y tamaño, **el tipo de buque cae del 41,6% al 6,6%**: era el sustituto de tres magnitudes medibles |
| **3. Simular** | `notebooks/08`, `src/simulador.py` | Un **10% menos de millas ahorraría 13,75 Mt de CO₂/año**. Focalizar el recorte en los cuatro tipos más intensivos **duplica el ahorro a igualdad de esfuerzo** (+99,3%) |
| **4. Valorar** | `src/cii.py`, `src/ets.py`, `src/fueleu.py` | Capa regulatoria sobre los tres anteriores: calificación **CII A-E** de 18.725 buques, coste del **EU ETS** y multa de **FuelEU**. Ese mismo recorte del 10% vale **767,7 M€/año** de cumplimiento evitado, y la flota del EEE afronta **8.737 millones de €** por sus emisiones de 2025: **58 € por tonelada** |
| **5. Probar** | `scripts/backtest_temporal.py` | La validación que un profesional acepta: entrenando **solo con datos hasta 2024**, el sistema estima la factura de derechos de 2025 con un **+4,58%** de error (R² 0,967 sobre los buques ya presentes), y la banda calibrada al 90% cubre el **89,9%** |

## Datos

**EMSA THETIS-MRV**, informe público de emisiones del Reglamento (UE) 2015/757:
<https://mrv.emsa.europa.eu/#public/emission-report>

Ocho ejercicios (2018-2025), **106.521 buques-año** y **1.110,7 Mt de CO₂** tras la consolidación.
La publicación es obligatoria para la Comisión (art. 21.1) y la reutilización está autorizada
citando la fuente (apartado 2 de la memoria).

## Estructura

```
data/raw/          Informes anuales .xlsx descargados de THETIS-MRV (sin modificar)
data/processed/    Datasets consolidados y de modelización (.parquet)
notebooks/         El trabajo, en orden de ejecución
src/               Código reutilizable: ingesta, features, modelización, simulador, capa regulatoria, API
                   (el cuaderno 04 usa además src/FuncionesMineria.py, funciones auxiliares de regresión)
web/               Panel web del simulador (un solo fichero, sin dependencias)
models/            Modelos entrenados y sus preprocesos
reports/           Figuras y tablas generadas para la memoria
docs/              Memoria del trabajo, con sus anexos
deploy/            Dockerfile y dependencias del servicio público
video/             Vídeo de presentación del trabajo
```

El repositorio se versiona entero **salvo** `data/raw/` (44 MB de Excel descargables de EMSA),
los artefactos regenerables (`mrv.sqlite`, `pasaportes.jsonl`, `reports/tableau/`) y el vídeo. Los
modelos y los datos procesados **sí** van dentro: clonar y ejecutar tiene que funcionar sin descargar
nada.

## Los notebooks, en orden

| # | Notebook | Qué hace |
|---|---|---|
| 01 | `01_ingesta_consolidacion` | Consolida los ocho ejercicios y resuelve la taxonomía cambiante de EMSA |
| 02 | `02_eda_estadistico` | Descriptivo, atípicos, *missings*, correlaciones |
| 03 | `03_feature_engineering` | Variables derivadas y **reconstrucción de la distancia, la velocidad y la capacidad de carga** —el tamaño del buque—, que el MRV no publica pero sí implica. Cobertura: 98,2% de las filas y 24.415 buques |
| 04 | `04_modelizacion_ml` | Lineal con selección, Ridge/Lasso, RF, GB, XGBoost, SVM y *stacking*, comparados en CV agrupada por buque |
| 05 | `05_deep_learning` | Cuatro arquitecturas de red con el mismo protocolo. **El deep learning no gana a XGBoost en datos tabulares**, pero combinarlos sí aporta |
| 07 | `07_interpretabilidad` | SHAP contrastado contra `total_gain` y permutación |
| 08 | `08_simulador_escenarios` | Simulador de escenarios sobre el motor de `src/simulador.py` |
| 09 | `09_bbdd_sql` | El consolidado como base de datos SQL. **Las restricciones del esquema encontraron 19 registros físicamente imposibles y un error de granularidad** que ocho notebooks de pandas no habían visto |
| 10 | `10_bbdd_nosql` | El pasaporte de cumplimiento como documento. El registro es relacional; **el producto que se entrega no lo es**, y `$unwind` sobre la trayectoria CII responde en una línea lo que la tabla ancha no sabe preguntar |
| 11 | `11_big_data_spark` | Spark contra pandas sobre la misma consulta. **A la escala real pierde 12,9 veces**, y el punto de cruce se mide, no se supone: ≈19 M de filas, 177 veces el registro completo |

*(El 06, RNN de series temporales por buque, se descarta con su medida: ocho pasos temporales por
buque y el 21,6% de la flota con un solo ejercicio. El descarte razonado está en
`notebooks/06_LEEME_rnn_descartada.md` y en la memoria, junto con los de NLP —el campo de texto
libre está relleno en el 0,1% de las filas— y modelos generativos.)*

El 09 es independiente del resto: no alimenta a ningún modelo, y se puede ejecutar en cualquier
momento después del 01. Genera `data/processed/mrv.sqlite` (24 MB, no versionado).

## La capa regulatoria

Tres módulos que no tocan el modelo: se construyen encima de sus predicciones y traducen toneladas a
la unidad en la que decide el sector. Los tres se apoyan en la **capacidad de carga reconstruida** en
el notebook 03, que es el dato que el registro no publica y que las tres normas necesitan.

| Módulo | Qué calcula | Resultado |
|---|---|---|
| `src/cii.py` | Calificación **CII A-E** por buque-año (MEPC.353/354(78), factores de MEPC.400(83)) | **18.725 buques** calificados desde datos públicos. La norma suspende al **53% de los buques pequeños y al 23% de los grandes**, y el residuo del modelo dice que los pequeños no emiten de más |
| `src/ets.py` | Ámbito, base imponible y coste del **EU ETS** marítimo | La flota pagó **5.068 M€** por 2025 y pagará **7.462 M€** bajo el régimen pleno de 2026. Solo el **57%** del CO₂ evitado en un escenario se convierte en derechos |
| `src/fueleu.py` | Intensidad GEI de pozo a estela, balance y multa de **FuelEU** | El tipo de combustible se deduce del cociente CO₂/combustible. La intensidad mediana reconstruida es **91,24 gCO₂eq/MJ** frente al **91,16** de referencia del reglamento: **0,09% de diferencia**. Solo el **4,4%** de la flota cumple |

**Las tres normas premian cosas distintas**, y el trabajo lo mide: el ETS grava emisiones absolutas,
así que recortar millas ataca su factura de frente; FuelEU grava *intensidad*, y recortar millas solo
baja la multa en proporción a la energía, sin mejorar la calificación del buque. El mismo escenario
del −10% vale 666,7 M€ en ETS y solo 114,8 M€ en FuelEU.

Todas las cifras de coste llevan su precio con fecha y fuente: el derecho de emisión es una
cotización de mercado, no un resultado del trabajo. Los tres módulos declaran sus limitaciones en la
cabecera y la API las devuelve en cada respuesta —la multa de FuelEU, por ejemplo, es una **cota
superior**, porque los biocombustibles certificados no son distinguibles en el registro público—.

## La API (productivización)

`src/api.py` sirve el simulador por HTTP con **FastAPI**. No recalcula nada: importa
`src/simulador.py`, el mismo módulo que ejecuta el notebook 08, así que la cifra que devuelve un
endpoint y la que aparece en la memoria salen de la misma línea de código.

```bash
# El entorno del proyecto es `tfm` (conda, Python 3.11.15).
# Si no lo tienes creado:  conda create -n tfm python=3.11 -y
conda activate tfm
pip install -r requirements.txt

# `python -m` usa el interprete activo; `uvicorn` a secas usa el primero del PATH.
python -m uvicorn src.api:app --reload
# http://127.0.0.1:8000/        <- el panel
# http://127.0.0.1:8000/docs    <- la API
```

Se arranca **desde la raíz del repositorio** y con `python -m uvicorn`, no con `uvicorn` a secas: el
ejecutable suelto usa el primer Python del PATH, que rara vez es el del entorno del proyecto, y el
síntoma es un `ModuleNotFoundError: No module named 'fastapi'` que parece un fallo del código y no lo
es.

| Ruta | Qué responde |
|---|---|
| `GET /` | **panel web**: el simulador con deslizadores, cifras y gráficos, en dos pestañas |
| `GET /salud`, `GET /metadatos` | estado del servicio, métricas de test, valores admitidos y límites del modelo |
| `GET /ets/parametros`, `GET /fueleu/parametros` | reglas vigentes de cada régimen, precio de referencia del derecho y limitaciones declaradas de la estimación |
| `POST /predecir` | cuánto CO₂ emite un buque con esas características |
| `POST /simular/buque` | efecto de uno o varios escenarios sobre un buque concreto (por IMO y año, o descrito a mano) |
| `POST /simular/flota` | ahorro agregado de un escenario sobre la flota, con banda de incertidumbre |
| `POST /simular/focalizado` | mismas horas de navegación evitadas, repartidas o concentradas: cuánto cambia |

Los tres endpoints de simulación aceptan `anio_regimen` y `precio_eua`, y devuelven junto al ahorro
en toneladas dos bloques `coste_ets` y `coste_fueleu` con el desglose completo —qué parte del ahorro
cae dentro del ámbito, qué fracción hay que entregar ese año y a qué precio—, más su suma en
`cumplimiento_evitado_millones_eur`. La cifra en euros nunca viaja sola.

Ejemplo, con un buque real de la partición de **test** —el modelo no lo vio entrenando, así que la
predicción no puede atribuirse a memorización—:

```bash
curl -X POST localhost:8000/simular/buque -H 'Content-Type: application/json' \
  -d '{"imo":"9351488","anio":2025,"escenarios":[{"delta_distancia":-0.10}]}'
# CRUISE BARCELONA (ro-pax, 2025): observado 100.073 t, predicho 99.584 t
# con un 10% menos de millas -> 82.916 t, ahorro de 16.668 t (-16,7%)
# y en dinero: 1.017.766 EUR de derechos del ETS + 221.098 EUR de multa de FuelEU
```

Ninguna respuesta sale sin su **aviso de fiabilidad**: el error relativo típico del tramo de emisión
en el que cae la predicción, medido sobre la partición de test. El modelo es proporcionalmente menos
preciso con los buques pequeños, y la respuesta lo dice en vez de dejar que el lector lo suponga.

## Tres cosas que hay que saber antes de tocar nada

1. **Informes `Full` y `Partial`.** Un buque que cambia de compañía a mitad de año aparece dos veces,
   y el CO₂ del `Partial` **ya está incluido** en el `Full`. El dataset de modelización se filtra a
   `report_coverage == 'Full'` para no contar dos veces la misma emisión (evidencia en la sección 6
   del notebook 01).
2. **Cinco niveles de información, y no son intercambiables.** Mismo algoritmo y mismos
   hiperparámetros en los cinco: lo único que cambia es qué se sabe del buque, de modo que la
   comparación mide lo que vale cada pieza de información y no lo que vale cada técnica.
   El nivel *principal* (buque + tiempo, R² 0,8337) es el de la **comparación de técnicas**, porque
   es donde se decidió el algoritmo y donde se midió el deep learning. Los niveles *tamaño*
   (+ capacidad de carga, 0,8999) y *operacional* (+ velocidad, 0,9164) añaden una variable
   reconstruida cada uno, y **operacional+tamaño** (0,9484) las dos: es el **mejor modelo del
   trabajo** y el que sirven el simulador (`notebooks/08`) y la API. Los cuatro cumplen el mismo
   criterio —sus predictores están disponibles para cualquier buque del registro—. El quinto, el de
   *control*, incluye el combustible consumido y sirve **solo como cota superior**: predecir CO₂ a
   partir del combustible declarado es casi una tautología (R² 0,9948).
3. **La partición es por buque, no por fila.** Un mismo buque no puede estar a la vez en train y en
   test: aparece hasta ocho veces en el dataset y separarlo por filas inflaría el resultado. La
   columna `es_train` viene ya calculada en los `.parquet` para que todos los notebooks usen
   exactamente la misma.

## Reproducir el proyecto

```bash
pip install -r requirements.txt
```

Las versiones van fijadas con `==` porque varios resultados dependen del comportamiento concreto
de una versión de librería. Reejecutado en otra máquina, el proyecto regenera **11 de 12 artefactos
byte a byte**; las versiones usadas también quedan grabadas en los metadatos de los propios
artefactos. Para `requirements-lock.txt` y la creación del entorno, ver la cabecera de
`requirements.txt`. Para reejecutar un notebook sin abrir Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/07_interpretabilidad.ipynb
```

## Repositorio

**https://github.com/davidsandovalaleman/tfm-emisiones-buques-ucm** (privado).

Historial en **commits temáticos** que siguen el orden real de trabajo —ingesta, EDA, features, ML,
deep learning, interpretabilidad, simulador, SQL, API, capacidad de carga y documentación—, con la
convención *Conventional Commits* y un cuerpo que explica el *porqué* de cada fase.

```bash
git log --oneline --reverse     # el proyecto, leído como historia
```

## Entrega

La memoria, con los anexos A-J, está en `docs/memoria/Memoria_TFM_David_Sandoval_v1.docx`.
Cada anexo remite a los cuadernos y ficheros de este repositorio que lo respaldan.
