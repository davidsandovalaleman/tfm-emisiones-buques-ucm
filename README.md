# TFM — Predicción de emisiones de CO₂ de buques

Trabajo Fin de Máster — UCM Online, Big Data, Data Science e Inteligencia Artificial.
Autor: David Sandoval.

Sistema de aprendizaje automático que predice las emisiones anuales de CO₂ de un buque a partir de
datos públicos, explica qué factores las determinan y permite simular escenarios operativos de
reducción sobre la flota del Espacio Económico Europeo.

## Objetivos

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

## La capa regulatoria

Tres módulos que no tocan el modelo: se construyen encima de sus predicciones y traducen toneladas a
la unidad en la que decide el sector. Los tres se apoyan en la **capacidad de carga reconstruida** en
el notebook 03, que es el dato que el registro no publica y que las tres normas necesitan.

| Módulo | Qué calcula 
|---|---|---|
| `src/cii.py` | Calificación **CII A-E** por buque-año (MEPC.353/354(78), factores de MEPC.400(83)) 
| `src/ets.py` | Ámbito, base imponible y coste del **EU ETS** marítimo 
| `src/fueleu.py` | Intensidad GEI de pozo a estela, balance y multa de **FuelEU** 

## La API (productivización)

`src/api.py` sirve el simulador por HTTP con **FastAPI**. No recalcula nada: importa
`src/simulador.py`, el mismo módulo que ejecuta el notebook 08, así que la cifra que devuelve un
endpoint y la que aparece en la memoria salen de la misma línea de código.

```bash
# El entorno del proyecto es `tfm` (conda, Python 3.11.15).
# Si no está creado:  conda create -n tfm python=3.11 -y
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

## A tener en cuenta

1. **Informes `Full` y `Partial`.** Un buque que cambia de compañía a mitad de año aparece dos veces,
   y el CO₂ del `Partial` **ya está incluido** en el `Full`. El dataset de modelización se filtra a
   `report_coverage == 'Full'` para no contar dos veces la misma emisión (evidencia en la sección 6
   del notebook 01).
2. **Cinco niveles de información.** Mismo algoritmo y mismos
   hiperparámetros en los cinco: lo único que cambia es qué se sabe del buque, de modo que la
   comparación mide lo que vale cada nivel de información y no lo que vale cada técnica.
   El nivel *principal* (buque + tiempo, R² 0,8337) es el de la **comparación de técnicas**, porque
   es donde se decidió el algoritmo y donde se midió el deep learning. Los niveles *tamaño*
   (+ capacidad de carga, 0,8999) y *operacional* (+ velocidad, 0,9164) añaden una variable
   reconstruida cada uno, y **operacional+tamaño** (0,9484) las dos: es el **mejor modelo del
   trabajo** y el que sirven el simulador (`notebooks/08`) y la API. El quinto, el de
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

**https://github.com/davidsandovalaleman/tfm-emisiones-buques-ucm** 

## Entrega

La memoria está en `docs/memoria/Memoria_TFM_David_Sandoval_v1.docx`.
Cada anexo remite a los cuadernos y ficheros que lo respaldan.
