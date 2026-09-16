# No hay cuaderno 06 — y esa es la conclusión, no un hueco

La numeración salta de `05_deep_learning.ipynb` a `07_interpretabilidad.ipynb` porque el cuaderno 06
iba a ser una **red recurrente sobre la serie temporal de cada buque** y esa fase se **descartó con su
medida**. El hueco se deja a propósito: renumerar habría borrado del índice una decisión que forma
parte del resultado.

## Por qué no se hizo

| Lo que necesita una RNN | Lo que da el registro |
|---|---|
| Series largas por unidad | **Ocho pasos** como máximo (2018-2025), uno por ejercicio anual |
| Series completas | El **21,6%** de la flota aparece con **un solo ejercicio** |
| Frecuencia intra-anual | El MRV publica **un agregado por año**: no hay estacionalidad que aprender |

Con ocho puntos y sin estructura intra-anual, una red recurrente no tiene de dónde aprender
dependencia temporal: reproduciría, con muchos más parámetros y sin capacidad de defensa, lo que el
modelo tabular ya hace mejor.

## Qué se hizo en su lugar

- **El descarte, documentado con su cifra**, en la memoria.
- **La dimensión temporal, atacada por donde sí da resultado**: `scripts/backtest_temporal.py`
  entrena con datos hasta 2024 y predice 2025 sin haberlo visto. Es la pregunta que una RNN habría
  intentado responder — «¿aciertas el año que viene?» — contestada con la técnica que el dato
  admite, y contestada en euros: **+4,58%** de error sobre la factura de derechos
  (`reports/21_backtest_factura.csv`).

## Líneas futuras

La RNN vuelve a tener sentido con datos AIS de posición, que es la escala que mide el cuaderno 11
(≈19 M de filas, 177 veces el registro MRV completo). Se desarrolla en el apartado 9 de la memoria.
