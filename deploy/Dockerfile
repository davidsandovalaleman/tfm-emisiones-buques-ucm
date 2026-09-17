# Estela — servicio de cumplimiento de la flota europea
#
# Imagen minima para publicar la API y el panel web en cualquier plataforma que sepa
# construir un Dockerfile (Hugging Face Spaces, Render, Fly.io, Cloud Run). No incluye TensorFlow:
# la banda XGBoost/red neuronal es opcional y la API ya la declara ausente en /salud en vez de
# fallar, asi que el servicio publico pesa ~400 MB en vez de ~2 GB y arranca en segundos.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY deploy/requirements-servicio.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Solo lo que el servicio necesita para responder. El resto del repositorio -- cuadernos, figuras,
# datos crudos, memoria -- no viaja a la imagen.
COPY src/ ./src/
COPY web/ ./web/
COPY models/modelo_xgb_operacional_tamano.pickle ./models/
COPY data/processed/mrv_consolidado.parquet ./data/processed/
COPY data/processed/mrv_features_operacional_tamano_ml.parquet ./data/processed/
COPY data/processed/mrv_features_operacional_tamano_raw.parquet ./data/processed/
# Tablas de reports/ que la API lee: capa economica (14-18), explicacion servida (04, 07) y pabellon
# (19). Sin ellas el arranque falla o las pestañas «Mi flota», «Por qué emite» y «Pabellón» se quedan
# vacias.
COPY reports/04_niveles_informacion.csv \
     reports/07_niveles_shap.csv \
     reports/07_pendiente_horas_por_tipo.csv \
     reports/14_ahorro_actividad_constante.csv \
     reports/14_persistencia_residuo.csv \
     reports/14_sensibilidad_precios.csv \
     reports/15_palancas_eur_por_dia.csv \
     reports/16_fueleu_pooling.csv \
     reports/16_navieras_exposicion.csv \
     reports/17_cii_trayectoria_bandas.csv \
     reports/18_pasaporte_buques.csv \
     reports/19_pabellon_bloques.csv \
     reports/19_pabellon_mapa.csv \
     ./reports/

EXPOSE 7860
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
