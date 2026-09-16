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

# Solo lo que el servicio necesita para responder. El resto del repositorio -- cuadernos, informes,
# datos crudos, memoria -- no viaja a la imagen.
COPY src/ ./src/
COPY web/ ./web/
COPY models/modelo_xgb_operacional_tamano.pickle ./models/
COPY data/processed/mrv_consolidado.parquet ./data/processed/
COPY data/processed/mrv_features_operacional_tamano_ml.parquet ./data/processed/
COPY data/processed/mrv_features_operacional_tamano_raw.parquet ./data/processed/
COPY reports/18_pasaporte_buques.csv ./reports/
COPY reports/16_navieras_exposicion.csv ./reports/

EXPOSE 7860
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
