FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY vision ./vision
COPY qa ./qa
COPY guide ./guide
COPY tracking ./tracking
COPY jobs ./jobs
COPY config.yaml .
ENV MODEL_DIR=/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS -H "Authorization: Bearer ${AI_SERVICE_TOKEN}" http://127.0.0.1:8000/api/v1/ai/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
