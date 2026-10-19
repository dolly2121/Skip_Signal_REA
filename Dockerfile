FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY config/ config/
COPY web/ web/
COPY artifacts/serving/ artifacts/serving/

CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT}