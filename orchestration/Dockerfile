FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    httpx \
    pydantic \
    google-cloud-bigquery

COPY vector_store /app/vector_store
COPY assets /app/assets
COPY orchestration /app/orchestration

EXPOSE 8080
CMD ["uvicorn", "orchestration.app:app", "--host", "0.0.0.0", "--port", "8080"]
