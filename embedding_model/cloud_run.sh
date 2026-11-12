#!/usr/bin/env bash
# Deploy Sovereign EmbeddingGemma 2 Service to Google Cloud Run
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-europe-west1}"
SERVICE_NAME="${EMBEDDING_SERVICE_NAME:-signbridge-embeddinggemma}"

echo "Deploying ${SERVICE_NAME} to Cloud Run (${REGION})..."
gcloud run deploy "${SERVICE_NAME}" \
  --source=. \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --cpu=4 \
  --memory=8Gi \
  --no-cpu-throttling \
  --no-allow-unauthenticated \
  --port=8080

EMBEDDING_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format="value(status.url)")

echo "EmbeddingGemma 2 Service URL: ${EMBEDDING_URL}"
