#!/usr/bin/env bash
# Deploy SignBridge FastAPI Orchestrator & Side-by-Side Web UI on Cloud Run
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-europe-west1}"
SERVICE_NAME="${ORCHESTRATOR_SERVICE_NAME:-signbridge-rag-ui}"
BQ_DATASET="${BQ_DATASET:-signbridge_sovereign_rag}"
BQ_TABLE="${BQ_TABLE:-sign_lexicon_embeddings}"

EMBEDDING_URL=$(gcloud run services describe "${EMBEDDING_SERVICE_NAME:-signbridge-embeddinggemma}" \
  --project="${PROJECT_ID}" --region="${REGION}" --format="value(status.url)" 2>/dev/null || echo "")
LLM_URL=$(gcloud run services describe "${LLM_SERVICE_NAME:-signbridge-gemma4-llm}" \
  --project="${PROJECT_ID}" --region="${REGION}" --format="value(status.url)" 2>/dev/null || echo "")

cd "$(dirname "$0")/.."

echo "Deploying ${SERVICE_NAME} to Cloud Run (${REGION})..."
gcloud run deploy "${SERVICE_NAME}" \
  --source=. \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --cpu=2 \
  --memory=4Gi \
  --allow-unauthenticated \
  --quiet \
  --set-env-vars="PROJECT_ID=${PROJECT_ID},REGION=${REGION},BQ_DATASET=${BQ_DATASET},BQ_TABLE=${BQ_TABLE},USE_BIGQUERY_VECTOR_SEARCH=true,EMBEDDING_SERVICE_URL=${EMBEDDING_URL},LLM_SERVICE_URL=${LLM_URL}" \
  --port=8080

UI_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" --region="${REGION}" --format="value(status.url)")

echo "SignBridge Side-by-Side Web UI Live URL: ${UI_URL}"
