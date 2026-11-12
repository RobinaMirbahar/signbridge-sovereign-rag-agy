#!/usr/bin/env bash
# ==============================================================================
# Clean Up All SignBridge Sovereign RAG Google Cloud Resources
# Prevents ongoing Cloud Run or BigQuery storage charges after the Codelab
# ==============================================================================
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-europe-west1}"
BQ_DATASET="${BQ_DATASET:-signbridge_sovereign_rag}"

echo "Deleting Cloud Run services in ${REGION}..."
for SVC in \
  "${ORCHESTRATOR_SERVICE_NAME:-signbridge-rag-ui}" \
  "${LLM_SERVICE_NAME:-signbridge-gemma4-llm}" \
  "${EMBEDDING_SERVICE_NAME:-signbridge-embeddinggemma}"; do
  echo "  -> Removing ${SVC}..."
  gcloud run services delete "${SVC}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --quiet || true
done

echo "Removing BigQuery dataset ${PROJECT_ID}:${BQ_DATASET}..."
bq rm -r -f -d "${PROJECT_ID}:${BQ_DATASET}" || true

echo "All SignBridge Sovereign RAG resources cleaned up!"
