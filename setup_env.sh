#!/usr/bin/env bash
# ==============================================================================
# SignBridge Sovereign RAG — Environment & GCP Project Initialization
# Stack: Gemma 4 + EmbeddingGemma 2 + BigQuery Vector Search + Cloud Run + AGY
# ==============================================================================
set -euo pipefail

export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION="${REGION:-europe-west1}"
export BQ_DATASET="${BQ_DATASET:-signbridge_sovereign_rag}"
export BQ_TABLE="${BQ_TABLE:-sign_lexicon_embeddings}"

export EMBEDDING_SERVICE_NAME="${EMBEDDING_SERVICE_NAME:-signbridge-embeddinggemma}"
export LLM_SERVICE_NAME="${LLM_SERVICE_NAME:-signbridge-gemma4-llm}"
export ORCHESTRATOR_SERVICE_NAME="${ORCHESTRATOR_SERVICE_NAME:-signbridge-rag-ui}"

export EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-embeddinggemma:latest}"
export LLM_MODEL_ID="${LLM_MODEL_ID:-gemma4:e4b}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: PROJECT_ID is not set. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

echo "Configuring Google Cloud project: ${PROJECT_ID} (region: ${REGION})"
gcloud config set project "${PROJECT_ID}"

echo "Enabling required Google Cloud APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  compute.googleapis.com \
  iam.googleapis.com \
  --project="${PROJECT_ID}"

echo "Environment initialized successfully!"
echo "  PROJECT_ID : ${PROJECT_ID}"
echo "  REGION     : ${REGION}"
echo "  BQ_DATASET : ${BQ_DATASET}.${BQ_TABLE}"
