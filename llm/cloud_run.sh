#!/usr/bin/env bash
# ==============================================================================
# Deploy Sovereign Open-Weights Gemma 4 LLM on Google Cloud Run
# Automatically inspects NVIDIA L4 GPU quota and falls back to High-CPU tier
# ==============================================================================
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-europe-west1}"
SERVICE_NAME="${LLM_SERVICE_NAME:-signbridge-gemma4-llm}"

echo "Checking NVIDIA L4 GPU quota in ${REGION}..."
L4_QUOTA=$(gcloud compute regions describe "${REGION}" \
  --project="${PROJECT_ID}" \
  --format="json(quotas)" | python3 -c "
import json, sys
data = json.load(sys.stdin)
quotas = data.get('quotas', [])
l4 = [q.get('limit', 0) for q in quotas if 'NVIDIA_L4' in q.get('metric', '')]
print(int(max(l4)) if l4 else 0)
" 2>/dev/null || echo "0")

if [[ "${L4_QUOTA}" -ge 1 ]]; then
  echo "Detected NVIDIA L4 GPU quota (${L4_QUOTA}). Deploying with Cloud Run L4 GPU acceleration..."
  gcloud run deploy "${SERVICE_NAME}" \
    --source=. \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --no-gpu-zonal-redundancy \
    --cpu=4 \
    --memory=16Gi \
    --no-cpu-throttling \
    --no-allow-unauthenticated \
    --port=8080
else
  echo "No NVIDIA L4 GPU quota in ${REGION} (limit=${L4_QUOTA}). Deploying with Sovereign High-CPU Fallback (8 vCPU / 16Gi)..."
  gcloud run deploy "${SERVICE_NAME}" \
    --source=. \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --cpu=8 \
    --memory=16Gi \
    --no-cpu-throttling \
    --no-allow-unauthenticated \
    --port=8080
fi

LLM_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format="value(status.url)")

echo "Sovereign Gemma 4 LLM Service URL: ${LLM_URL}"
