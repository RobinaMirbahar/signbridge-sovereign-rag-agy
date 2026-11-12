"""
BigQuery Vector Search Initialization & Sign-Language Lexicon Ingestion
Stack: EmbeddingGemma 2 + BigQuery VECTOR_SEARCH (COSINE distance)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import httpx

CORPUS_PATH = Path(__file__).parent / "sign_lexicon_corpus.json"
EMBEDDING_DIM = 768


def deterministic_local_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Offline deterministic unit-norm fallback vector for local testing without Cloud Run."""
    tokens = [tok.strip(".,?!:;()[]-\"'").lower() for tok in text.split() if tok.strip()]
    vec = [0.0] * dim
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        for i in range(0, 16, 2):
            idx = ((digest[i] << 8) | digest[i + 1]) % dim
            sign = 1.0 if (digest[(i + 2) % 32] % 2 == 0) else -1.0
            vec[idx] += sign * (1.0 + (digest[(i + 3) % 32] / 255.0))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def fetch_embedding(text: str, embedding_service_url: str | None = None) -> list[float]:
    """Calls the sovereign EmbeddingGemma 2 Cloud Run service, or falls back locally in test mode."""
    url = embedding_service_url or os.getenv("EMBEDDING_SERVICE_URL", "")
    if not url:
        return deterministic_local_embedding(text)
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{url.rstrip('/')}/embed", json={"text": text})
        resp.raise_for_status()
        return resp.json()["embedding"]


def format_lexicon_chunk(entry: dict[str, Any]) -> str:
    """Builds a rich semantic document chunk combining phrase, gloss, handshape, and NMM."""
    return (
        f"[{entry['doc_id']} | {entry['sign_system']} | {entry['category']}]\n"
        f"Spoken/Written Input: {entry['phrase']}\n"
        f"Verified Sign Gloss: {entry['verified_gloss']}\n"
        f"Handshape Parameters: {entry['handshape']}\n"
        f"Spatial Location & Movement: {entry['location_movement']}\n"
        f"Non-Manual Facial Grammar (NMM): {entry['non_manual_markers']}\n"
        f"Clinical/Educational Safety Rule: {entry['clinical_safety_note']}"
    )


def build_bigquery_ddl(project_id: str, dataset_id: str, table_id: str) -> str:
    """Returns the sovereign BigQuery DDL statement for storing SignBridge vector embeddings."""
    return f"""
CREATE SCHEMA IF NOT EXISTS `{project_id}.{dataset_id}`;

CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{table_id}` (
  doc_id STRING NOT NULL,
  category STRING,
  sign_system STRING,
  phrase STRING,
  verified_gloss STRING,
  handshape STRING,
  location_movement STRING,
  non_manual_markers STRING,
  clinical_safety_note STRING,
  privacy_tier STRING,
  chunk_text STRING,
  embedding ARRAY<FLOAT64>
);
""".strip()


def build_vector_search_sql(project_id: str, dataset_id: str, table_id: str, top_k: int = 3) -> str:
    """Parameterized BigQuery VECTOR_SEARCH query using COSINE distance."""
    return f"""
SELECT
  base.doc_id,
  base.category,
  base.sign_system,
  base.phrase,
  base.verified_gloss,
  base.handshape,
  base.location_movement,
  base.non_manual_markers,
  base.clinical_safety_note,
  base.privacy_tier,
  base.chunk_text,
  distance
FROM VECTOR_SEARCH(
  TABLE `{project_id}.{dataset_id}.{table_id}`,
  'embedding',
  (SELECT @query_embedding AS embedding),
  top_k => {top_k},
  distance_type => 'COSINE'
)
ORDER BY distance ASC
""".strip()


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def init_bigquery_vector_store() -> None:
    """Creates the BigQuery dataset/table and ingests embedded SignBridge lexicon documents."""
    from google.cloud import bigquery  # Lazy import so offline unit tests run cleanly

    project_id = os.environ["PROJECT_ID"]
    dataset_id = os.getenv("BQ_DATASET", "signbridge_sovereign_rag")
    table_id = os.getenv("BQ_TABLE", "sign_lexicon_embeddings")
    region = os.getenv("REGION", "europe-west1")
    embedding_url = os.getenv("EMBEDDING_SERVICE_URL", "")

    client = bigquery.Client(project=project_id, location=region)
    ddl = build_bigquery_ddl(project_id, dataset_id, table_id)
    for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
        client.query(stmt).result()

    corpus = load_corpus()
    rows_to_insert = []
    for item in corpus:
        chunk_text = format_lexicon_chunk(item)
        vec = fetch_embedding(chunk_text, embedding_url)
        rows_to_insert.append({**item, "chunk_text": chunk_text, "embedding": vec})

    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    client.query(f"TRUNCATE TABLE `{table_ref}`").result()
    errors = client.insert_rows_json(table_ref, rows_to_insert)
    if errors:
        raise RuntimeError(f"BigQuery ingestion failed: {errors}")
    print(f"Successfully ingested {len(rows_to_insert)} verified sign-language records into {table_ref}!")


if __name__ == "__main__":
    init_bigquery_vector_store()
