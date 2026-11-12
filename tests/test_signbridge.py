"""Automated Verification Suite for SignBridge Sovereign Sign-Language RAG."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.app import app  # noqa: E402
from vector_store.init_db import (  # noqa: E402
    EMBEDDING_DIM,
    build_bigquery_ddl,
    build_vector_search_sql,
    deterministic_local_embedding,
    load_corpus,
)

client = TestClient(app)


def test_corpus_integrity() -> None:
    corpus = load_corpus()
    assert len(corpus) >= 8
    required_keys = {
        "doc_id",
        "category",
        "sign_system",
        "phrase",
        "verified_gloss",
        "handshape",
        "location_movement",
        "non_manual_markers",
        "clinical_safety_note",
        "privacy_tier",
    }
    for entry in corpus:
        assert required_keys.issubset(entry.keys())


def test_embedding_dimension_and_normalization() -> None:
    vec = deterministic_local_embedding("Are you allergic to penicillin?")
    assert len(vec) == EMBEDDING_DIM
    norm_sq = sum(x * x for x in vec)
    assert abs(norm_sq - 1.0) < 1e-3


def test_bigquery_vector_search_sql_generation() -> None:
    ddl = build_bigquery_ddl("demo-proj", "signbridge_sovereign_rag", "sign_lexicon_embeddings")
    assert "ARRAY<FLOAT64>" in ddl
    sql = build_vector_search_sql("demo-proj", "signbridge_sovereign_rag", "sign_lexicon_embeddings", top_k=3)
    assert "VECTOR_SEARCH" in sql
    assert "distance_type => 'COSINE'" in sql


def test_side_by_side_compare_endpoint_fasting_homonym() -> None:
    resp = client.post(
        "/api/compare",
        json={
            "query": "Please fast for 8 hours before your blood sugar test tomorrow morning.",
            "sign_system_filter": "PSL",
            "top_k": 2,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "EAT-NOT" in payload["sovereign_rag_output"]["gloss"]
    assert payload["retrieved_sources"][0]["doc_id"] == "SL-PSL-003"


def test_newsroom_and_corporate_embargo_endpoints() -> None:
    news_resp = client.post(
        "/api/compare",
        json={
            "query": "BREAKING NEWS: Flash flood evacuation ordered for riverfront districts—move to high ground immediately!",
            "sign_system_filter": "ALL",
            "top_k": 2,
        },
    )
    assert news_resp.status_code == 200
    news_data = news_resp.json()
    assert news_data["retrieved_sources"][0]["doc_id"] == "SL-NEWS-001"
    assert "EVACUATE" in news_data["sovereign_rag_output"]["gloss"]

    corp_resp = client.post(
        "/api/compare",
        json={
            "query": "Confidential All-Hands: Q3 cloud revenue grew 28% year-over-year, and there will be zero layoffs during the merger.",
            "sign_system_filter": "ASL",
            "top_k": 2,
        },
    )
    assert corp_resp.status_code == 200
    corp_data = corp_resp.json()
    assert corp_data["retrieved_sources"][0]["doc_id"] == "SL-CORP-002"
    assert "LAYOFF ZERO-NONE" in corp_data["sovereign_rag_output"]["gloss"]


def test_health_and_ui_index() -> None:
    assert client.get("/health").status_code == 200
    ui = client.get("/")
    assert ui.status_code == 200
    assert "SignBridge Sovereign RAG" in ui.text
