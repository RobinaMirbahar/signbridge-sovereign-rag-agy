"""
SignBridge Sovereign RAG — FastAPI Orchestration & Side-by-Side Comparison API
Compares Ungrounded Open-Weights LLM vs. BigQuery Vector Search + Grounded Sovereign RAG
for Clinical, Educational, and Legal Sign Language (ASL & PSL).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vector_store.init_db import (  # noqa: E402
    build_vector_search_sql,
    deterministic_local_embedding,
    fetch_embedding,
    format_lexicon_chunk,
    load_corpus,
)

app = FastAPI(
    title="SignBridge Sovereign Sign-Language RAG",
    description="Sovereign RAG on Google Cloud Run with Gemma 4, EmbeddingGemma 2 & BigQuery Vector Search",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSETS_DIR = ROOT_DIR / "assets"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Interpreter, clinician, or educator sentence to translate/verify")
    sign_system_filter: str = Field(default="ALL", description="ALL, ASL, or PSL")
    top_k: int = Field(default=2, ge=1, le=5)


class RetrievedSource(BaseModel):
    doc_id: str
    category: str
    sign_system: str
    phrase: str
    verified_gloss: str
    handshape: str
    non_manual_markers: str
    clinical_safety_note: str
    privacy_tier: str
    cosine_similarity: float


class CompareResponse(BaseModel):
    query: str
    execution_mode: str
    ungrounded_output: dict[str, Any]
    sovereign_rag_output: dict[str, Any]
    retrieved_sources: list[RetrievedSource]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return round(dot / (na * nb), 4)


def retrieve_sign_lexicon(query: str, sign_system_filter: str = "ALL", top_k: int = 2) -> tuple[list[RetrievedSource], str]:
    """Queries BigQuery Vector Search in production, or local cosine index in offline sandbox mode."""
    use_bigquery = os.getenv("USE_BIGQUERY_VECTOR_SEARCH", "false").lower() == "true"
    project_id = os.getenv("PROJECT_ID", "")
    dataset_id = os.getenv("BQ_DATASET", "signbridge_sovereign_rag")
    table_id = os.getenv("BQ_TABLE", "sign_lexicon_embeddings")

    if use_bigquery and project_id:
        from google.cloud import bigquery

        q_vec = fetch_embedding(query)
        sql = build_vector_search_sql(project_id, dataset_id, table_id, top_k=top_k)
        client = bigquery.Client(project=project_id)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", q_vec)]
        )
        rows = list(client.query(sql, job_config=job_config).result())
        results = [
            RetrievedSource(
                doc_id=r["doc_id"],
                category=r["category"],
                sign_system=r["sign_system"],
                phrase=r["phrase"],
                verified_gloss=r["verified_gloss"],
                handshape=r["handshape"],
                non_manual_markers=r["non_manual_markers"],
                clinical_safety_note=r["clinical_safety_note"],
                privacy_tier=r["privacy_tier"],
                cosine_similarity=round(1.0 - float(r["distance"]), 4),
            )
            for r in rows
        ]
        return results, "bigquery-vector-search"

    # Offline / local sandbox cosine search using deterministic EmbeddingGemma-compatible dimension
    corpus = load_corpus()
    if sign_system_filter and sign_system_filter.upper() != "ALL":
        filtered = [c for c in corpus if sign_system_filter.upper() in c["sign_system"].upper()]
        corpus = filtered or corpus

    q_vec = deterministic_local_embedding(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in corpus:
        c_vec = deterministic_local_embedding(format_lexicon_chunk(entry))
        sim = _cosine_sim(q_vec, c_vec)
        scored.append((sim, entry))
    scored.sort(key=lambda item: item[0], reverse=True)

    results = [
        RetrievedSource(
            doc_id=item["doc_id"],
            category=item["category"],
            sign_system=item["sign_system"],
            phrase=item["phrase"],
            verified_gloss=item["verified_gloss"],
            handshape=item["handshape"],
            non_manual_markers=item["non_manual_markers"],
            clinical_safety_note=item["clinical_safety_note"],
            privacy_tier=item["privacy_tier"],
            cosine_similarity=sim,
        )
        for sim, item in scored[:top_k]
    ]
    return results, "local-sovereign-sandbox"


async def call_sovereign_llm(prompt: str, system_prompt: str) -> str | None:
    """Calls the open-weights Gemma 4 Ollama service on Cloud Run if configured."""
    llm_url = os.getenv("LLM_SERVICE_URL", "").strip()
    model_id = os.getenv("LLM_MODEL_ID", "gemma4:e4b")
    if not llm_url:
        return None
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{llm_url.rstrip('/')}/api/generate",
            json={
                "model": model_id,
                "system": system_prompt,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


def build_simulated_ungrounded(query: str, top_match: RetrievedSource) -> dict[str, Any]:
    """Demonstrates typical word-for-word English SEE hallucination when no live LLM URL is configured."""
    words = [w.strip(".,?!").upper() for w in query.split() if w.strip(".,?!")]
    naive_gloss = " ".join(words)
    return {
        "gloss": naive_gloss,
        "handshape_guidance": "Generic English-ordered lexical signs (no verified classifier or regional PSL/ASL distinction).",
        "non_manual_markers": "None specified (missing polar brow-raise / WH brow-furrow / mouth morphemes).",
        "risk_warning": top_match.clinical_safety_note,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "signbridge-orchestrator"}


@app.get("/api/corpus")
async def list_corpus() -> dict[str, Any]:
    corpus = load_corpus()
    return {"count": len(corpus), "items": corpus}


@app.post("/api/compare", response_model=CompareResponse)
async def compare_rag_vs_ungrounded(req: CompareRequest) -> CompareResponse:
    sources, mode = retrieve_sign_lexicon(req.query, req.sign_system_filter, req.top_k)
    primary = sources[0]

    ungrounded_sys = (
        "You are an ungrounded general assistant with no access to verified hospital or school sign-language lexicons. "
        "Provide a brief sign-language gloss for the user's sentence."
    )
    context_block = "\n\n".join(
        f"[{s.doc_id} | {s.sign_system}] Phrase: {s.phrase}\n"
        f"Verified Gloss: {s.verified_gloss}\n"
        f"Handshape: {s.handshape}\n"
        f"Non-Manual Markers: {s.non_manual_markers}\n"
        f"Safety Note: {s.clinical_safety_note}"
        for s in sources
    )
    grounded_sys = (
        "You are SignBridge Sovereign RAG, a privacy-preserving Sign Language Interpreter Copilot running inside "
        "a private Google Cloud boundary. Ground your response STRICTLY in the retrieved BigQuery Vector Search "
        "lexicon entries below. Always output: (1) Verified Gloss, (2) Handshape & Spatial Movement, "
        "(3) Non-Manual Facial Grammar [NMM], and (4) Document ID citation.\n\n"
        f"RETRIEVED SOVEREIGN LEXICON:\n{context_block}"
    )

    live_ungrounded = await call_sovereign_llm(req.query, ungrounded_sys)
    live_grounded = await call_sovereign_llm(req.query, grounded_sys)

    if live_ungrounded and live_grounded:
        ungrounded_payload = {
            "gloss": live_ungrounded,
            "handshape_guidance": "Generated without sovereign lexicon retrieval.",
            "non_manual_markers": "Unverified",
            "risk_warning": primary.clinical_safety_note,
        }
        grounded_payload = {
            "gloss": primary.verified_gloss,
            "llm_explanation": live_grounded,
            "handshape_guidance": primary.handshape,
            "non_manual_markers": primary.non_manual_markers,
            "citation": f"{primary.doc_id} ({primary.sign_system} · {primary.privacy_tier})",
        }
        exec_mode = f"{mode}+live-gemma4"
    else:
        ungrounded_payload = build_simulated_ungrounded(req.query, primary)
        grounded_payload = {
            "gloss": primary.verified_gloss,
            "llm_explanation": (
                f"Grounded via [{primary.doc_id}]: Uses natural Topic-Comment spatial syntax rather than "
                f"word-for-word English, preserving critical non-manual facial grammar and avoiding homonym traps."
            ),
            "handshape_guidance": primary.handshape,
            "non_manual_markers": primary.non_manual_markers,
            "citation": f"{primary.doc_id} ({primary.sign_system} · {primary.privacy_tier})",
        }
        exec_mode = mode

    return CompareResponse(
        query=req.query,
        execution_mode=exec_mode,
        ungrounded_output=ungrounded_payload,
        sovereign_rag_output=grounded_payload,
        retrieved_sources=sources,
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
