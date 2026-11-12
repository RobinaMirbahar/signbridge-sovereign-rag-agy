"""
Sovereign EmbeddingGemma 2 Microservice on Google Cloud Run
Transforms private Sign-Language Lexicons & Interpreter Queries into 768-d vectors
inside your private GCP project boundary.
"""
from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

app = FastAPI(
    title="SignBridge Sovereign EmbeddingGemma 2 Service",
    version="1.0.0",
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_ID", "embeddinggemma:latest")


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text or sign gloss chunk to embed")


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    embedding: list[float]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "model": EMBEDDING_MODEL, "sovereign_boundary": "gcp-cloud-run"}


@app.post("/embed", response_model=EmbedResponse)
async def create_embedding(req: EmbedRequest) -> EmbedResponse:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": req.text},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama embedding error: {resp.text}")
        data = resp.json()
        embeddings = data.get("embeddings") or [data.get("embedding", [])]
        vec = embeddings[0]
        return EmbedResponse(model=EMBEDDING_MODEL, dimensions=len(vec), embedding=vec)
