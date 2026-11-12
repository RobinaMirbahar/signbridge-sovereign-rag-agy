FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install --no-cache-dir --break-system-packages fastapi uvicorn[standard] httpx pydantic

ENV EMBEDDING_MODEL_ID=embeddinggemma:latest
ENV OLLAMA_MODELS=/models

# Pre-bake EmbeddingGemma 2 weights into container image for zero external runtime calls!
RUN ollama serve & \
    SERVER_PID=$! && \
    sleep 5 && \
    ollama pull ${EMBEDDING_MODEL_ID} && \
    kill $SERVER_PID

COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
