FROM ollama/ollama:latest

ENV LLM_MODEL_ID=gemma4:e4b
ENV OLLAMA_MODELS=/models
ENV OLLAMA_HOST=0.0.0.0:8080
ENV OLLAMA_KEEP_ALIVE=-1

# Pre-pull open-weights Gemma 4 model into container image for zero external egress at runtime
RUN ollama serve & \
    SERVER_PID=$! && \
    sleep 5 && \
    ollama pull ${LLM_MODEL_ID} && \
    kill $SERVER_PID

EXPOSE 8080
ENTRYPOINT ["ollama", "serve"]
