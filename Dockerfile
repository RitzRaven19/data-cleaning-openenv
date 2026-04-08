# ─────────────────────────────────────────────────────────────────────────────
# Data Cleaning OpenEnv – Docker image
#
# Build:  docker build -t data-cleaning-env .
# Run:    docker run -p 7860:7860 \
#           -e API_BASE_URL="https://router.huggingface.co/v1" \
#           -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
#           -e HF_TOKEN="hf_..." \
#           data-cleaning-env
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY env/       ./env/
COPY app.py     .
COPY inference.py .
COPY openenv.yaml .

# Ownership
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

CMD ["python", "-m", "uvicorn", "app:app", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--ws-ping-interval", "300", "--ws-ping-timeout", "300"]
