# =============================================
# R2 — RAG Document Assistant Dockerfile (Hugging Face Spaces Compatible)
# =============================================
# Multi-stage build for a lean production image

# Stage 1: Builder — install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Runtime — lean final image
FROM python:3.11-slim

# Hugging Face Spaces require running as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy installed packages from builder
COPY --chown=user --from=builder /install /usr/local

# Copy application code
COPY --chown=user . $HOME/app

# Create directories with appropriate permissions for Hugging Face
RUN mkdir -p uploads chroma_data && chmod -R 777 uploads chroma_data

# Expose the API port (Hugging Face Spaces defaults to 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
