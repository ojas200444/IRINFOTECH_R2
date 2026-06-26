# =============================================
# R2 — RAG Document Assistant Dockerfile
# =============================================
# Multi-stage build for a lean production image

# Stage 1: Builder — install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY R2/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Runtime — lean final image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and shared utilities
COPY R2 /app
COPY needful /app/needful

# Create directories for uploads and ChromaDB data
RUN mkdir -p uploads chroma_data

# Expose the API port
EXPOSE 8000

# Health check — ensures the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

