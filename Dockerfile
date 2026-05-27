# Stage 1: Builder
FROM python:3.12-slim-bookworm@sha256:73dbd1a2ad74137451593a8ac30f7bdd0f5010fc05fb34644190cffa7696bbf3 AS builder

# Set working directory for build
WORKDIR /build

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies to /install prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime  
FROM python:3.12-slim-bookworm@sha256:73dbd1a2ad74137451593a8ac30f7bdd0f5010fc05fb34644190cffa7696bbf3

# Add OCI labels for container metadata
LABEL maintainer="EPdacoder05" \
      description="Sportsbook Aggregation Engine - Autonomous Sports Betting Analysis" \
      org.opencontainers.image.source="https://github.com/EPdacoder05/Sportsbook-aggregation" \
      org.opencontainers.image.title="Sportsbook Aggregation Engine" \
      org.opencontainers.image.description="Algorithmic Sports Betting Analysis with RLM Detection" \
      org.opencontainers.image.vendor="EPdacoder05"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 -s /sbin/nologin -d /app appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Healthcheck using curl to verify API /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Default command - run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
