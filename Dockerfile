FROM python:3.11-slim

# Install system dependencies for OpenCV and headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary from Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# Install project dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy project source and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY Makefile README.md ./

# Install project into virtual environment
RUN uv sync --frozen --no-dev

# Switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["rigsim"]
CMD ["--help"]
