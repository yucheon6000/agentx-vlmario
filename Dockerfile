FROM python:3.11-slim

# Install system dependencies (Java for simulation, ffmpeg for video, git for dependencies)
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    ffmpeg \
    git \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src src
COPY scenarios scenarios

# Sync dependencies using uv
RUN uv sync --locked

# Set environment variables
ENV PYTHONPATH=/app/src

# Default command: Execute the Mario benchmark
CMD ["uv", "run", "agentbeats-run", "scenarios/mario/scenario.toml", "--show-logs"]