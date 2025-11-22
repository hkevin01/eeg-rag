# EEG-RAG Docker Image
# Python 3.9 base image for stability and compatibility
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment inside container (NOT in root!)
RUN python -m venv /app/venv

# Activate virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies in virtual environment
RUN pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy project files
COPY . .

# Install project in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/embeddings logs

# Expose ports (if needed for API later)
EXPOSE 8000

# Set default command
CMD ["python", "-m", "eeg_rag.cli.query", "--help"]
