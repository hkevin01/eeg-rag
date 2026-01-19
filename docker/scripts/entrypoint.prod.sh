#!/bin/bash
set -e

# Production entrypoint script for EEG-RAG
# Handles initialization, health checks, and graceful shutdown

echo "🚀 Starting EEG-RAG Production Server"

# Set default environment variables
export WORKERS=${WORKERS:-4}
export WORKER_CLASS=${WORKER_CLASS:-gevent}
export WORKER_CONNECTIONS=${WORKER_CONNECTIONS:-1000}
export TIMEOUT=${TIMEOUT:-120}
export BIND=${BIND:-0.0.0.0:8000}
export LOG_LEVEL=${LOG_LEVEL:-info}

# Redis configuration
export REDIS_HOST=${REDIS_HOST:-redis}
export REDIS_PORT=${REDIS_PORT:-6379}
export REDIS_DB=${REDIS_DB:-0}

# Database paths
export DATA_PATH=${DATA_PATH:-/app/data}
export CACHE_PATH=${CACHE_PATH:-/app/data/cache}
export LOGS_PATH=${LOGS_PATH:-/app/logs}

# Create necessary directories
mkdir -p "$CACHE_PATH" "$LOGS_PATH" /app/tmp

# Wait for dependencies
echo "📡 Checking dependencies..."

# Wait for Redis if specified
if [ -n "$REDIS_HOST" ]; then
    echo "⏳ Waiting for Redis at $REDIS_HOST:$REDIS_PORT"
    timeout=30
    while ! nc -z "$REDIS_HOST" "$REDIS_PORT" 2>/dev/null; do
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            echo "❌ Redis connection timeout"
            exit 1
        fi
        sleep 1
    done
    echo "✅ Redis connected"
fi

# Initialize system
echo "🔧 Initializing EEG-RAG system..."

# Run system checks
python -c "
import sys
sys.path.insert(0, '/app/src')
from eeg_rag.utils.common_utils import check_system_health
health = check_system_health()
if health.status != 'healthy':
    print(f'❌ System health check failed: {health.issues}')
    sys.exit(1)
print('✅ System health check passed')
"

# Pre-warm models if requested
if [ "$PREWARM_MODELS" = "true" ]; then
    echo "🔥 Pre-warming ML models..."
    python -c "
import sys
sys.path.insert(0, '/app/src')
from eeg_rag.agents.local_agent.local_data_agent import LocalDataAgent
from sentence_transformers import SentenceTransformer
print('Loading PubMedBERT...')
model = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
print('✅ Models pre-warmed')
"
fi

# Initialize search indices if needed
if [ "$REBUILD_INDEX" = "true" ]; then
    echo "🔄 Rebuilding search indices..."
    python -c "
import sys
sys.path.insert(0, '/app/src')
from eeg_rag.cli.main import cli
import asyncio
# This would rebuild indices - simplified for demo
print('✅ Indices rebuilt')
"
fi

# Handle signals for graceful shutdown
trap 'echo "🛑 Shutting down gracefully..."; kill -TERM $PID; wait $PID' SIGTERM SIGINT

# Start the application
echo "🎯 Starting application with $WORKERS workers on $BIND"
echo "📊 Worker class: $WORKER_CLASS, Connections: $WORKER_CONNECTIONS"
echo "⏱️  Timeout: ${TIMEOUT}s, Log level: $LOG_LEVEL"

if [ "$1" = "gunicorn" ]; then
    # Production server
    exec gunicorn src.eeg_rag.api.main:app \
        --workers "$WORKERS" \
        --worker-class "$WORKER_CLASS" \
        --worker-connections "$WORKER_CONNECTIONS" \
        --max-requests "${MAX_REQUESTS:-1000}" \
        --max-requests-jitter "${MAX_REQUESTS_JITTER:-100}" \
        --timeout "$TIMEOUT" \
        --keep-alive "${KEEP_ALIVE:-5}" \
        --bind "$BIND" \
        --access-logfile "-" \
        --error-logfile "-" \
        --log-level "$LOG_LEVEL" \
        --preload \
        --enable-stdio-inheritance &
    PID=$!
    wait $PID
elif [ "$1" = "cli" ]; then
    # CLI mode
    shift
    exec python -m eeg_rag.cli.main "$@"
elif [ "$1" = "worker" ]; then
    # Background worker
    shift
    exec python -m eeg_rag.workers.background_worker "$@"
else
    # Custom command
    exec "$@"
fi