#!/bin/bash
# ============================================================================
# EEG-RAG Automated Corpus Update Script
# ============================================================================
# This script fetches new EEG papers from the last 30 days
# Schedule with cron to run daily for continuous corpus updates
#
# Cron examples:
#   Daily at 3 AM:    0 3 * * * /path/to/schedule_updates.sh
#   Every 6 hours:    0 */6 * * * /path/to/schedule_updates.sh
#   Weekly Sunday:    0 4 * * 0 /path/to/schedule_updates.sh
#
# Setup:
#   1. chmod +x scripts/schedule_updates.sh
#   2. crontab -e
#   3. Add: 0 3 * * * /home/kevin/Projects/eeg-rag/scripts/schedule_updates.sh
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"
LOG_DIR="$PROJECT_ROOT/logs"
LOCK_FILE="/tmp/eeg_rag_update.lock"

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/scheduled_update_$TIMESTAMP.log"

# Create logs directory if needed
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check for lock file to prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    PID=$(cat "$LOCK_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        log "ERROR: Another update is already running (PID: $PID)"
        exit 1
    else
        log "WARNING: Stale lock file found, removing..."
        rm -f "$LOCK_FILE"
    fi
fi

# Create lock file
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

# Start update
log "=============================================="
log "EEG-RAG Scheduled Corpus Update"
log "=============================================="
log "Project root: $PROJECT_ROOT"
log "Virtual env: $VENV_PATH"

# Change to project directory
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    log "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    log "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Load environment variables if .env exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    log "Loading environment variables from .env..."
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Run the update
log "Starting paper ingestion (last 30 days)..."
log "----------------------------------------------"

python scripts/run_massive_ingestion.py \
    --update-latest \
    --include-preprints \
    2>&1 | tee -a "$LOG_FILE"

UPDATE_STATUS=$?

log "----------------------------------------------"

if [ $UPDATE_STATUS -eq 0 ]; then
    log "✓ Update completed successfully"
    
    # Optionally rebuild index after update
    if [ "$REBUILD_INDEX" = "true" ]; then
        log "Rebuilding search index..."
        python -c "from eeg_rag.retrieval import HybridRetriever; r = HybridRetriever(); r.rebuild_index()" \
            2>&1 | tee -a "$LOG_FILE"
        log "✓ Index rebuilt"
    fi
    
    # Send notification if configured
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        echo "EEG-RAG corpus update completed successfully at $(date)" | \
            mail -s "EEG-RAG Update Success" "$NOTIFICATION_EMAIL"
    fi
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"✅ EEG-RAG corpus update completed successfully"}' \
            "$SLACK_WEBHOOK" 2>/dev/null
    fi
else
    log "✗ Update failed with status $UPDATE_STATUS"
    
    # Send error notification
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        tail -100 "$LOG_FILE" | \
            mail -s "EEG-RAG Update FAILED" "$NOTIFICATION_EMAIL"
    fi
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"❌ EEG-RAG corpus update FAILED. Check logs."}' \
            "$SLACK_WEBHOOK" 2>/dev/null
    fi
    
    exit 1
fi

# Cleanup old logs (keep last 30 days)
log "Cleaning up old log files..."
find "$LOG_DIR" -name "scheduled_update_*.log" -mtime +30 -delete

log "=============================================="
log "Update process complete"
log "=============================================="
