#!/bin/bash

# Health check script for EEG-RAG production container
# Validates system components and API availability

set -e

# Configuration
HOST=${BIND:-0.0.0.0:8000}
HEALTH_ENDPOINT="http://${HOST}/health"
TIMEOUT=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if port is accessible
check_port() {
    local host_port="$1"
    if echo "$host_port" | grep -q ":"; then
        local host=$(echo "$host_port" | cut -d: -f1)
        local port=$(echo "$host_port" | cut -d: -f2)
    else
        local host="localhost"
        local port="$host_port"
    fi
    
    if nc -z "$host" "$port" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Check HTTP health endpoint
check_http_health() {
    local response
    local http_code
    
    # Try to get health status
    response=$(curl -s -m "$TIMEOUT" -w "%{http_code}" "$HEALTH_ENDPOINT" 2>/dev/null || echo "000")
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" = "200" ]; then
        log_info "HTTP health check passed"
        return 0
    else
        log_error "HTTP health check failed (code: $http_code)"
        return 1
    fi
}

# Check Redis connection
check_redis() {
    if [ -n "$REDIS_HOST" ]; then
        if check_port "${REDIS_HOST}:${REDIS_PORT:-6379}"; then
            log_info "Redis connection OK"
            return 0
        else
            log_warn "Redis connection failed"
            return 1
        fi
    else
        log_info "Redis not configured"
        return 0
    fi
}

# Check disk space
check_disk_space() {
    local threshold=90
    local usage
    
    usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$usage" -lt "$threshold" ]; then
        log_info "Disk space OK (${usage}% used)"
        return 0
    else
        log_warn "Disk space high (${usage}% used)"
        return 1
    fi
}

# Check memory usage
check_memory() {
    local threshold=90
    local usage
    
    if command -v free >/dev/null; then
        usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
        
        if [ "$usage" -lt "$threshold" ]; then
            log_info "Memory usage OK (${usage}% used)"
            return 0
        else
            log_warn "Memory usage high (${usage}% used)"
            return 1
        fi
    else
        log_info "Memory check skipped (free command not available)"
        return 0
    fi
}

# Check Python process
check_python_process() {
    if pgrep -f "gunicorn\|python.*eeg_rag" >/dev/null; then
        log_info "EEG-RAG process running"
        return 0
    else
        log_error "EEG-RAG process not found"
        return 1
    fi
}

# Main health check
main() {
    local exit_code=0
    
    echo "🩺 EEG-RAG Health Check $(date)"
    echo "================================"
    
    # Critical checks (must pass)
    if ! check_python_process; then
        exit_code=1
    fi
    
    if ! check_port "${BIND:-8000}"; then
        log_error "Application port not accessible"
        exit_code=1
    fi
    
    # HTTP health check (if port is up)
    if [ $exit_code -eq 0 ]; then
        if ! check_http_health; then
            exit_code=1
        fi
    fi
    
    # Non-critical checks (warnings only)
    check_redis
    check_disk_space
    check_memory
    
    # Summary
    echo "================================"
    if [ $exit_code -eq 0 ]; then
        log_info "Overall health: HEALTHY"
    else
        log_error "Overall health: UNHEALTHY"
    fi
    
    exit $exit_code
}

# Run health check
main "$@"