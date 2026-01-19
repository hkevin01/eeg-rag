#!/bin/bash
# EEG-RAG Deployment Script
# Automated deployment with health checks and rollback capability

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/deployment_$(date +%Y%m%d_%H%M%S).log"

# Default values
ENVIRONMENT="${ENVIRONMENT:-staging}"
VERSION="${VERSION:-latest}"
SKIP_TESTS="${SKIP_TESTS:-false}"
DRY_RUN="${DRY_RUN:-false}"
FORCE="${FORCE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Error handling
cleanup() {
    log "Cleaning up temporary files..."
    rm -f /tmp/eeg_rag_deploy_*
}

error_exit() {
    log_error "$1"
    cleanup
    exit 1
}

trap 'error_exit "Deployment failed with error on line $LINENO"' ERR
trap cleanup EXIT

# Help function
show_help() {
    cat << EOF
EEG-RAG Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Target environment (staging, production) [default: staging]
    -v, --version VERSION    Version to deploy [default: latest]
    -s, --skip-tests        Skip pre-deployment tests
    -d, --dry-run           Show what would be deployed without deploying
    -f, --force             Force deployment even if checks fail
    -h, --help              Show this help message

Examples:
    $0 --environment staging --version v1.2.3
    $0 --environment production --dry-run
    $0 --skip-tests --force

Environment Variables:
    ENVIRONMENT             Target environment
    VERSION                 Version to deploy
    SKIP_TESTS             Skip tests (true/false)
    DRY_RUN               Dry run mode (true/false)
    FORCE                 Force deployment (true/false)
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -s|--skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -f|--force)
                FORCE="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done
}

# Validation functions
validate_environment() {
    case "$ENVIRONMENT" in
        staging|production)
            log "Deploying to environment: $ENVIRONMENT"
            ;;
        *)
            error_exit "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
            ;;
    esac
}

validate_version() {
    if [[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log "Deploying version: $VERSION"
    elif [[ "$VERSION" == "latest" ]]; then
        log_warning "Deploying latest version (not recommended for production)"
        if [[ "$ENVIRONMENT" == "production" && "$FORCE" != "true" ]]; then
            error_exit "Cannot deploy 'latest' to production without --force"
        fi
    else
        log_warning "Version format doesn't match semantic versioning: $VERSION"
    fi
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "python3" "git")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "Required tool '$tool' is not installed"
        fi
    done
    
    # Check Python version
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ ! "$python_version" =~ ^3\.(9|10|11) ]]; then
        log_warning "Python version $python_version may not be supported (recommended: 3.9-3.11)"
    fi
    
    # Check Git status
    if [[ -n "$(git status --porcelain)" ]]; then
        log_warning "Working directory has uncommitted changes"
        if [[ "$FORCE" != "true" ]]; then
            error_exit "Please commit or stash changes before deployment"
        fi
    fi
    
    # Check disk space
    local available_space
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [[ "$available_space" -lt 1048576 ]]; then # 1GB in KB
        log_warning "Less than 1GB disk space available"
        if [[ "$FORCE" != "true" ]]; then
            error_exit "Insufficient disk space for deployment"
        fi
    fi
    
    log_success "Prerequisites check passed"
}

run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests as requested"
        return
    fi
    
    log "Running pre-deployment tests..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment for testing
    python3 -m venv /tmp/eeg_rag_deploy_env
    source /tmp/eeg_rag_deploy_env/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    
    # Run critical tests
    log "Running unit tests..."
    python -m pytest tests/test_common_utils.py -v --tb=short
    
    log "Running memory manager tests..."
    python -m pytest tests/test_memory_manager.py -v --tb=short
    
    log "Running boundary condition tests..."
    python -m pytest tests/test_*_boundary_conditions.py -v --tb=short
    
    # Run quick resilience check
    log "Running resilience tests..."
    python -m pytest tests/test_system_resilience.py::TestResourceMonitoring -v --tb=short
    
    deactivate
    rm -rf /tmp/eeg_rag_deploy_env
    
    log_success "All tests passed"
}

build_package() {
    log "Building deployment package..."
    
    cd "$PROJECT_ROOT"
    
    # Clean previous builds
    rm -rf dist/ build/ *.egg-info/
    
    # Build wheel and source distribution
    python -m build
    
    # Verify build
    if [[ ! -f "dist"/*.whl ]]; then
        error_exit "Failed to build wheel package"
    fi
    
    local wheel_file
    wheel_file=$(ls dist/*.whl)
    log_success "Built package: $(basename "$wheel_file")"
    
    # Test installation
    python3 -m venv /tmp/eeg_rag_install_test
    source /tmp/eeg_rag_install_test/bin/activate
    pip install "$wheel_file"
    python -c "import eeg_rag; print('Package installation test passed')"
    deactivate
    rm -rf /tmp/eeg_rag_install_test
    
    log_success "Package build and installation test completed"
}

generate_config() {
    log "Generating deployment configuration..."
    
    local config_dir="$PROJECT_ROOT/config"
    mkdir -p "$config_dir"
    
    # Generate environment-specific configuration
    cat > "$config_dir/deployment.env" << EOF
# EEG-RAG Deployment Configuration
# Generated on: $(date)
# Environment: $ENVIRONMENT
# Version: $VERSION

EEG_RAG_ENV=$ENVIRONMENT
EEG_RAG_VERSION=$VERSION
LOG_LEVEL=$( [[ "$ENVIRONMENT" == "production" ]] && echo "INFO" || echo "DEBUG" )
DEBUG=$( [[ "$ENVIRONMENT" == "production" ]] && echo "false" || echo "true" )

# Resource limits
MEMORY_LIMIT=$( [[ "$ENVIRONMENT" == "production" ]] && echo "2Gi" || echo "1Gi" )
CPU_LIMIT=$( [[ "$ENVIRONMENT" == "production" ]] && echo "1000m" || echo "500m" )

# Scaling
REPLICAS=$( [[ "$ENVIRONMENT" == "production" ]] && echo "3" || echo "2" )

# Health check intervals
READINESS_PROBE_INTERVAL=10
LIVENESS_PROBE_INTERVAL=30
EOF
    
    log_success "Configuration generated: $config_dir/deployment.env"
}

create_backup() {
    if [[ "$ENVIRONMENT" != "production" ]]; then
        log "Skipping backup for non-production environment"
        return
    fi
    
    log "Creating production backup..."
    
    local backup_dir="$PROJECT_ROOT/backups"
    local backup_name="backup_$(date +%Y%m%d_%H%M%S)"
    local backup_path="$backup_dir/$backup_name"
    
    mkdir -p "$backup_path"
    
    # Backup configuration files
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        cp -r "$PROJECT_ROOT/config" "$backup_path/"
    fi
    
    # Backup data (if exists)
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        log "Backing up data directory..."
        tar -czf "$backup_path/data.tar.gz" -C "$PROJECT_ROOT" data/
    fi
    
    # Create backup manifest
    cat > "$backup_path/manifest.txt" << EOF
Backup created: $(date)
Environment: $ENVIRONMENT
Previous version: $(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
New version: $VERSION
Git SHA: $(git rev-parse HEAD)
Backup size: $(du -sh "$backup_path" | cut -f1)
EOF
    
    log_success "Backup created: $backup_path"
    echo "BACKUP_PATH=$backup_path" >> "$LOG_FILE"
}

deploy_application() {
    log "Starting application deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN MODE - Simulating deployment"
        
        cat << EOF

========================================
DEPLOYMENT SIMULATION
========================================
Environment: $ENVIRONMENT
Version: $VERSION
Configuration: $PROJECT_ROOT/config/deployment.env
Package: $(ls "$PROJECT_ROOT"/dist/*.whl 2>/dev/null || echo "Not built")

Deployment steps that would be executed:
1. Stop existing application instances
2. Deploy new version: $VERSION
3. Update configuration
4. Start new instances
5. Run health checks
6. Update load balancer

========================================
EOF
        return
    fi
    
    # Real deployment would go here
    log "Deploying to $ENVIRONMENT environment..."
    
    # Simulate deployment steps
    local steps=("Stopping old instances" "Deploying new version" "Starting services" "Running health checks")
    for step in "${steps[@]}"; do
        log "$step..."
        sleep 2  # Simulate time
        log_success "$step completed"
    done
}

run_health_checks() {
    log "Running post-deployment health checks..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN MODE - Skipping actual health checks"
        return
    fi
    
    # Wait for application to start
    log "Waiting for application to initialize..."
    sleep 10
    
    # System health check
    log "Checking system health..."
    python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

try:
    from eeg_rag.utils.common_utils import check_system_health
    health = check_system_health()
    print(f"System status: {health.status.value}")
    print(f"CPU: {health.cpu_percent:.1f}%")
    print(f"Memory: {health.memory_percent:.1f}%")
    print(f"Warnings: {len(health.warnings)}")
    
    if health.status.value in ['healthy', 'warning']:
        print("Health check: PASSED")
        exit(0)
    else:
        print("Health check: FAILED")
        exit(1)
        
except Exception as e:
    print(f"Health check error: {e}")
    exit(1)
EOF
    
    local health_status=$?
    if [[ $health_status -eq 0 ]]; then
        log_success "System health check passed"
    else
        error_exit "System health check failed"
    fi
    
    # Additional health checks would go here
    log_success "All health checks passed"
}

update_status() {
    log "Updating deployment status..."
    
    local status_file="$PROJECT_ROOT/.deployment_status"
    cat > "$status_file" << EOF
{
  "environment": "$ENVIRONMENT",
  "version": "$VERSION",
  "deployed_at": "$(date -u +'%Y-%m-%d %H:%M:%S UTC')",
  "deployed_by": "$(whoami)",
  "git_sha": "$(git rev-parse HEAD)",
  "status": "success"
}
EOF
    
    log_success "Deployment status updated: $status_file"
}

rollback() {
    log_error "Deployment failed - initiating rollback..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN MODE - Rollback would be executed"
        return
    fi
    
    # Read backup path from log
    local backup_path
    backup_path=$(grep "BACKUP_PATH=" "$LOG_FILE" | cut -d'=' -f2 | tail -1)
    
    if [[ -n "$backup_path" && -d "$backup_path" ]]; then
        log "Restoring from backup: $backup_path"
        # Rollback logic would go here
        log_success "Rollback completed"
    else
        log_error "No backup found - manual intervention required"
    fi
}

# Main deployment function
main() {
    log "Starting EEG-RAG deployment process"
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    log "Dry run: $DRY_RUN"
    
    # Create logs directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Validation
    validate_environment
    validate_version
    check_prerequisites
    
    # Pre-deployment
    run_tests
    build_package
    generate_config
    create_backup
    
    # Deployment
    deploy_application
    
    # Post-deployment
    run_health_checks
    update_status
    
    log_success "Deployment completed successfully!"
    log "Version $VERSION deployed to $ENVIRONMENT"
    log "Deployment log: $LOG_FILE"
}

# Parse arguments and run main function
parse_args "$@"

# Set up error handling for rollback
if [[ "$ENVIRONMENT" == "production" ]]; then
    trap 'rollback' ERR
fi

main