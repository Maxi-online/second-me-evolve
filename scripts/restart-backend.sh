#!/bin/bash

# Source the logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/logging.sh"

# Load configuration from .env file
load_env() {
    if [ -f .env ]; then
        # Only load necessary environment variables
        export LOCAL_APP_PORT=$(grep '^LOCAL_APP_PORT=' .env | cut -d '=' -f2)
    else
        # Use default port if .env not found
        export LOCAL_APP_PORT=8002
        log_info "Using default port: ${LOCAL_APP_PORT}"
    fi
}

# Main function to restart backend services
restart_backend() {
    local force=false
    
    # Parse arguments
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --force) force=true ;;
            *) log_error "Unknown parameter: $1"; return 1 ;;
        esac
        shift
    done
    
    log_step "Restarting backend services..."
    
    # Load environment variables
    if ! load_env; then
        return 1
    fi
    
    # If --force parameter is used, clear data folder
    if [ "$force" = true ]; then
        log_warning "Force restart mode: This will clear the data folder, all data will be deleted!"
        read -p "Are you sure you want to continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Clearing data folder..."
            rm -rf "${LOCAL_BASE_DIR}/data"
            log_success "Data folder cleared, database will be reinitialized"
        else
            log_info "Operation cancelled"
            return 0
        fi
    fi
    
    # 1. Stop backend service
    BACKEND_PID=$(lsof -ti:8002)
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID
        log_success "Backend service stopped"
    fi
    rm -f run/.backend.pid backend.log
    
    # 2. Wait for port release
    sleep 2
    
    # 3. Start backend service
    ./scripts/start.sh --backend-only
}

# Execute restart service
restart_backend "$@"
