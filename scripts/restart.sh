#!/bin/bash

# Source the logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/logging.sh"

# Main function to restart services
restart_services() {
    log_section "RESTARTING SERVICES"
    
    # Stop services
    log_info "Stopping services..."
    ./scripts/stop.sh
    
    # Start services
    log_info "Starting services..."
    ./scripts/start.sh
}

# Execute restart services
restart_services
