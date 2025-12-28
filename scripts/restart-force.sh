#!/bin/bash

# Source the logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/logging.sh"

# Main function to force restart services
restart_services_force() {
    log_section "FORCE RESTARTING SERVICES"
    
    # Stop services
    log_info "Stopping services..."
    "${SCRIPT_DIR}/scripts/stop.sh"
    
    # Remove data directory
    log_warning "Removing data directory..."
    rm -rf data

    # Start services
    log_info "Starting services..."
    "${SCRIPT_DIR}/scripts/start.sh"
}

# Execute force restart services
restart_services_force
