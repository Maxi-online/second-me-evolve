#!/bin/bash

# Source the logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/logging.sh"

# enter frontend directory
cd lpm_frontend

# start frontend service
log_info "Starting frontend service on port ${LOCAL_FRONTEND_PORT}..."
npm run dev
