#!/bin/bash

# Logging utilities for Second-Me scripts
# This file contains common logging functions that can be used by other scripts

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get current timestamp
get_timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Log a message to the console with formatting
log() {
    local message="$1"
    local level="${2:-INFO}"
    local color="${NC}"
    
    case $level in
        INFO) color="${BLUE}" ;;
        SUCCESS) color="${GREEN}" ;;
        WARNING) color="${YELLOW}" ;;
        ERROR) color="${RED}" ;;
        DEBUG) color="${GRAY}" ;;
        SECTION) color="${BOLD}" ;;
        STEP) color="${GRAY}" ;;
    esac
    
    echo -e "${GRAY}[$(get_timestamp)]${NC} ${color}[${level}]${NC} ${message}"
}

# Print formatted log messages with different levels
log_info() {
    log "$1" "INFO"
}

log_success() {
    log "$1" "SUCCESS"
}

log_warning() {
    log "$1" "WARNING"
}

log_error() {
    log "$1" "ERROR"
}

log_debug() {
    log "$1" "DEBUG"
}

log_section() {
    log "$1" "SECTION"
}

log_step() {
    log "$1" "STEP"
}

# Initialize logging
init_logging() {
    # Check if we should enable debug output
    if [ "${DEBUG_MODE}" = "true" ]; then
        log_debug "Debug logging enabled"
    fi
    
    # Set up trap for cleanup if needed
    if [ "${1}" = "trap_cleanup" ] && [ -n "${2}" ]; then
        trap "${2}" INT
    fi
}
