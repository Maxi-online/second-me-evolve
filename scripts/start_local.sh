#!/bin/bash

# Source the logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/logging.sh"

# Activate Poetry virtual environment if available (but only if not already activated)
log_info "Setting up Python environment..."

# Check if Python environment is already activated by looking for specific environment markers
if [[ "$VIRTUAL_ENV" != "" ]]; then
    log_info "Python virtual environment already activated: $VIRTUAL_ENV"
else
    POETRY_ENV_PATH=""
    if command -v poetry &>/dev/null; then
        POETRY_ENV_PATH=$(poetry env info -p 2>/dev/null)
        if [ -n "$POETRY_ENV_PATH" ] && [ -f "$POETRY_ENV_PATH/bin/activate" ]; then
            log_info "Activating Poetry virtual environment: $POETRY_ENV_PATH"
            source "$POETRY_ENV_PATH/bin/activate"
        else
            # Try using the local activation script if it exists
            if [ -f ".poetry-venv/activate" ]; then
                log_info "Activating Poetry environment via local script"
                source ".poetry-venv/activate"
            else
                log_warning "Poetry environment not found. Some dependencies might be missing."
            fi
        fi
    else
        log_warning "Poetry is not installed. Some dependencies might be missing."
    fi
fi

# Set environment variables
log_info "Setting environment variables..."
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Load environment variables from .env file
set -a
source .env
set +a

# Use local base directory
export BASE_DIR=${LOCAL_BASE_DIR}

# Ensure using the correct Python environment
log_info "Checking Python environment..."
PYTHON_PATH=$(which python)
log_info "Using Python: $PYTHON_PATH"
PYTHON_VERSION=$(python --version)
log_info "Python version: $PYTHON_VERSION"

# Check necessary Python packages
log_info "Checking necessary Python packages..."
python -c "import flask" || { log_error "Error: Missing flask package"; exit 1; }
python -c "import chromadb" || { log_error "Error: Missing chromadb package"; exit 1; }

# Initialize database
log_info "Initializing database..."
SQLITE_DB_PATH="${BASE_DIR}/data/sqlite/lpm.db"
mkdir -p "${BASE_DIR}/data/sqlite"

if [ ! -f "$SQLITE_DB_PATH" ]; then
    log_info "Initializing database..."
    cat docker/sqlite/init.sql | sqlite3 "$SQLITE_DB_PATH"
    log_success "Database initialization completed"
else
    log_info "Database already exists"
fi

# Ensure necessary directories exist
log_info "Checking necessary directories..."
mkdir -p ${BASE_DIR}/data/chroma_db
mkdir -p ${LOCAL_LOG_DIR}
#mkdir -p ${BASE_DIR}/raw_content
#mkdir -p ${BASE_DIR}/data_pipeline

# Initialize ChromaDB
log_info "Initializing ChromaDB..."
python docker/app/init_chroma.py

# Get local IP address (excluding localhost and docker networks)
LOCAL_IP=$(ifconfig | grep "inet " | grep -v "127.0.0.1" | grep "192.168" | awk '{print $2}' | head -n 1)

# Run database migrations first
log_info "Running database migrations..."
python scripts/run_migrations.py

# Start Flask application
log_info "Starting Flask application..."
log_info "Application will run at the following addresses:"
log_info "- Local access: http://localhost:${LOCAL_APP_PORT}"
log_info "- LAN access: http://${LOCAL_IP}:${LOCAL_APP_PORT}"

# Output logs to file
exec python -m flask run --host=0.0.0.0 --port=${LOCAL_APP_PORT} >> "${LOCAL_LOG_DIR}/backend.log" 2>&1
