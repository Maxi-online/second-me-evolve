#!/bin/bash

# Source the logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/logging.sh"

# Load configuration from .env file
load_env() {
    if [ -f .env ]; then
        # Only load necessary environment variables
        export LOCAL_APP_PORT=$(grep '^LOCAL_APP_PORT=' .env | cut -d '=' -f2)
        export LOCAL_FRONTEND_PORT=$(grep '^LOCAL_FRONTEND_PORT=' .env | cut -d '=' -f2)
    else
        # Use default port if .env not found
        export LOCAL_APP_PORT=8002
        log_info "Using default port: ${LOCAL_APP_PORT}"
    fi
}

# Main function to stop services
stop_services() {
    log_step "Stopping services..."
    
    # Load environment variables
    load_env
    
    # Create run directory if it doesn't exist
    mkdir -p run
    
    # Stop backend service
    if [ -f run/.backend.pid ]; then
        BACKEND_PID=$(cat run/.backend.pid)
        if ps -p $BACKEND_PID > /dev/null; then
            kill $BACKEND_PID
            log_success "Backend service stopped (PID: $BACKEND_PID)"
        else
            log_info "Backend process not running, checking port ${LOCAL_APP_PORT}..."
            PORT_PID=$(lsof -ti:${LOCAL_APP_PORT} 2>/dev/null)
            if [ ! -z "$PORT_PID" ]; then
                kill -9 $PORT_PID
                log_success "Backend service forcefully stopped (Port PID: $PORT_PID)"
            else
                log_info "No backend service found running on port ${LOCAL_APP_PORT}"
            fi
        fi
        rm -f run/.backend.pid
    else
        log_info "No backend PID file found, checking port ${LOCAL_APP_PORT}..."
        PORT_PID=$(lsof -ti:${LOCAL_APP_PORT} 2>/dev/null)
        if [ ! -z "$PORT_PID" ]; then
            kill -9 $PORT_PID
            log_success "Backend service forcefully stopped (Port PID: $PORT_PID)"
        else
            log_info "No backend service found running on port ${LOCAL_APP_PORT}"
        fi
    fi
    
    # Double-check if port 8002 is still in use
    PORT_PID=$(lsof -ti:${LOCAL_APP_PORT} 2>/dev/null)
    if [ ! -z "$PORT_PID" ]; then
        log_warning "Port ${LOCAL_APP_PORT} is still in use, forcefully terminating process..."
        kill -9 $PORT_PID
        sleep 0.5
        
        # Check again
        PORT_PID=$(lsof -ti:${LOCAL_APP_PORT} 2>/dev/null)
        if [ ! -z "$PORT_PID" ]; then
            log_error "Failed to release port ${LOCAL_APP_PORT}, process (PID: $PORT_PID) is still running"
        else
            log_success "Successfully released port ${LOCAL_APP_PORT}"
        fi
    fi
    
    # Stop llama-server process
    log_info "Checking for llama-server processes..."
    LLAMA_PIDS=$(pgrep -f "llama-server")
    if [ ! -z "$LLAMA_PIDS" ]; then
        echo "$LLAMA_PIDS" | while read pid; do
            log_info "Stopping llama-server process (PID: $pid)..."
            kill $pid 2>/dev/null
            sleep 0.5
            # Check if process is still running
            if ps -p $pid > /dev/null 2>&1; then
                log_warning "Process still running, using force kill..."
                kill -9 $pid 2>/dev/null
            fi
        done
        log_success "llama-server processes stopped"
    else
        log_info "No llama-server processes found"
    fi
    
    # Check if port 8080 is still in use (common port for llama-server)
    log_info "Checking if port 8080 is in use..."
    PORT_PID=$(lsof -ti:8080 2>/dev/null)
    if [ ! -z "$PORT_PID" ]; then
        log_info "Stopping process using port 8080 (PID: $PORT_PID)..."
        kill -9 $PORT_PID
        log_success "Process using port 8080 forcefully terminated"
    else
        log_info "Port 8080 is not in use"
    fi
    
    # Stop frontend service
    if [ -f run/.frontend.pid ]; then
        FRONTEND_PID=$(cat run/.frontend.pid)
        if ps -p $FRONTEND_PID > /dev/null; then
            log_info "Stopping frontend process (PID: $FRONTEND_PID)..."
            # use pkill to terminate process tree
            pkill -P $FRONTEND_PID 2>/dev/null
            kill $FRONTEND_PID 2>/dev/null
            sleep 1
            
            # check if process is still running
            if ps -p $FRONTEND_PID > /dev/null; then
                log_warning "Frontend process still running, using force kill..."
                kill -9 $FRONTEND_PID 2>/dev/null
            fi
            
            log_success "Frontend process stopped (PID: $FRONTEND_PID)"
        else
            log_info "Frontend process not running"
        fi
        rm -f run/.frontend.pid
    fi
    
    # check and terminate all possible frontend related processes
    log_info "Checking for any remaining Next.js processes..."
    
    # find and terminate all Next.js related processes
    NEXT_PIDS=$(pgrep -f "next dev|next-server")
    if [ ! -z "$NEXT_PIDS" ]; then
        echo "$NEXT_PIDS" | while read pid; do
            log_info "Stopping Next.js process (PID: $pid)..."
            kill $pid 2>/dev/null
            sleep 0.5
            # check if process is still running
            if ps -p $pid > /dev/null 2>&1; then
                log_warning "Process still running, using force kill..."
                kill -9 $pid 2>/dev/null
            fi
        done
        log_success "Next.js processes stopped"
    fi
    
    # find and terminate all frontend related npm processes
    NPM_PIDS=$(pgrep -f "npm run dev")
    if [ ! -z "$NPM_PIDS" ]; then
        echo "$NPM_PIDS" | while read pid; do
            log_info "Stopping npm process (PID: $pid)..."
            kill $pid 2>/dev/null
            sleep 0.5
            # check if process is still running
            if ps -p $pid > /dev/null 2>&1; then
                log_warning "Process still running, using force kill..."
                kill -9 $pid 2>/dev/null
            fi
        done
        log_success "npm processes stopped"
    fi
    
    # check if frontend port is still in use
    if [ ! -z "${LOCAL_FRONTEND_PORT}" ]; then
        log_info "Checking if port ${LOCAL_FRONTEND_PORT} is still in use..."
        PORT_PID=$(lsof -ti:${LOCAL_FRONTEND_PORT} 2>/dev/null)
        if [ ! -z "$PORT_PID" ]; then
            log_info "Stopping process using port ${LOCAL_FRONTEND_PORT} (PID: $PORT_PID)..."
            kill -9 $PORT_PID
            log_success "Process using port ${LOCAL_FRONTEND_PORT} forcefully terminated"
        else
            log_info "Port ${LOCAL_FRONTEND_PORT} is not in use"
        fi
    fi
    
    log_success "All services stopped successfully"
}

# Execute stop services
stop_services
