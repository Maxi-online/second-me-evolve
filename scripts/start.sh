#!/bin/bash

# Script version
VERSION="1.0.0"

# Source the logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/logging.sh"

# Check if port is available
check_port() {
    local port=$1
    if lsof -i:${port} > /dev/null 2>&1; then
        return 1
    fi
    return 0
}

# Check if backend is healthy with real-time log display
check_backend_health() {
    local max_attempts=$1
    local attempt=1
    local backend_url="http://127.0.0.1:${LOCAL_APP_PORT}/health"
    local backend_log="logs/start.log"
    local log_pid=0
    
    log_info "Waiting for backend service to be ready (showing real-time logs)..."
    
    # Start real-time log display in background if log file exists
    if [ -f "$backend_log" ]; then
        echo -e "${GRAY}---Backend logs begin (real-time)---${NC}"
        tail -f "$backend_log" &
        log_pid=$!
    fi
    
    while [ $attempt -le $max_attempts ]; do
        # Non-blocking health check
        if curl -s -f "$backend_url" &>/dev/null; then
            # Stop the log display process
            if [ $log_pid -ne 0 ]; then
                kill $log_pid >/dev/null 2>&1
                echo -e "${GRAY}---Backend logs end---${NC}"
            fi
            return 0
        fi
        
        sleep 1
        attempt=$((attempt + 1))
    done
    
    # Stop the log display process if it's still running
    if [ $log_pid -ne 0 ]; then
        kill $log_pid >/dev/null 2>&1
        echo -e "${GRAY}---Backend logs end---${NC}"
    fi
    
    return 1
}

# Check if frontend is ready with real-time log display
check_frontend_ready() {
    
    local max_attempts=$1
    local attempt=1
    local frontend_log="logs/frontend.log"
    local log_pid=0
    
    log_info "Waiting for frontend service to be ready (showing real-time logs)..."
    
    # Don't wait for file to exist, just start tail which will wait for the file
    echo -e "${GRAY}---Frontend logs begin (real-time)---${NC}"
    tail -f "$frontend_log" 2>/dev/null &
    log_pid=$!
    
    # Give a small delay to allow initial logs to appear
    sleep 1
    
    while [ $attempt -le $max_attempts ]; do
        # Non-blocking ready check - check for "Local:" in the file or the existence of the frontend URL
        if grep -q "Local:" "$frontend_log" 2>/dev/null || curl -s -f "http://localhost:${LOCAL_FRONTEND_PORT}" &>/dev/null; then
            # Frontend is ready! Stop the log display process
            if [ $log_pid -ne 0 ]; then
                kill $log_pid >/dev/null 2>&1
                echo -e "${GRAY}---Frontend logs end---${NC}"
            fi
            
            # Display the frontend URL that was found in the logs
            if grep -q "Local:" "$frontend_log" 2>/dev/null; then
                local frontend_url=$(grep "Local:" "$frontend_log" | head -n 1)
                log_success "Frontend URL detected: $frontend_url"
            else
                log_success "Frontend is responding at http://localhost:${LOCAL_FRONTEND_PORT}"
            fi
            
            return 0
        fi
        
        sleep 1
        attempt=$((attempt + 1))
    done
    
    # Stop the log display process if it's still running
    if [ $log_pid -ne 0 ]; then
        kill $log_pid >/dev/null 2>&1
        echo -e "${GRAY}---Frontend logs end---${NC}"
    fi
    
    return 1
}

# Check if setup is complete
check_setup_complete() {
    log_info "Checking if setup is complete..."
   
    
    # Check if frontend dependencies are installed
    if [ ! -d "lpm_frontend/node_modules" ] && [ "$BACKEND_ONLY" != "true" ]; then
        log_error "Frontend dependencies not installed. Please run 'make setup' first."
        return 1
    fi
    
    log_success "Setup check passed"
    return 0
}

# Main function to start services
start_services() {
    log_section "STARTING SERVICES"
    
    # Parse arguments
    BACKEND_ONLY="false"
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --backend-only) BACKEND_ONLY="true" ;;
            *) log_error "Unknown parameter: $1"; return 1 ;;
        esac
        shift
    done
    
    # Check if setup is complete
    if ! check_setup_complete; then
        return 1
    fi

    log_step "Loading environment variables"
    
    # Load environment variables
    if [[ -f .env ]]; then
        export LOCAL_APP_PORT="$(grep '^LOCAL_APP_PORT=' .env | cut -d '=' -f2)"
        export LOCAL_FRONTEND_PORT="$(grep '^LOCAL_FRONTEND_PORT=' .env | cut -d '=' -f2)"
        
        if [[ -z "$LOCAL_APP_PORT" ]]; then
            export LOCAL_APP_PORT="8002"
        fi
        if [[ -z "$LOCAL_FRONTEND_PORT" ]]; then
            export LOCAL_FRONTEND_PORT="3000"
        fi
    else
        log_error ".env file not found!"
        return 1
    fi
    
    log_success "Environment variables loaded"
    
    # Check if ports are available
    log_step "Checking port availability..."
    if ! check_port ${LOCAL_APP_PORT}; then
        log_error "Backend port ${LOCAL_APP_PORT} is already in use!"
        return 1
    fi
    if ! check_port ${LOCAL_FRONTEND_PORT} && [[ "$BACKEND_ONLY" != "true" ]]; then
        log_error "Frontend port ${LOCAL_FRONTEND_PORT} is already in use!"
        return 1
    fi
    log_success "All ports are available"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    mkdir -p run
    
    # Start backend service
    log_step "Starting backend service..."
    
    nohup bash -c ./scripts/start_local.sh > logs/start.log 2>&1 &

    
    echo $! > run/.backend.pid
    log_info "Backend service started in background with PID: $(cat run/.backend.pid)"
    
    # Wait for backend to be healthy
    log_info "Waiting for backend service to be ready..."
    if ! check_backend_health 300; then
        log_error "Backend service failed to start within 300 seconds"
        return 1
    fi
    log_success "Backend service is ready"
    
    # Start frontend service if not backend-only mode
    if [[ "$BACKEND_ONLY" != "true" ]]; then
        if [ ! -d "lpm_frontend" ]; then
            log_error "Frontend directory 'lpm_frontend' not found!"
            return 1
        fi
        
        log_step "Starting frontend service..."
        cd lpm_frontend

        # Copy environment variables from root directory to frontend directory
        log_info "Copying environment variables to frontend directory..."
        if [[ -f "../.env" ]]; then
            # Extract required environment variables and create frontend .env file
            grep -E "^(HOST_ADDRESS|LOCAL_APP_PORT)=" "../.env" > .env
            log_success "Environment variables copied to frontend .env file"
        else
            log_warning "Root directory .env file does not exist, cannot copy environment variables"
        fi
        
        # Check if node_modules exists
        if [ ! -d "node_modules" ]; then
            log_info "Installing frontend dependencies..."
            if ! npm install; then
                log_error "Failed to install frontend dependencies"
                cd ..
                return 1
            fi
            log_success "Frontend dependencies installed"
        fi
        
        # Start frontend in background
        log_info "Starting frontend dev server..."
        nohup npm run dev > ../logs/frontend.log 2>&1 &
        echo $! > ../run/.frontend.pid
        log_info "Frontend service started in background with PID: $(cat ../run/.frontend.pid)"
        
        # Wait for frontend to be ready
        log_info "Waiting for frontend service to be ready..."
        if ! check_frontend_ready 300; then
            log_error "Frontend service failed to start within 300 seconds"
            cd ..
            return 1
        fi
        log_success "Frontend service is ready"
        cd ..
    fi
    
    # Display service URLs
    log_section "Services are running"
    if [[ "$BACKEND_ONLY" == "true" ]]; then
        log_info "Backend URL:  http://localhost:${LOCAL_APP_PORT}"
    else
        log_info "Frontend URL: http://localhost:${LOCAL_FRONTEND_PORT}"
        log_info "Backend URL:  http://localhost:${LOCAL_APP_PORT}"
    fi
}

# Execute start services
start_services "$@"

# Error handling
if [[ $? -ne 0 ]]; then
    log_error "Failed to start services!"
    exit 1
fi
