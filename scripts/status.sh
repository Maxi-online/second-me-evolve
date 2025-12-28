#!/bin/bash

# Source the logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/logging.sh"

# Load configuration from .env file
load_env() {
    if [ -f .env ]; then
        # Load necessary environment variables
        export LOCAL_APP_PORT=$(grep '^LOCAL_APP_PORT=' .env | cut -d '=' -f2)
        export LOCAL_FRONTEND_PORT=$(grep '^LOCAL_FRONTEND_PORT=' .env | cut -d '=' -f2)
    else
        # Use default ports if .env not found
        export LOCAL_APP_PORT=8002
        export LOCAL_FRONTEND_PORT=3000
        log_warning "Using default ports: Backend=${LOCAL_APP_PORT}, Frontend=${LOCAL_FRONTEND_PORT}"
    fi
}

# Check if a port is in use
check_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null)
    
    if [ -z "$pid" ]; then
        echo -e "${RED}Not running${NC}"
        return 1
    else
        local process_name=$(ps -p $pid -o comm= 2>/dev/null)
        local process_cmd=$(ps -p $pid -o args= 2>/dev/null)
        echo -e "${GREEN}Running${NC} (PID: $pid, Process: $process_name)"
        echo "  Command: $process_cmd"
        return 0
    fi
}

# Check if a process is running by PID
check_process() {
    local pid_file=$1
    local service_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat $pid_file)
        if ps -p $pid > /dev/null 2>&1; then
            local process_name=$(ps -p $pid -o comm= 2>/dev/null)
            local process_cmd=$(ps -p $pid -o args= 2>/dev/null)
            echo -e "${GREEN}Running${NC} (PID: $pid, Process: $process_name)"
            echo "  Command: $process_cmd"
            return 0
        else
            echo -e "${RED}Not running${NC} (PID file exists but process is dead)"
            return 1
        fi
    else
        echo -e "${RED}Not running${NC} (No PID file found)"
        return 1
    fi
}

# Main function to check status
check_status() {
    log_section "SERVICE STATUS"
    
    # Load environment variables
    load_env
    
    # Check backend service status
    echo -e "${BOLD}Backend Service:${NC}"
    echo -n "  PID File: "
    check_process "run/.backend.pid" "Backend"
    backend_pid_status=$?
    
    echo -n "  Port ${LOCAL_APP_PORT}: "
    check_port ${LOCAL_APP_PORT}
    backend_port_status=$?
    
    # Check frontend service status
    echo -e "\n${BOLD}Frontend Service:${NC}"
    echo -n "  PID File: "
    check_process "run/.frontend.pid" "Frontend"
    frontend_pid_status=$?
    
    echo -n "  Port ${LOCAL_FRONTEND_PORT}: "
    check_port ${LOCAL_FRONTEND_PORT}
    frontend_port_status=$?
    
    # Check port 8080 status (additional port)
    echo -e "\n${BOLD}Port 8080 Status:${NC}"
    echo -n "  Port 8080: "
    check_port 8080
    port_8080_status=$?
    
    # Summary
    echo -e "\n${BOLD}Summary:${NC}"
    if [ $backend_pid_status -eq 0 ] || [ $backend_port_status -eq 0 ]; then
        echo -e "  Backend: ${GREEN}Active${NC}"
    else
        echo -e "  Backend: ${RED}Inactive${NC}"
    fi
    
    if [ $frontend_pid_status -eq 0 ] || [ $frontend_port_status -eq 0 ]; then
        echo -e "  Frontend: ${GREEN}Active${NC}"
    else
        echo -e "  Frontend: ${RED}Inactive${NC}"
    fi
    
    if [ $port_8080_status -eq 0 ]; then
        echo -e "  Port 8080: ${GREEN}In Use${NC}"
    else
        echo -e "  Port 8080: ${RED}Not In Use${NC}"
    fi
}

# Execute check status
check_status
