#!/bin/bash
# Python utility functions

# Import logging utilities if not already imported
if ! command -v log_warning &>/dev/null; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/logging.sh"
fi

# Get the appropriate Python command (python3 or python)
get_python_command() {
    local python_cmd=""
    
    # First check for python3 command
    if command -v python3 &>/dev/null; then
        python_cmd="python3"
    # Then check for python command
    elif command -v python &>/dev/null; then
        python_cmd="python"
    fi
    
    # Return the command
    echo "$python_cmd"
}

# Get the appropriate pip command (pip3 or pip)
get_pip_command() {
    local pip_cmd=""
    local python_cmd=$(get_python_command)
    
    # First try to use the matching pip version
    if [ "$python_cmd" = "python3" ] && command -v pip3 &>/dev/null; then
        pip_cmd="pip3"
    # Then check for any pip command
    elif command -v pip &>/dev/null; then
        pip_cmd="pip"
    # If no pip is found, try to use python -m pip
    elif [ -n "$python_cmd" ]; then
        if $python_cmd -m pip --version &>/dev/null; then
            pip_cmd="$python_cmd -m pip"
        fi
    fi
    
    # Return the command
    echo "$pip_cmd"
}
