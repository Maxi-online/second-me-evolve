#!/bin/bash
# OS detection utility functions

# Detect OS type: Returns "macos", "linux", "windows", or "unknown"
detect_os_type() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Detect Linux distribution: Returns "debian", "fedora", "redhat", "arch", "alpine", or "other"
detect_linux_distro() {
    if [ -f /etc/debian_version ]; then
        echo "debian"
    elif [ -f /etc/fedora-release ]; then
        echo "fedora"
    elif [ -f /etc/redhat-release ]; then
        echo "redhat"
    elif [ -f /etc/arch-release ]; then
        echo "arch"
    elif [ -f /etc/alpine-release ]; then
        echo "alpine"
    else
        echo "other"
    fi
}

# Get full system identification
get_system_id() {
    local os_type=$(detect_os_type)
    
    if [ "$os_type" = "linux" ]; then
        local linux_distro=$(detect_linux_distro)
        echo "${os_type}-${linux_distro}"
    else
        echo "$os_type"
    fi
}
