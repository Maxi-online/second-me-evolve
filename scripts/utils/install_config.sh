#!/bin/bash
# Installation recommendations configuration

# Import logging utilities if not already imported
if ! command -v log_warning &>/dev/null; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/logging.sh"
fi

# Get installation recommendation for a package
get_install_recommendation() {
    local package="$1"
    local system_id="$2"
    
    case "$package" in
        "python")
            get_python_recommendation "$system_id"
            ;;
        "npm")
            get_npm_recommendation "$system_id"
            ;;
        "node")
            get_node_recommendation "$system_id"
            ;;
        "cmake")
            get_cmake_recommendation "$system_id"
            ;;
        "poetry")
            get_poetry_recommendation "$system_id"
            ;;
        "sqlite")
            get_sqlite_recommendation "$system_id"
            ;;
        *)
            log_warning "No specific recommendation available for $package"
            ;;
    esac
}

# Python installation recommendations
get_python_recommendation() {
    local system_id="$1"
    
    case "$system_id" in
        "macos")
            log_warning "Recommended installation for macOS: 'brew install python3'"
            log_warning "Or download from: https://www.python.org/downloads/macos/"
            ;;
        "linux-debian")
            log_warning "Recommended installation for Debian/Ubuntu: 'sudo apt update && sudo apt install python3 python3-pip'"
            ;;
        "linux-fedora")
            log_warning "Recommended installation for Fedora: 'sudo dnf install python3 python3-pip'"
            ;;
        "linux-redhat")
            log_warning "Recommended installation for CentOS/RHEL: 'sudo yum install python3 python3-pip'"
            ;;
        "linux-arch")
            log_warning "Recommended installation for Arch Linux: 'sudo pacman -S python python-pip'"
            ;;
        "linux-alpine")
            log_warning "Recommended installation for Alpine Linux: 'apk add python3 py3-pip'"
            ;;
        "linux-other")
            log_warning "Please install Python 3.12+ using your distribution's package manager"
            log_warning "Or download from: https://www.python.org/downloads/linux/"
            ;;
        "windows")
            log_warning "Recommended installation for Windows:"
            log_warning "1. Download from: https://www.python.org/downloads/windows/"
            log_warning "2. Or using winget: 'winget install Python.Python.3'"
            log_warning "3. Or using Chocolatey: 'choco install python'"
            ;;
        *)
            log_warning "Please download Python from: https://www.python.org/downloads/"
            ;;
    esac
}

# NPM installation recommendations
get_npm_recommendation() {
    local system_id="$1"
    
    case "$system_id" in
        "macos")
            log_warning "Recommended installation for macOS: 'brew install npm'"
            ;;
        "linux-debian")
            log_warning "Recommended installation for Debian/Ubuntu: 'sudo apt update && sudo apt install npm'"
            ;;
        "linux-fedora")
            log_warning "Recommended installation for Fedora: 'sudo dnf install npm'"
            ;;
        "linux-redhat")
            log_warning "Recommended installation for CentOS/RHEL: 'sudo yum install npm'"
            ;;
        "linux-arch")
            log_warning "Recommended installation for Arch Linux: 'sudo pacman -S npm'"
            ;;
        "linux-alpine")
            log_warning "Recommended installation for Alpine Linux: 'apk add npm'"
            ;;
        "linux-other")
            log_warning "Please install npm using your distribution's package manager"
            ;;
        "windows")
            log_warning "Recommended installation for Windows:"
            log_warning "1. Install Node.js (includes npm): https://nodejs.org/en/download/"
            log_warning "2. Or using winget: 'winget install OpenJS.NodeJS'"
            log_warning "3. Or using Chocolatey: 'choco install nodejs'"
            ;;
        *)
            log_warning "Please install Node.js (includes npm): https://nodejs.org/en/download/"
            ;;
    esac
}

# Node.js installation recommendations
get_node_recommendation() {
    local system_id="$1"
    
    case "$system_id" in
        "macos")
            log_warning "Recommended installation for macOS: 'brew install node'"
            ;;
        "linux-debian")
            log_warning "Recommended installation for Debian/Ubuntu: 'sudo apt update && sudo apt install nodejs'"
            ;;
        "linux-fedora")
            log_warning "Recommended installation for Fedora: 'sudo dnf install nodejs'"
            ;;
        "linux-redhat")
            log_warning "Recommended installation for CentOS/RHEL: 'sudo yum install nodejs'"
            ;;
        "linux-arch")
            log_warning "Recommended installation for Arch Linux: 'sudo pacman -S nodejs'"
            ;;
        "linux-alpine")
            log_warning "Recommended installation for Alpine Linux: 'apk add nodejs'"
            ;;
        "linux-other")
            log_warning "Please install Node.js using your distribution's package manager"
            ;;
        "windows")
            log_warning "Recommended installation for Windows:"
            log_warning "1. Download from: https://nodejs.org/en/download/"
            log_warning "2. Or using winget: 'winget install OpenJS.NodeJS'"
            log_warning "3. Or using Chocolatey: 'choco install nodejs'"
            ;;
        *)
            log_warning "Please download Node.js from: https://nodejs.org/en/download/"
            ;;
    esac
}

# CMake installation recommendations
get_cmake_recommendation() {
    local system_id="$1"
    
    case "$system_id" in
        "macos")
            log_warning "Recommended installation for macOS: 'brew install cmake'"
            ;;
        "linux-debian")
            log_warning "Recommended installation for Debian/Ubuntu: 'sudo apt update && sudo apt install cmake'"
            ;;
        "linux-fedora")
            log_warning "Recommended installation for Fedora: 'sudo dnf install cmake'"
            ;;
        "linux-redhat")
            log_warning "Recommended installation for CentOS/RHEL: 'sudo yum install cmake'"
            ;;
        "linux-arch")
            log_warning "Recommended installation for Arch Linux: 'sudo pacman -S cmake'"
            ;;
        "linux-alpine")
            log_warning "Recommended installation for Alpine Linux: 'apk add cmake'"
            ;;
        "linux-other")
            log_warning "Please install CMake using your distribution's package manager"
            log_warning "Or download from: https://cmake.org/download/"
            ;;
        "windows")
            log_warning "Recommended installation for Windows:"
            log_warning "1. Download from: https://cmake.org/download/"
            log_warning "2. Or using winget: 'winget install Kitware.CMake'"
            log_warning "3. Or using Chocolatey: 'choco install cmake'"
            ;;
        *)
            log_warning "Please download CMake from: https://cmake.org/download/"
            ;;
    esac
}

# Poetry installation recommendations
get_poetry_recommendation() {
    local system_id="$1"
    
    case "$system_id" in
        "macos")
            log_warning "Recommended installation for macOS:"
            log_warning "1. 'brew install poetry'"
            log_warning "2. Or using the official installer: 'curl -sSL https://install.python-poetry.org | python3 -'"
            ;;
        "linux-debian")
            log_warning "Recommended installation for Debian/Ubuntu:"
            log_warning "1. Using pipx (recommended): 'sudo apt install pipx && pipx install poetry'"
            log_warning "2. Or using the official installer in your home directory:"
            log_warning "   'curl -sSL https://install.python-poetry.org | python3 -'"
            log_warning "3. Or in a virtual environment:"
            log_warning "   'python3 -m venv ~/.poetry-venv && ~/.poetry-venv/bin/pip install poetry'"
            log_warning "   Then add ~/.poetry-venv/bin to your PATH"
            ;;
        "linux-fedora")
            log_warning "Recommended installation for Fedora:"
            log_warning "1. 'sudo dnf install poetry'"
            log_warning "2. Or using the official installer: 'curl -sSL https://install.python-poetry.org | python3 -'"
            ;;
        "linux-redhat")
            log_warning "Recommended installation for CentOS/RHEL:"
            log_warning "1. Using the official installer: 'curl -sSL https://install.python-poetry.org | python3 -'"
            ;;
        "linux-arch")
            log_warning "Recommended installation for Arch Linux:"
            log_warning "1. 'sudo pacman -S python-poetry'"
            ;;
        "linux-alpine")
            log_warning "Recommended installation for Alpine Linux:"
            log_warning "1. 'apk add py3-poetry'"
            log_warning "2. Or using the official installer: 'curl -sSL https://install.python-poetry.org | python3 -'"
            ;;
        "linux-other")
            log_warning "Recommended installation for Linux:"
            log_warning "1. Using the official installer: 'curl -sSL https://install.python-poetry.org | python3 -'"
            ;;
        "windows")
            log_warning "Recommended installation for Windows:"
            log_warning "1. Using PowerShell: '(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -'"
            log_warning "2. Or using Chocolatey: 'choco install poetry'"
            ;;
        *)
            log_warning "Please install Poetry using the official installer:"
            log_warning "curl -sSL https://install.python-poetry.org | python3 -"
            ;;
    esac
}

# SQLite installation recommendations
get_sqlite_recommendation() {
    local system_id="$1"
    
    case "$system_id" in
        "macos")
            log_warning "Recommended installation for SQLite on macOS:"
            log_warning "1. 'brew install sqlite'"
            log_warning "SQLite is usually pre-installed on macOS, but this will ensure you have the latest version."
            ;;
        "linux-debian")
            log_warning "Recommended installation for SQLite on Debian/Ubuntu:"
            log_warning "1. 'sudo apt update && sudo apt install sqlite3'"
            ;;
        "linux-fedora")
            log_warning "Recommended installation for SQLite on Fedora:"
            log_warning "1. 'sudo dnf install sqlite'"
            ;;
        "linux-redhat")
            log_warning "Recommended installation for SQLite on CentOS/RHEL:"
            log_warning "1. 'sudo yum install sqlite'"
            ;;
        "linux-arch")
            log_warning "Recommended installation for SQLite on Arch Linux:"
            log_warning "1. 'sudo pacman -S sqlite'"
            ;;
        "linux-alpine")
            log_warning "Recommended installation for SQLite on Alpine Linux:"
            log_warning "1. 'apk add sqlite'"
            ;;
        "linux-other")
            log_warning "Recommended installation for SQLite on Linux:"
            log_warning "Please install SQLite using your distribution's package manager"
            ;;
        "windows")
            log_warning "Recommended installation for SQLite on Windows:"
            log_warning "1. Download from: https://www.sqlite.org/download.html"
            log_warning "2. Or using Chocolatey: 'choco install sqlite'"
            ;;
        *)
            log_warning "Please download SQLite from: https://www.sqlite.org/download.html"
            ;;
    esac
}
