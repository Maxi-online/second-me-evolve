#!/bin/bash
# Script to rebuild llama.cpp with CUDA support at runtime
# This ensures the build happens with full knowledge of the GPU environment

set -e  # Exit on error but don't print each command (for cleaner logs)
cd /app

echo "========== STARTING LLAMA.CPP CUDA REBUILD PROCESS =========="
echo "Current directory: $(pwd)"

# First check if CUDA is actually available in the container
echo "Verifying NVIDIA drivers and CUDA availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: NVIDIA drivers not found. Cannot build with CUDA support!"
    echo "Make sure the container has access to the GPU and NVIDIA Container Toolkit is installed."
    echo "Consider running Docker with: --gpus all"
    exit 0  # Exit without error as there's no point trying to build with CUDA when no GPU is detected
fi

# Run nvidia-smi to check GPU access
echo "Detected NVIDIA GPU:"
nvidia-smi || {
    echo "ERROR: nvidia-smi command failed. GPU is not properly accessible from the container."
    echo "Make sure you're running Docker with GPU access enabled (--gpus all)"
    exit 0  # Exit without error since there's no GPU access
}

# Install build dependencies
echo "Installing build dependencies..."
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    cmake \
    git \
    ca-certificates \
    gnupg \
    libopenblas-dev

# Clean up apt cache to free space
apt-get clean
rm -rf /var/lib/apt/lists/*

# Install CUDA using NVIDIA's official Debian 12 network installation method
echo "Installing CUDA using NVIDIA's official method for Debian 12..."
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
apt-get update

# Install CUDA packages needed for building llama.cpp with CUDA support
apt-get install -y --fix-missing --no-install-recommends cuda-compiler-12-8
apt-get clean
rm -rf /var/lib/apt/lists/*

apt-get update
apt-get install -y --fix-missing --no-install-recommends cuda-runtime-12-8
apt-get clean
rm -rf /var/lib/apt/lists/*

apt-get update
apt-get install -y --fix-missing --no-install-recommends cuda-libraries-dev-12-8
apt-get clean
rm -rf /var/lib/apt/lists/*

# Set up environment for build
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.8
# Set CUDACXX environment variable explicitly to help CMake find the CUDA compiler
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc

# Verify CUDA compiler is available
echo "Verifying CUDA compiler (nvcc) is available:"
which nvcc || echo "ERROR: nvcc not found in PATH!"
nvcc --version || echo "ERROR: nvcc not working properly!"

echo "CUDA environment:"
echo "- CUDA_HOME: $CUDA_HOME"
echo "- CUDACXX: $CUDACXX"
echo "- CMAKE_CUDA_COMPILER: $CMAKE_CUDA_COMPILER"
echo "- PATH includes CUDA: $PATH"
echo "- LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Show available disk space
echo "Available disk space:"
df -h

# Use local build approach to avoid volume mount issues
echo "Building llama.cpp with CUDA in a local directory..."
cd /tmp
rm -rf llama_build
mkdir -p llama_build
cd llama_build

# Clone a fresh copy of llama.cpp - this avoids volume mount issues
echo "Cloning fresh copy of llama.cpp..."
git clone https://github.com/ggerganov/llama.cpp.git .

# Configure and build with CUDA support
mkdir -p build
cd build
echo "Configuring with CMake..."
cmake -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=all \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_NATIVE=OFF \
      -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets" \
      ..

echo "Building llama.cpp with CUDA support..."
cmake --build . --config Release --target all -j $(nproc)

if [ -f "bin/llama-server" ]; then
    echo "Build successful! Copying binaries to /app/llama.cpp/build/bin/"
    mkdir -p /app/llama.cpp/build/bin
    cp bin/llama-server /app/llama.cpp/build/bin/
    cp bin/llama-cli /app/llama.cpp/build/bin/ 2>/dev/null || true
    chmod +x /app/llama.cpp/build/bin/llama-server /app/llama.cpp/build/bin/llama-cli
    
    # Create GPU optimized marker
    echo "{ \"gpu_optimized\": true, \"optimized_on\": \"$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")\" }" > /app/data/gpu_optimized.json
    
    echo "Testing CUDA support in built binary..."
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH /app/llama.cpp/build/bin/llama-server --version
    echo ""
    echo "========== CUDA BUILD COMPLETED SUCCESSFULLY =========="
else
    echo "ERROR: Build failed - llama-server executable not found!"
    exit 1
fi