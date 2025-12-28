#!/bin/bash
# Helper script to check if GPU support is available at runtime

echo "=== GPU Support Check ==="

# Check if llama-server binary exists and is linked to CUDA libraries
if [ -f "/app/llama.cpp/build/bin/llama-server" ]; then
    echo "llama-server binary found, checking for CUDA linkage..."
    CUDA_LIBS=$(ldd /app/llama.cpp/build/bin/llama-server | grep -i "cuda\|nvidia")
    
    if [ -n "$CUDA_LIBS" ]; then
        echo "✅ llama-server is built with CUDA support:"
        echo "$CUDA_LIBS"
        echo "GPU acceleration is available"
        
        # Check for GPU optimization marker file (optional, not required)
        GPU_MARKER_FILE="/app/data/gpu_optimized.json"
        if [ -f "$GPU_MARKER_FILE" ]; then
            GPU_OPTIMIZED=$(grep -o '"gpu_optimized": *true' "$GPU_MARKER_FILE" || echo "false")
            OPTIMIZED_DATE=$(grep -o '"optimized_on": *"[^"]*"' "$GPU_MARKER_FILE" | cut -d'"' -f4)
            
            if [[ "$GPU_OPTIMIZED" == *"true"* ]]; then
                echo "📝 GPU-optimized build marker found (built on: $OPTIMIZED_DATE)"
            else
                echo "📝 GPU marker file found but not marked as optimized (built on: $OPTIMIZED_DATE)"
            fi
        else
            echo "📝 No GPU optimization marker file found, but CUDA support is detected in binary"
        fi
        
        # Check if NVIDIA GPU is accessible at runtime
        if nvidia-smi &>/dev/null; then
            echo "🔍 NVIDIA GPU is available at runtime"
            echo "=== GPU ACCELERATION IS READY TO USE ==="
            exit 0
        else
            echo "⚠️ WARNING: llama-server has CUDA support, but NVIDIA GPU is not accessible"
            echo "Check that Docker is running with GPU access (--gpus all)"
            exit 1
        fi
    else
        echo "❌ llama-server is not linked with CUDA libraries"
        echo "Container was built without CUDA support"
    fi
else
    echo "❌ llama-server binary not found at /app/llama.cpp/build/bin/llama-server"
fi

# Final check for GPU hardware
if nvidia-smi &>/dev/null; then
    echo "🔍 NVIDIA GPU is available at runtime, but llama-server doesn't support CUDA"
    echo "To enable GPU support, rebuild using: make docker-up (and select CUDA support when prompted)"
    exit 1
else
    echo "❌ No NVIDIA GPU detected at runtime"
    exit 1
fi