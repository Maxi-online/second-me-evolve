#!/usr/bin/env python3

import torch
import subprocess
import sys
import os

print("=== PyTorch CUDA Version Information ===")
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA available: Yes")
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    
    # Try to check system CUDA version
    try:
        nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        print("\nSystem NVCC version:")
        print(nvcc_output)
    except:
        print("\nNVCC not found in PATH")
        
    # Check CUDA libraries
    try:
        print("\nChecking required CUDA libraries:")
        for lib in ["libcudart.so", "libcublas.so", "libcublasLt.so"]:
            print(f"\nSearching for {lib}:")
            find_result = subprocess.run(f"find /usr -name '{lib}*'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if find_result.returncode == 0 and find_result.stdout:
                print(find_result.stdout.decode("utf-8"))
            else:
                print(f"No {lib} found in /usr")
    except Exception as e:
        print(f"Error checking libraries: {e}")
        
    # Check LD_LIBRARY_PATH 
    print("\nLD_LIBRARY_PATH:")
    print(os.environ.get("LD_LIBRARY_PATH", "Not set"))
    
else:
    print("CUDA not available")
    
# Check system CUDA installation
print("\n=== System CUDA Information ===")
try:
    nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
    print("NVIDIA-SMI output:")
    print(nvidia_smi)
except:
    print("nvidia-smi not found or not working")