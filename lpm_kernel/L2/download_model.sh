#!/bin/bash
# Script to download model from Hugging Face
# Usage: ./download_model.sh [model_name]
#   If no model name is provided, will attempt to get it from config

if [ "$1" != "" ]; then
    # Use provided model name
    python lpm_kernel/L2/utils.py "$1"
else
    # No model name provided, let utils.py determine from config
    python lpm_kernel/L2/utils.py
fi
