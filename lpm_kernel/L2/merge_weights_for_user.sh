#!/bin/sh

python lpm_kernel/L2/merge_lora_weights.py  --base_model_path "${MODEL_BASE_PATH}"  --lora_adapter_path "${MODEL_PERSONAL_DIR}"  --output_model_path "${MODEL_MERGED_DIR}"