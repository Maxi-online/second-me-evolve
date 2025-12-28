# DPO Workflow

## Overview

The DPO (Direct Preference Optimization) workflow is a systematic process designed to optimize models based on preference signals. The entire workflow consists of the following key stages:

1. **SFT Model Deployment**: Deploy the SFT (Supervised Fine-Tuning) model using `llama.cpp` and expose it via an API endpoint.
2. **DPO Data Synthesis**: Generate training data tailored for DPO by leveraging the deployed SFT model.
3. **Model Training**: Train the model using the synthesized DPO data with configurable hyperparameters.
4. **Merge Weights**: Merge Adapter Weights with the Base Model.

This document provides a comprehensive guide to executing the DPO workflow, both automatically and manually.

---

## Getting Started

### Setup

Before executing the subsequent steps, please ensure that your API key and base URL are configured in `lpm_kernel/L2/dpo/utils.py`. Additionally, you need to manually fill in the global bio, including your interests, occupation, etc.

### Automatically

To execute the DPO workflow automatically, follow these steps:

1. Navigate to the project root directory in your terminal.
2. Deploy your sft model by using llama.cpp.Before deploy the model, you should convert it into gguf format. Ensure the model is accessible via an API endpoint on port 8080.
```bash
    # Example command to run the SFT model with llama.cpp
    ./llama.cpp/build/llama_server --model /path/to/sft_model --port 8080
```
2. Run the following command in another terminal(also in the project root directory):
```bash
   bash lpm_kernel/L2/dpo/dpo_pipeline.sh
```
3. Before execution, ensure the following prerequisites are met:
   - Verify that the personal model path is correctly configured in the script.
   - Confirm that the SFT training dataset is saved in the designated location.

This script encapsulates all the necessary steps for the DPO workflow, streamlining the process for users.


### Manually

For users who prefer granular control, the DPO workflow can be executed step-by-step as follows:

#### Step 1: Deploy SFT Model

Before synthesizing DPO training data, deploy the SFT model using llama.cpp. Before deploy the model, you should convert it into gguf format. Ensure the model is accessible via an API endpoint on port 8080.


```bash
# Example command to run the SFT model with llama.cpp
./llama.cpp/build/llama_server --model /path/to/sft_model --port 8080
```

#### Step 2: Synthesize DPO Data

Once the SFT model is deployed, proceed to generate DPO training data.

```bash
python lpm_kernel/L2/dpo/dpo_data.py
```

This script synthesizes the required data for DPO training.

#### Step 3: Train the Model

After completing the data synthesis, train the model with the following command:

```bash
python lpm_kernel/L2/dpo/dpo_train.py \
    --num_train_epochs 2 \
    --learning_rate 5e-6 \
    --lora_r 32 \
    --lora_alpha 64 \
    --batch_size 4
```

This command initiates the training process with specified hyperparameters. Adjust these parameters as needed for optimal results.

#### Step 4: Merge the Weights

After training, merge the adapter weights with the base model using the following command(If you use Lora to dpo):

```bash
python lpm_kernel/L2/merge_lora_weights.py \
--base_model_path "resources/model/output/merged_model" \
--lora_adapter_path "resources/model/output/dpo_model/adapter" \
--output_model_path "resources/model/output/dpo_model/merged_model"
```

But if you do not use lora, you can skip this step.

Additional Notes
- For manual execution, ensure each step is completed successfully before proceeding to the next.
- Refer to the respective script documentation for additional configuration options and troubleshooting tips.

Feel free to expand upon this framework with specific details pertinent to your use case.