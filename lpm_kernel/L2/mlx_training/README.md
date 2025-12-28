# MLX Training Support for Apple Silicon

## Overview

We have integrated the MLX training framework for Apple Silicon to provide faster, more memory-efficient training capabilities. MLX is an array framework specifically designed for machine learning on Apple silicon, developed by Apple's machine learning research team.

**Framework Source**: [ml-explore/mlx](https://github.com/ml-explore/mlx)

While MLX is still under active development and doesn't yet offer the comprehensive features of frameworks like Transformers, it provides significant advantages for training on Apple Silicon devices.

## Performance

Through our testing, we've confirmed that on machines with 32GB of memory, MLX can support 4-bit quantized QLora fine-tuning of 7B parameter models. This represents substantial memory efficiency compared to traditional training approaches.

## Getting Started

### Prerequisites

Before starting, you'll need to install the required dependencies via terminal:

```bash
pip install mlx-lm
```

### Model Selection

You can choose from a variety of pre-trained models available in the [MLX Community on Hugging Face](https://huggingface.co/mlx-community). This community hosts ready-to-use models specifically optimized for Apple Silicon, including various quantized versions of popular models.

### Workflow

The training process involves the following steps:

1. **Data Conversion**: 
   Run the data conversion script to transform your previously processed training data into MLX-compatible format(run from the project root directory default):
   ```bash
   python lpm_kernel/L2/mlx_training/data_transform.py
   ```
   Before running the data conversion script, ensure that your raw data (`merged.json`) is located in the `resources/data/` directory. The converted data will be stored in the `resources/data/mlx_train_data` directory.
   Please verify that the username, COT mode, and data read/write paths are correctly configured in the data conversion script.
   You can customize the COT mode, username, and data paths in the script according to your preferences.

2. **Training**:
   Execute the MLX training script to fine-tune your model (run from the project root directory default):
   ```bash
   ./lpm_kernel/L2/mlx_training/train_by_mlx.sh
   ```
   You can modify the train_by_mlx.sh script to use your selected model from the MLX Community.

   You can start the training process using two methods: either by configuring the training parameters in a `.yaml` file or by specifying them directly in the command line. Both methods are demonstrated in the `train_by_mlx.sh` script. We recommend using the `.yaml` file method, especially for LoRA fine-tuning, as the LoRA parameters are only supported in the `.yaml` configuration.
   
   Additionally, if you encounter path errors during training, please verify that the paths in the `lora_config.yaml` file are correctly configured.


3. **Model Conversion and Serving**:
   Merge the adapter weights with the base model and start the model server (run from the project root directory):
   ```bash
   ./lpm_kernel/L2/mlx_training/convert_and_serve.sh
   ```
4. **Testing the Model**:
   After serving the model, you can test it to verify that it responds correctly:
   ```bash
   python lpm_kernel/L2/mlx_training/test_mlx.py
   ```
   
   This script sends a test request to the model server and displays the response. Note that the built-in prompt in the test script is configured for Felix Tao's Chain-of-Thought (COT) model. You should modify the prompt in the test script to match your specific training objectives and prompt format.
   
   Example of modifying the prompt in `test_mlx.py`:
   ```python
   payload = {
       "messages": [
           {
               "role": "system",
               "content": "Your custom system prompt here..."
           },
           {
               "role": "user",
               "content": "Your test question here"
           }
       ],
       "temperature": 0.7
   }
   ```

## Advantages

- Optimized performance on Apple Silicon (M1/M2/M3 chips)
- Reduced memory footprint
- Faster training times
- Support for quantized models

## Limitations

- MLX is still under development and lacks some features available in more established frameworks
- Limited to Apple Silicon devices
- Some model architectures may not be fully supported yet

## Future Work

We plan to continue enhancing our MLX integration as the framework matures, providing more features and improved performance over time.
