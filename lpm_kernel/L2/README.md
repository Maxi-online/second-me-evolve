# Long Chain-of-Thought (CoT) Feature Implementation

## Overview

This implementation adds Long Chain-of-Thought (CoT) capability to the data synthesis pipeline when using DeepSeek R1 as the base model. The feature enables multi-step reasoning for enhanced context-aware responses.

## Feature Description

- **Long CoT Mode**: When enabled, the system generates synthetic data with extended reasoning chains
    
- **DeepSeek R1 Integration**: Exclusive use of DeepSeek-R1 model for CoT data generation
    
- **Enhanced Training**: Produces models with improved long-context reasoning capabilities
    

## Implementation Details

### Configuration Options

1. **Backend Configuration**:
    
    - Set `is_cot=True` in `trainprocess_service.py` initialization
        
    - Configure via `train_for_user.sh` with `--is_cot True/False`
        
    - Environment variables in `lpm_kernel/L2/.env`:
    ```
        DEEPSEEK_MODEL_NAME=deepseek-*
        
        DEEPSEEK_API_KEY=your_api_key
        
        DEEPSEEK_BASE_URL=your_base_url
    ```

### Data Synthesis Pipeline

1. **Supported Data Types**:
    
    - SelfQA data
        
    - Preference data
        
    - Diversity data
        
2. **Prompt Structure**:
```
	<think>reasoning_content</think>
    <answer>final_content</answer>
```
3. **Model Whitelisting**:
    
    - Only DeepSeek-R1 is allowed for CoT data generation

### Code Changes

1. **Modified Files**:
    
    - `selfqa.py`:
        
        - Added `is_cot` initialization option
            
        - Updated prompt templates
            
        - Modified response handling
            
    - `preference_QA_generate.py`:
        
        - Added CoT support
            
        - Enhanced question extraction
            
    - `diversity_data_generator.py`:
        
        - Added CoT templates
            
        - Updated generation logic
            
2. **New Functions**:
    
    - Unified `get_remote_response()` function
        
    - Enhanced logging with tqdm integration
        