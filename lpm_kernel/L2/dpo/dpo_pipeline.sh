# dpo_pipeline.sh
python lpm_kernel/L2/dpo/dpo_data.py && \
python lpm_kernel/L2/dpo/dpo_train.py --num_train_epochs 2 --learning_rate 5e-6 --lora_r 32 --lora_alpha 64 --batch_size 4 && \
python lpm_kernel/L2/merge_lora_weights.py \
    --base_model_path "resources/model/output/merged_model" \
    --lora_adapter_path "resources/model/output/dpo_model/adapter" \
    --output_model_path "resources/model/output/dpo_model/merged_model"