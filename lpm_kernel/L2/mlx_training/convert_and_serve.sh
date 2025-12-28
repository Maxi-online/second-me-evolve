mlx_lm.fuse --model mlx-community/Qwen2.5-7B-Instruct-4bit \
--adapter-path "resources/model/output/mlx/adapters" \
--save-path "resources/model/output/mlx" 

echo "Adapter weights have been successfully fused to resources/model/output/mlx folder."

mlx_lm.server --model resources/model/output/mlx