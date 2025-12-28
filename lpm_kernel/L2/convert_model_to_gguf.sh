python convert_hf_to_gguf.py resources/model/output/merged_model --outfile resources/model/output/lpm.gguf --outtype f16

llama-server -m lpm.gguf --port 8080