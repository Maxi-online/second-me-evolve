# Custom Model Endpoint Guide with Ollama

## 1. Prerequisites: Ollama Setup

First, download and install Ollama from the official website:

🔗 **Download Link**: [https://ollama.com/download](https://ollama.com/download)

📚 **Additional Resources**:
- Official Website: [https://ollama.com](https://ollama.com/)
- Model Library: [https://ollama.com/library](https://ollama.com/library)
- GitHub Repository: [https://github.com/ollama/ollama/](https://github.com/ollama/ollama)

---

## 2. Basic Ollama Commands

| Command | Description |
|------|------|
| `ollama pull model_name` | Download a model |
| `ollama serve` | Start the Ollama service |
| `ollama ps` | List running models |
| `ollama list` | List all downloaded models |
| `ollama rm model_name` | Remove a model |
| `ollama show model_name` | Show model details |

## 3. Using Ollama API for Custom Model

### OpenAI-Compatible API


#### Chat Request

```bash
curl http://127.0.0.1:11434/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen2.5:0.5b",
  "messages": [
    {"role": "user", "content": "Why is the sky blue?"}
  ]
}'
```

#### Embedding Request

```bash
curl http://127.0.0.1:11434/v1/embeddings -d '{
  "model": "snowflake-arctic-embed:110m",
  "input": "Why is the sky blue?"
}'
```

More Details: [https://github.com/ollama/ollama/blob/main/docs/openai.md](https://github.com/ollama/ollama/blob/main/docs/openai.md)

## 4. Configuring Custom Embedding in Second Me

1. Start the Ollama service: `ollama serve`
2. Check your Ollama embedding model context length:

```bash
# Example: ollama show snowflake-arctic-embed:110m
$ ollama show snowflake-arctic-embed:110m

Model
  architecture        bert       
  parameters          108.89M    
  context length      512        
  embedding length    768        
  quantization        F16        

License
  Apache License               
  Version 2.0, January 2004
```

3. Modify `EMBEDDING_MAX_TEXT_LENGTH` in `Second_Me/.env` to match your embedding model's context window. This prevents chunk length overflow and avoids server-side errors (500 Internal Server Error).

```bash
# Embedding configurations

EMBEDDING_MAX_TEXT_LENGTH=embedding_model_context_length
```

4. Configure Custom Embedding in Settings

```
Chat:
Model Name: qwen2.5:0.5b
API Key: ollama
API Endpoint: http://127.0.0.1:11434/v1

Embedding:
Model Name: snowflake-arctic-embed:110m
API Key: ollama
API Endpoint: http://127.0.0.1:11434/v1
```

**When running Second Me in Docker environments**, please replace `127.0.0.1` in API Endpoint with `host.docker.internal`:

```
Chat:
Model Name: qwen2.5:0.5b
API Key: ollama
API Endpoint: http://host.docker.internal:11434/v1

Embedding:
Model Name: snowflake-arctic-embed:110m
API Key: ollama
API Endpoint: http://host.docker.internal:11434/v1