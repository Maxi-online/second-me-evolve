# Chat API Documentation

> **Fork notice (Maxim Strekolovskiy):** This documentation applies to this fork. The chat endpoint remains compatible with OpenAI Chat Completions, but responses are selected via automatic A/B + judge, and RL logs go to ClickHouse (required). See `NOTICE` and `README.md`.

## Overview

This API provides chat functionality compatible with OpenAI V1 Chat Completions API, supporting streaming responses for interactive conversations with AI assistants.

## API Details

- **URL**: `/api/kernel2/chat`
- **Method**: POST
- **Description**: Chat interface - Streaming response (compatible with OpenAI V1 API)
- **Access**: Available through local endpoint at `localhost:8002`

## Request Parameters

The request body is compatible with OpenAI Chat Completions API, using JSON format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who are you?"},
    {"role": "assistant", "content": "I am a helpful assistant."},
    {"role": "user", "content": "What can you do for me?"}  
  ],
  "metadata": {
    "enable_l0_retrieval": true,
    "role_id": "uuid-string"
  },
  "stream": true,
  "model": "gpt-3.5-turbo",
  "temperature": 0.1,
  "max_tokens": 2000
}
```

### Parameter Description

| Parameter | Type | Required | Description |
|------|------|------|------|
| messages | Array | Yes | Standard OpenAI message list containing conversation history |
| metadata | Object | No | Additional parameters for request processing |
| metadata.enable_l0_retrieval | Boolean | No | Whether to enable basic knowledge retrieval |
| metadata.role_id | String | No | System customized role UUID |
| stream | Boolean | No | Whether to return streaming response (default: true) |
| model | String | No | Model identifier (default: configured model) |
| temperature | Float | No | Controls randomness (default: 0.1) |
| max_tokens | Integer | No | Maximum number of tokens to generate (default: 2000) |

## Response Format

The response format is compatible with OpenAI Chat Completions API, using Server-Sent Events (SSE) format for streaming responses:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1677652288,
  "model": "gpt-3.5-turbo",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "delta": {"content": "Hello"},
      "finish_reason": null
    }
  ]
}
```

### Format of Each Chunk in Streaming Response

| Field | Type | Description |
|------|------|------|
| id | String | Unique identifier for the response |
| object | String | Fixed as "chat.completion.chunk" |
| created | Integer | Timestamp |
| model | String | Model identifier |
| system_fingerprint | String | System fingerprint |
| choices | Array | List of generated results |
| choices[0].index | Integer | Result index, usually 0 |
| choices[0].delta | Object | Incremental content of the current chunk |
| choices[0].delta.content | String | Incremental text content |
| choices[0].finish_reason | String | Reason for completion, null or "stop" |

## Usage Examples

### cURL Request Example

```bash
curl -X POST \
  'http://localhost:8002/api/kernel2/chat' \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about artificial intelligence."}
    ],
    "stream": true
  }'
```

### Python Request Example

```python
import json
import http.client

url = "localhost:8002"
path = "/api/kernel2/chat"
headers = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream"
}
data = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about artificial intelligence."}
    ],
    "stream": True
}

conn = http.client.HTTPConnection(url)

conn.request("POST", path, body=json.dumps(data), headers=headers)

response = conn.getresponse()

for line in response:
    if line:
        decoded_line = line.decode('utf-8').strip()
        if decoded_line == 'data: [DONE]':
            break
        if decoded_line.startswith('data: '):
            try:
                json_str = decoded_line[6:]
                chunk = json.loads(json_str)
                content = chunk['choices'][0]['delta'].get('content', '')
                if content:
                    print(content, end='', flush=True)
            except json.JSONDecodeError:
                pass

conn.close()
```

## Error Handling

When an error occurs, the API will return standard HTTP error status codes and error details in JSON format:

```json
{
  "success": false,
  "message": "Error message",
  "code": 400
}
```

| Error Code | Description |
|------|------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Internal Server Error |
