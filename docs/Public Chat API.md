# Chat Completion API

> **Fork notice (Maxim Strekolovskiy):** This documentation is upstream-oriented. In this fork, the server still exposes OpenAI-compatible streaming chat, but internally uses automatic A/B + LLM judge and stores RL logs in ClickHouse (required). See `NOTICE` and `README.md`.

## API Overview

This API is used to create chat completions, process provided messages, and generate responses. The API supports streaming responses and is compatible with OpenAI format.

## Prerequisites

Before using this API, you need to:

1. **Register**: Execute the register operation to create your instance
2. **Status Check**: Wait until your instance status becomes "online"
3. **Get Instance ID**: Obtain your unique `{instance_id}` from the registration response
4. **API Access**: Use the instance ID to construct the API endpoint: `https://app.secondme.io/api/chat/{instance_id}`

## API Endpoints

```
POST /api/chat/{instance_id}
POST /api/chat/{instance_id}/chat/completions
```

## Path Parameters

| Parameter | Type | Required | Description |
|------|------|------|------|
| `instance_id` | string | Yes | Unique identifier for the model instance, obtained during registration |

## Request Body

| Field | Type | Required | Default | Description |
|------|------|------|------|------|
| `messages` | array | Yes | - | List of messages in the conversation |
| `metadata` | object | No | null | Additional metadata for the request |
| `temperature` | float | No | 0.7 | Controls randomness of the response, value between 0 and 1 |
| `max_tokens` | integer | No | 2000 | Maximum number of tokens to generate |
| `stream` | boolean | No | true | Whether to stream the response |

### messages Field

Each message should contain the following fields:

| Field | Type | Required | Description |
|------|------|------|------|
| `role` | string | Yes | Role of the message sender. Can be 'system', 'user', or 'assistant' |
| `content` | string | Yes | Content of the message |

### metadata Field

| Field | Type | Required | Description |
|------|------|------|------|
| `enable_l0_retrieval` | boolean | No | Whether to enable L0 level retrieval |
| `role_id` | string | No | Role ID to use for this chat |

## Response

- Server-Sent Events (SSE) stream in OpenAI-compatible format
- Each event contains a fragment of the generated response
- The last event is marked as `[DONE]`

### Response Format Example

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"lpm-registry-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"lpm-registry-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"content":" world!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"lpm-registry-model","system_fingerprint":null,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Request Examples

### cURL

```bash
curl -X POST "https://app.secondme.io/api/chat/{instance_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, please introduce yourself."}
    ],
    "metadata": {
      "enable_l0_retrieval": false,
      "role_id": "default_role"
    },
    "temperature": 0.7,
    "max_tokens": 2000,
    "stream": true
  }'
```

### Python

```python
import http.client
import json

url = "app.secondme.io"
path = "/api/chat/{instance_id}"           

headers = {"Content-Type": "application/json"}
data = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, please introduce yourself."}
    ],
    "metadata": {
        "enable_l0_retrieval": False,
        "role_id": "default_role"
    },
    "temperature": 0.7,
    "max_tokens": 2000,
    "stream": True
}

# Prepare the connection
conn = http.client.HTTPSConnection(url)

# Send the POST request
conn.request("POST", path, body=json.dumps(data), headers=headers)

# Get the response
response = conn.getresponse()


# Read the body line by line
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

# Close the connection when done
conn.close()

```

## Error Codes

| Status Code | Description |
|------|------|
| 404 | Instance not found |
| 422 | Invalid request parameters |
| 503 | Instance not connected or unavailable |

## Notes

1. Before using this API, ensure that the instance is registered and connected to the server (status: "online")
2. The instance ID is unique and required for all API calls
3. For streaming responses, the client should be able to handle data in SSE format
4. Roles in the message list should follow the conversation order, typically starting with 'system' or 'user'
