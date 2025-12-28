# MCP Content

## Installation and operation

### Installation dependencies
Please ensure that Python version 3.7 or above is installed in the system. You need to install the following dependencies:

```bash
pip install mcp
```
### Configuration File
The tool manages server and client settings through a configuration file. Below is an example structure of the `config.json` file:

```json
{
  "mcpServers": {
    "mindverse": {
      "command": "python",
      "args": ["{replace-with-your-path}/Second-Me/mcp/mcp_public.py"]
    }
  }
}
```
### Configuration Items Explanation

#### Server Configuration (object)
- `command` (string):  
  The command to execute the script (typically the Python interpreter)
  
- `args` (array):  
  Arguments passed to the script, including:
  - Path to the Python script (replace `{replace-with-your-path}` with your actual path)

### Input Parameter Explanation
The `get_response` function has two input parameters:

`query`: This is the user input query that the tool will use to request the model. For example, the user might ask questions like "How is the weather?" or "What can you do for me?"

`instance_id`: This is a string used to identify a specific model instance. Typically, `instance_id` might be the model's URL path or a unique identifier used to specify an instance on the model platform.

## Response

- Server-Sent Events (SSE) stream in OpenAI-compatible format
- Each event contains a fragment of the generated response
- The last event is marked as `[DONE]`

### Response Format Example

```

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"lpm-registry-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"content":" world!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"lpm-registry-model","system_fingerprint":null,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Note
Currently, MCP services automatically process streaming data to generate coherent paragraphs. If you need RAW results, you can directly return a 'response' in the code. You can change it yourself according to the following code.

```
@mindverse.tool()
async def get_response(query:str, instance_id:str) -> HTTPResponse:
    """
    Received a response based on public mindverse model.

    Args:
        query (str): Questions raised by users regarding the mindverse model.
        instance_id (str): ID used to identify the mindverse model, or url used to identify the mindverse model.

    """
    id = instance_id.split('/')[-1]
    path = f"/api/chat/{id}"
    headers = {"Content-Type": "application/json"}
    messages.append({"role": "user", "content": query})

    data = {
        "messages": messages,
        "metadata": {
        "enable_l0_retrieval": False,
        "role_id": "default_role"
    },
    "temperature": 0.7,
    "max_tokens": 2000,
    "stream": True
    }

    conn = http.client.HTTPSConnection(url)

    # Send the POST request
    conn.request("POST", path, body=json.dumps(data), headers=headers)

    # Get the response
    response = conn.getresponse()
    return response
```