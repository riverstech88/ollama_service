# Ollama Service Updated

A Python client library for interacting with an [Ollama](https://ollama.com/) LLM backend. Provides model management, text generation, chat, and embedding capabilities with optional authentication via an Nginx reverse proxy.

## Dependencies

- Python 3.7+
- `requests`

Install:

```bash
pip install requests
```

## Quick Start

```python
from ollama_service_updated import OllamaService

with OllamaService(host="18.205.235.191", port=11434) as service:
    if service.is_healthy():
        print(service.list_model_names())
        print(service.generate_simple("mistral", "Hello!"))
```

## Constructor

```python
OllamaService(
    host: str = "18.205.235.191",
    port: int = 11434,
    timeout: int = 120,           # seconds
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    headers: Optional[dict] = None,
)
```

Supports context manager usage (`with` statement) for automatic session cleanup.

## Ollama API Endpoints

The service wraps the following Ollama REST endpoints:

| Endpoint | Method | Description | Wrapper Method |
|---|---|---|---|
| `/` | GET | Health check | `is_healthy()` |
| `/api/tags` | GET | List available models | `list_models()`, `list_model_names()` |
| `/api/show` | POST | Get model details | `show_model(model_name)` |
| `/api/generate` | POST | Generate text completion | `generate(request)`, `generate_simple(model, prompt, system)` |
| `/api/chat` | POST | Multi-turn chat | `chat(request)`, `chat_simple(model, user_message, system_message)` |
| `/api/embed` | POST | Generate embeddings | `embeddings(request)`, `get_embedding(model, text)` |
| `/api/ps` | GET | List running models | `list_running()` |
| `/api/pull` | POST | Download a model | `pull_model(model_name)` |
| `/api/delete` | DELETE | Delete a model | `delete_model(model_name)` |

## Interacting with the Endpoints

### Health Check

```python
service = OllamaService()
print(service.is_healthy())  # True / False
```

### List Models

```python
# Get model objects
models = service.list_models()  # list[ModelInfo]
for m in models:
    print(f"{m.name}  size={m.size}  digest={m.digest}")

# Get names only
names = service.list_model_names()  # list[str]
```

### Show Model Details

```python
detail = service.show_model("mistral:latest")
print(detail.parameters)
print(detail.template)
print(detail.details)
```

### Generate Text

Simple usage:

```python
response = service.generate_simple("mistral", "What is 2+2?")
print(response)  # plain string
```

With full control:

```python
from ollama_service_updated import GenerateRequest

req = GenerateRequest(
    model="mistral",
    prompt="Explain gravity in one sentence.",
    system="You are a physics teacher.",
    stream=False,
    options={"temperature": 0.3, "num_predict": 100},
)
resp = service.generate(req)  # GenerateResponse
print(resp.response)
print(resp.eval_count, resp.total_duration)
```

### Chat

Simple single-turn:

```python
reply = service.chat_simple("mistral", "What are you?")
print(reply)  # plain string
```

With system message:

```python
reply = service.chat_simple(
    "mistral",
    "Translate 'hello' to French.",
    system_message="You are a translator.",
)
```

Multi-turn conversation:

```python
from ollama_service_updated import ChatRequest

messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Hello Alice!"},
    {"role": "user", "content": "What is my name?"},
]
req = ChatRequest(model="mistral", messages=messages, stream=False)
resp = service.chat(req)  # ChatResponse
print(resp.message.content)
```

### Embeddings

Single text:

```python
vector = service.get_embedding("mistral", "Hello world")  # list[float]
print(len(vector))  # dimensionality
```

With request object:

```python
from ollama_service_updated import EmbeddingRequest

req = EmbeddingRequest(model="mistral", input=["text one", "text two"])
resp = service.embeddings(req)  # EmbeddingResponse
for emb in resp.embeddings:
    print(len(emb))
```

### Pull / Delete Models

```python
result = service.pull_model("llama2")
print(result.status)

success = service.delete_model("llama2")  # True / False
```

### List Running Models

```python
running = service.list_running()  # list[dict]
```

## Authentication

### No Auth (direct connection)

```python
service = OllamaService(host="18.205.235.191", port=11434)
```

### Bearer Token

```python
service = OllamaService(
    host="localhost", port=8080,
    api_key="ollama-key-001",
)
```

### Basic Auth

```python
service = OllamaService(
    host="localhost", port=8080,
    username="admin", password="ollama-admin-pass",
)
```

### Verify Authentication

```python
if service.verify_auth():
    print("Authenticated")
```

## Nginx Proxy (Optional)

An Nginx reverse proxy is included under `nginx/` to add authentication in front of the Ollama backend.

### Start the Proxy

```bash
cd ollama_service_updated/nginx
docker-compose up --build
```

This exposes the proxy on port `11434` (mapped from container port `8080`), forwarding to the Ollama backend at `18.205.235.191:11434`.

An unauthenticated `/health` endpoint is available for monitoring.

### Proxy Auth Methods

Requests must include **either** a valid bearer token **or** valid basic auth credentials. If a recognized bearer token is present, basic auth is bypassed.

## Error Handling

```python
from ollama_service_updated import (
    OllamaServiceError,       # base exception
    OllamaConnectionError,    # connection failure or timeout
    OllamaAPIError,           # non-200 HTTP response
    OllamaAuthError,          # 401 / 403
)

try:
    service.generate_simple("mistral", "Hello")
except OllamaConnectionError as e:
    print(f"Connection failed: {e}")
except OllamaAuthError as e:
    print(f"Auth failed: {e}")
except OllamaAPIError as e:
    print(f"HTTP {e.status_code}: {e.message}")
```

## Running Tests

```bash
pytest ollama_service_updated/tests/test_service.py -v
```

## Project Structure

```
ollama_service_updated/
    __init__.py          # Package exports
    service.py           # OllamaService class and exceptions
    models.py            # Request/response dataclasses
    nginx/
        nginx.conf       # Nginx proxy configuration
        Dockerfile       # Nginx Docker image
        docker-compose.yml
        .htpasswd        # Basic auth credentials
    tests/
        test_service.py  # Test suite
```