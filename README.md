# Ollama Service — Usage Guide

This guide explains how to interact with the Ollama LLM endpoint for everyday tasks. All examples assume the server is already running at `http://18.205.235.191:11434`.

## What is Ollama?

[Ollama](https://ollama.com/) is an open-source platform for running large language models. It hosts open-weight models (Mistral, Llama, Gemma, DeepSeek, Qwen, Dolphin, and others) behind a simple REST API. There are no API keys or per-token fees — the models run on infrastructure we control and are available to the team over HTTP.

## Available Models

To see which models are loaded on the server:

```bash
curl http://18.205.235.191:11434/api/tags
```

Common models on this server include `mistral`, `llama3.1`, `gemma2`, `deepseek-r1`, `qwen2.5`, `dolphin-llama3`, and `dolphin-phi`. Any of these can be used in the examples below.

## Using the Endpoint with curl

Every request goes to `http://18.205.235.191:11434`. The two endpoints you will use most are `/api/generate` (one-shot prompt in, text out) and `/api/chat` (conversational messages in, reply out).

### Generate Text (`/api/generate`)

Send a prompt, get a completion back. Set `"stream": false` to receive the full response in one JSON object.

```bash
curl http://18.205.235.191:11434/api/generate \
  -d '{
    "model": "mistral",
    "prompt": "Explain what a reverse proxy is in two sentences.",
    "stream": false
  }'
```

The response JSON contains a `response` field with the generated text.

#### Adding a System Instruction

Use the `system` field to steer the model's behaviour:

```bash
curl http://18.205.235.191:11434/api/generate \
  -d '{
    "model": "mistral",
    "prompt": "What causes a segmentation fault?",
    "system": "You are a systems programming tutor. Keep answers concise.",
    "stream": false
  }'
```

#### Tuning Generation Parameters

Pass `options` to control temperature, token limit, and other settings:

```bash
curl http://18.205.235.191:11434/api/generate \
  -d '{
    "model": "mistral",
    "prompt": "List 5 names for a coffee shop.",
    "stream": false,
    "options": {"temperature": 0.9, "num_predict": 200}
  }'
```

| Option | What it does |
|---|---|
| `temperature` | Creativity. Lower (e.g. 0.1) = focused, higher (e.g. 1.0) = varied. |
| `num_predict` | Maximum number of tokens to generate. |
| `top_p` | Nucleus sampling threshold (0.0–1.0). |
| `top_k` | Limits token choices to the top-k most likely. |

### Chat (`/api/chat`)

Use `/api/chat` when you want conversational interaction. Messages use the familiar `role`/`content` format.

#### Single Question

```bash
curl http://18.205.235.191:11434/api/chat \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "user", "content": "Write a Python function that checks if a number is prime."}
    ],
    "stream": false
  }'
```

#### With a System Role

```bash
curl http://18.205.235.191:11434/api/chat \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "system", "content": "You are a code reviewer. Point out bugs and suggest improvements."},
      {"role": "user", "content": "Review this: def add(a, b): return a - b"}
    ],
    "stream": false
  }'
```

#### Multi-Turn Conversation

Include the full message history so the model has context:

```bash
curl http://18.205.235.191:11434/api/chat \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "user", "content": "My project uses FastAPI and PostgreSQL."},
      {"role": "assistant", "content": "Got it. How can I help with your project?"},
      {"role": "user", "content": "Write an endpoint that returns all users from the database."}
    ],
    "stream": false
  }'
```

### Generate Embeddings (`/api/embed`)

Get a vector representation of text, useful for semantic search or similarity comparison.

```bash
curl http://18.205.235.191:11434/api/embed \
  -d '{
    "model": "mistral",
    "input": "Kubernetes pod scheduling"
  }'
```

Multiple texts in one call:

```bash
curl http://18.205.235.191:11434/api/embed \
  -d '{
    "model": "mistral",
    "input": ["first document", "second document", "third document"]
  }'
```

### Get Model Details (`/api/show`)

Inspect a model's parameters, template, and architecture:

```bash
curl http://18.205.235.191:11434/api/show \
  -d '{"name": "mistral"}'
```

### List Running Models (`/api/ps`)

See which models are currently loaded in memory:

```bash
curl http://18.205.235.191:11434/api/ps
```

## Daily Task Examples

### Draft an Email

```bash
curl http://18.205.235.191:11434/api/generate \
  -d '{
    "model": "mistral",
    "prompt": "Draft a professional email to a client apologising for a delayed deployment and proposing a new timeline.",
    "system": "You write clear, professional business emails.",
    "stream": false
  }'
```

### Explain or Debug Code

```bash
curl http://18.205.235.191:11434/api/chat \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "system", "content": "You are a senior developer. Explain code clearly and suggest fixes."},
      {"role": "user", "content": "Why does this fail?\n\nimport json\ndata = json.loads(None)"}
    ],
    "stream": false
  }'
```

### Generate a SQL Query

```bash
curl http://18.205.235.191:11434/api/generate \
  -d '{
    "model": "mistral",
    "prompt": "Write a PostgreSQL query that returns the top 10 customers by total order value in the last 30 days. Tables: customers(id, name), orders(id, customer_id, total, created_at).",
    "stream": false
  }'
```

### Summarise Text

```bash
curl http://18.205.235.191:11434/api/generate \
  -d '{
    "model": "mistral",
    "prompt": "Summarise the following in 3 bullet points:\n\nKubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerised applications. It groups containers into logical units for easy management and discovery.",
    "stream": false
  }'
```

### Write Unit Tests

```bash
curl http://18.205.235.191:11434/api/chat \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "system", "content": "You write pytest unit tests with good edge-case coverage."},
      {"role": "user", "content": "Write tests for this function:\n\ndef divide(a, b):\n    if b == 0:\n        raise ValueError(\"Cannot divide by zero\")\n    return a / b"}
    ],
    "stream": false
  }'
```

### Translate Text

```bash
curl http://18.205.235.191:11434/api/chat \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "system", "content": "You are a translator. Translate the user message to French."},
      {"role": "user", "content": "The deployment was successful and all services are running normally."}
    ],
    "stream": false
  }'
```

### Generate Documentation

```bash
curl http://18.205.235.191:11434/api/chat \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "system", "content": "You write clear, concise docstrings in Google style."},
      {"role": "user", "content": "Write a docstring for this function:\n\ndef retry(fn, max_attempts=3, delay=1.0):\n    for attempt in range(max_attempts):\n        try:\n            return fn()\n        except Exception:\n            if attempt == max_attempts - 1:\n                raise\n            time.sleep(delay)"}
    ],
    "stream": false
  }'
```

## Tips

- **Choose the right endpoint.** Use `/api/generate` for single prompts (summaries, drafts, one-off questions). Use `/api/chat` when you need role-based instructions or multi-turn context.
- **Use system messages.** They are the most effective way to control tone, format, and scope of the response.
- **Keep `"stream": false`** for scripting and automation. This returns the entire response in a single JSON object. Omit it or set to `true` if you want token-by-token streaming.
- **Adjust temperature.** Low values (0.1–0.3) for factual or deterministic output. Higher values (0.7–1.0) for creative tasks.
- **Try different models.** `mistral` is a good general-purpose default. `deepseek-r1` is strong at reasoning. `dolphin-llama3` is uncensored and permissive.

---

## Python Client

This repository also provides a Python wrapper around the same endpoints.

### Install

```bash
pip install requests
```

### Quick Start

```python
from ollama_service_updated import OllamaService

service = OllamaService()

# List models
print(service.list_model_names())

# Generate text
print(service.generate_simple("mistral", "What is 2+2?"))

# Chat
print(service.chat_simple("mistral", "Explain Docker in one sentence."))

# Chat with a system message
print(service.chat_simple(
    "mistral",
    "Translate 'hello' to French.",
    system_message="You are a translator.",
))

# Embeddings
vector = service.get_embedding("mistral", "Hello world")

# Close when done
service.close()
```

The service also works as a context manager:

```python
with OllamaService() as service:
    reply = service.chat_simple("mistral", "What are you?")
    print(reply)
```

### Advanced Generate

```python
from ollama_service_updated import OllamaService, GenerateRequest

service = OllamaService()
req = GenerateRequest(
    model="mistral",
    prompt="Explain gravity in one sentence.",
    system="You are a physics teacher.",
    stream=False,
    options={"temperature": 0.3, "num_predict": 100},
)
resp = service.generate(req)
print(resp.response)
print(resp.eval_count, resp.total_duration)
```

### Advanced Chat (Multi-Turn)

```python
from ollama_service_updated import OllamaService, ChatRequest

service = OllamaService()
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Hello Alice!"},
    {"role": "user", "content": "What is my name?"},
]
req = ChatRequest(model="mistral", messages=messages, stream=False)
resp = service.chat(req)
print(resp.message.content)
```

### Authentication

If accessing through the Nginx proxy, pass credentials when creating the service:

```python
# Bearer token
service = OllamaService(host="localhost", port=8080, api_key="ollama-key-001")

# Basic auth
service = OllamaService(host="localhost", port=8080, username="admin", password="ollama-admin-pass")
```

### Error Handling

```python
from ollama_service_updated import (
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

### API Endpoint Mapping

| Endpoint | Method | Python Method |
|---|---|---|
| `/api/tags` | GET | `list_models()`, `list_model_names()` |
| `/api/show` | POST | `show_model(name)` |
| `/api/generate` | POST | `generate(request)`, `generate_simple(model, prompt, system)` |
| `/api/chat` | POST | `chat(request)`, `chat_simple(model, message, system_message)` |
| `/api/embed` | POST | `embeddings(request)`, `get_embedding(model, text)` |
| `/api/ps` | GET | `list_running()` |
| `/api/pull` | POST | `pull_model(name)` |
| `/api/delete` | DELETE | `delete_model(name)` |