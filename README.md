# LocalMind

A lightweight, self-hosted chat interface for a local [LM Studio](https://lmstudio.ai/) server.

LocalMind is a small FastAPI application with a vanilla HTML/CSS/JS frontend. It talks to LM Studio's OpenAI-compatible API and gives you:

- **Dynamic model selection** — the dropdown is populated with the models currently loaded in LM Studio.
- **Real-time streaming** — responses render token-by-token via Server-Sent Events.
- **Parameter controls** — adjust `temperature` and `max_tokens` from a settings panel before sending.
- **Model management** — load and unload models straight from the UI (📦 panel), with:
  - **idle TTL (auto-unload)** — automatically unload a model after it has been idle for a configurable number of seconds,
  - **context length override** — load a model with a custom context window,
  - **live status** — loaded/not-loaded badges, parameter count, quantization, on-disk size, max context, and an estimate of memory in use, auto-refreshed while the panel is open,
  - **multiple instances** — load a model more than once and unload each instance individually,
  - **unload all** — free all memory with one click.
- **Image generation (capability-detected)** — a 🎨 mode in the composer that generates images from text prompts via an OpenAI-compatible `/images/generations` endpoint. The UI detects automatically whether the connected server supports it; generated images render in the chat with click-to-zoom and a download link, and the image size is adjustable in settings.
- **Robust error handling** — a clear UI banner (with retry) when LM Studio is offline, unreachable, or has no model loaded; toast notifications for load/unload results.
- **Simple configuration** — a single `config.json`, with safe fallback to `config.template.json` and built-in defaults.

## Directory Structure

```
LocalMind/
├── app.py                  # FastAPI backend (config loading, model list, streaming chat)
├── config.template.json    # Configuration template — copy to config.json
├── requirements.txt        # Python dependencies
├── static/
│   ├── index.html          # Chat UI
│   ├── style.css           # Styling (dark, responsive)
│   └── script.js           # Frontend logic (models, settings, SSE streaming)
├── .gitignore
├── LICENSE
└── README.md
```

## Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) installed, with:
  - at least one model downloaded and loaded, and
  - the local server running (**Developer** tab → **Start Server**, default `http://localhost:1234`).

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/ramhee98/LocalMind.git
cd LocalMind
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your configuration

Copy the template and adjust it to your environment:

```bash
cp config.template.json config.json
```

| Key | Description | Default |
| --- | --- | --- |
| `lm_studio_base_url` | Base URL of the LM Studio OpenAI-compatible API | `http://localhost:1234/v1` |
| `api_key` | API key sent to LM Studio (any non-empty string works) | `lm-studio` |
| `host` | Interface the web app binds to | `0.0.0.0` |
| `port` | Port the web app listens on | `8000` |
| `request_timeout_seconds` | Timeout for requests to LM Studio | `120` |
| `defaults.temperature` | Initial temperature in the UI | `0.7` |
| `defaults.max_tokens` | Initial max tokens in the UI | `1024` |
| `defaults.system_prompt` | System prompt prepended to every conversation | `You are a helpful assistant.` |
| `model_management.default_ttl_seconds` | Default idle TTL applied when loading a model from the UI | `600` |
| `model_management.auto_unload_by_default` | Whether the "auto-unload after idle" checkbox starts enabled | `true` |
| `model_management.default_context_length` | Default context length for loads (`null` = model default) | `null` |
| `model_management.load_timeout_seconds` | How long to wait for a model load to complete | `600` |
| `model_management.status_refresh_seconds` | Auto-refresh interval of the model panel while open | `10` |
| `image_generation.api_base_url` | OpenAI-compatible image API base URL (`null` = use `lm_studio_base_url`) | `null` |
| `image_generation.model` | Model name sent to the image API (`null` = let the server choose) | `null` |
| `image_generation.default_size` | Initial image size in the UI | `1024x1024` |
| `image_generation.timeout_seconds` | Timeout for image generation requests | `300` |

`config.json` is gitignored, so your local settings never end up in version control. If it is missing, the app falls back to `config.template.json`, and finally to built-in defaults.

### 4. Run the app

```bash
python app.py
```

or, for development with auto-reload:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000> in your browser.

## Usage

1. Pick a model from the dropdown (populated live from LM Studio).
2. Optionally open the ⚙ settings panel to tune temperature and max tokens.
3. Type a message and press **Enter** (Shift+Enter inserts a newline).
4. Watch the response stream in token-by-token. Use 🗑 to start a fresh conversation.

If LM Studio is not running, a banner explains the problem and offers a **Retry** button.

### Managing models

Open the 📦 panel to see every model downloaded in LM Studio with its load state and details.

- **Load** a model with the options set at the top of the panel:
  - *Auto-unload after idle* + *TTL (seconds)* — the model is automatically unloaded by LM Studio once it has been idle for that long. Uncheck to keep it loaded indefinitely.
  - *Context length* — leave blank for the model's default, or set a custom context window.
- **Unload** any loaded instance individually, or use **Unload all** to free all memory at once.
- **Load another** creates an additional instance of an already-loaded model (e.g. with a different context length).
- The panel auto-refreshes while open, so changes made in LM Studio itself (or TTL expirations) show up automatically.

The chat model dropdown always reflects the currently loaded LLM instances and refreshes after every load/unload.

> Model management uses LM Studio's native REST API (`/api/v1/models`), which requires a recent LM Studio version. Chat keeps working through the OpenAI-compatible API even if the native API is unavailable.

### Generating images

Click the 🎨 button in the composer to switch to image mode, describe the image, and press **Enter** — the result appears in the chat (click to zoom, ⬇ to download). Pick the output size in the ⚙ settings panel.

The button activates only when the connected server actually supports image generation, which is detected at startup by probing the OpenAI-compatible `/images/generations` endpoint:

- **LM Studio does not currently support image generation** (its models are text/vision-input only), so against a plain LM Studio setup the 🎨 button stays greyed out and explains why when clicked.
- To enable it, run an OpenAI-compatible image server (e.g. [LocalAI](https://localai.io/) with a Stable Diffusion backend, or any gateway exposing `/v1/images/generations`) and point `image_generation.api_base_url` at it in `config.json`, e.g.:

```json
"image_generation": {
  "api_base_url": "http://localhost:8080/v1",
  "model": "stablediffusion",
  "default_size": "1024x1024",
  "timeout_seconds": 300
}
```

Chat continues to use LM Studio regardless — the image API is fully independent. Image prompts are not added to the LLM chat history.

## Deployment Notes

The app is a standard ASGI application and runs anywhere uvicorn does.

**Linux (systemd-style):**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

**Docker:** a minimal image only needs Python, the requirements, and the project files:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

When running in Docker on the same machine as LM Studio, set `lm_studio_base_url` to `http://host.docker.internal:1234/v1` (or your host's LAN IP on Linux).

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Chat UI |
| `GET` | `/api/config` | UI defaults (generation parameters, model management defaults) |
| `GET` | `/api/models` | Loaded LLM instances (chat model dropdown source) |
| `POST` | `/api/chat` | Streaming chat completion (Server-Sent Events) |
| `GET` | `/api/models/manage` | All downloaded models with load state and details |
| `POST` | `/api/models/load` | Load a model: `{"model", "ttl_seconds"?, "context_length"?}` |
| `POST` | `/api/models/unload` | Unload one instance: `{"instance_id"}` |
| `POST` | `/api/models/unload-all` | Unload every loaded instance |
| `GET` | `/api/images/capability` | Whether the image API supports generation (`?refresh=true` re-probes) |
| `POST` | `/api/images` | Generate image(s): `{"prompt", "size"?, "n"?}` |

## License

See [LICENSE](LICENSE).
