# LocalMind

A lightweight, self-hosted chat interface for a local [LM Studio](https://lmstudio.ai/) server.

LocalMind is a small FastAPI application with a vanilla HTML/CSS/JS frontend. It talks to LM Studio's OpenAI-compatible API and gives you:

- **Dynamic model selection** — the dropdown is populated from LM Studio's `/v1/models` endpoint at load time.
- **Real-time streaming** — responses render token-by-token via Server-Sent Events.
- **Parameter controls** — adjust `temperature` and `max_tokens` from a settings panel before sending.
- **Robust error handling** — a clear UI banner (with retry) when LM Studio is offline, unreachable, or has no model loaded.
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
| `GET` | `/api/config` | UI defaults (temperature, max tokens, system prompt) |
| `GET` | `/api/models` | Models currently available in LM Studio |
| `POST` | `/api/chat` | Streaming chat completion (Server-Sent Events) |

## License

See [LICENSE](LICENSE).
