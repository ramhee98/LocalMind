"""LocalMind — a lightweight chat interface for a local LM Studio server.

FastAPI backend that proxies the LM Studio OpenAI-compatible API:
  * GET  /api/config  -> default generation parameters for the UI
  * GET  /api/models  -> models currently loaded/available in LM Studio
  * POST /api/chat    -> streaming chat completion (Server-Sent Events)
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("localmind")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CONFIG_FILE = BASE_DIR / "config.json"
CONFIG_TEMPLATE_FILE = BASE_DIR / "config.template.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "lm_studio_base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "host": "0.0.0.0",
    "port": 8000,
    "request_timeout_seconds": 120,
    "defaults": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "system_prompt": "You are a helpful assistant.",
    },
}


def load_config() -> dict[str, Any]:
    """Load config.json, falling back to config.template.json, then built-in defaults.

    Loaded values are merged over the defaults so a partial config file is valid.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    for path in (CONFIG_FILE, CONFIG_TEMPLATE_FILE):
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                loaded = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s (%s), trying next fallback.", path.name, exc)
            continue
        if not isinstance(loaded, dict):
            logger.warning("%s does not contain a JSON object, trying next fallback.", path.name)
            continue
        if path is CONFIG_TEMPLATE_FILE:
            logger.warning("config.json not found — using config.template.json. "
                           "Copy it to config.json to customize your setup.")
        defaults = {**config["defaults"], **loaded.get("defaults", {})}
        config.update(loaded)
        config["defaults"] = defaults
        logger.info("Configuration loaded from %s", path.name)
        return config
    logger.warning("No config file found — using built-in defaults.")
    return config


CONFIG = load_config()

client = OpenAI(
    base_url=CONFIG["lm_studio_base_url"],
    api_key=CONFIG["api_key"],
    timeout=CONFIG["request_timeout_seconds"],
)

app = FastAPI(title="LocalMind", version="1.0.0")


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=CONFIG["defaults"]["temperature"], ge=0.0, le=2.0)
    max_tokens: int = Field(default=CONFIG["defaults"]["max_tokens"], ge=1, le=131072)


def connection_error_payload() -> dict[str, str]:
    return {
        "error": (
            "Cannot reach the LM Studio server at "
            f"{CONFIG['lm_studio_base_url']}. Make sure LM Studio is running "
            "and its local server is started (Developer tab > Start Server)."
        )
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Expose only UI-relevant defaults — never the full server config."""
    return {"defaults": CONFIG["defaults"], "base_url": CONFIG["lm_studio_base_url"]}


@app.get("/api/models")
def list_models() -> JSONResponse:
    try:
        models = sorted(model.id for model in client.models.list())
    except (APIConnectionError, APITimeoutError):
        logger.exception("LM Studio unreachable while listing models")
        return JSONResponse(status_code=502, content=connection_error_payload())
    except APIStatusError as exc:
        logger.exception("LM Studio returned an error while listing models")
        return JSONResponse(status_code=502, content={"error": f"LM Studio error: {exc.message}"})
    return JSONResponse(content={"models": models})


def stream_completion(request: ChatRequest) -> Iterator[str]:
    """Yield Server-Sent Events with incremental completion tokens.

    Errors are reported as an SSE `error` field because the HTTP status
    is already committed once streaming has started.
    """
    try:
        stream = client.chat.completions.create(
            model=request.model,
            messages=[message.model_dump() for message in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'content': delta})}\n\n"
        yield "data: [DONE]\n\n"
    except (APIConnectionError, APITimeoutError):
        logger.exception("LM Studio unreachable during chat completion")
        yield f"data: {json.dumps(connection_error_payload())}\n\n"
    except APIStatusError as exc:
        logger.exception("LM Studio returned an error during chat completion")
        yield f"data: {json.dumps({'error': f'LM Studio error: {exc.message}'})}\n\n"
    except Exception:  # noqa: BLE001 — never leak a raw traceback into the stream
        logger.exception("Unexpected error during chat completion")
        yield f"data: {json.dumps({'error': 'Unexpected server error. Check the application logs.'})}\n\n"


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_completion(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
