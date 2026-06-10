"""LocalMind — a lightweight chat interface for a local LM Studio server.

FastAPI backend that proxies the LM Studio OpenAI-compatible API:
  * GET  /api/config             -> default generation parameters for the UI
  * GET  /api/models             -> models currently loaded/available in LM Studio
  * POST /api/chat               -> streaming chat completion (Server-Sent Events)

and the LM Studio native REST API for model lifecycle management:
  * GET  /api/models/manage      -> all downloaded models with load state and details
  * POST /api/models/load        -> load a model (optional idle TTL and context length)
  * POST /api/models/unload      -> unload a single loaded instance
  * POST /api/models/unload-all  -> unload every loaded instance

Image generation (OpenAI-compatible /images/generations, capability-detected):
  * GET  /api/images/capability  -> whether the connected server can generate images
  * POST /api/images             -> generate image(s) from a text prompt
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx
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
    "model_management": {
        "default_ttl_seconds": 600,
        "auto_unload_by_default": True,
        "default_context_length": None,
        "load_timeout_seconds": 600,
        "status_refresh_seconds": 10,
    },
    "image_generation": {
        "api_base_url": None,
        "model": None,
        "default_size": "1024x1024",
        "timeout_seconds": 300,
    },
}


def load_config() -> dict[str, Any]:
    """Load config.json, falling back to config.template.json, then built-in defaults.

    Loaded values are merged over the defaults (one level deep for nested
    sections) so a partial config file is valid.
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
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
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

# The model management endpoints live on LM Studio's native REST API at the
# server root (e.g. http://localhost:1234/api/v1), not under the
# OpenAI-compatible /v1 prefix.
LM_SERVER_ROOT = CONFIG["lm_studio_base_url"].rstrip("/").removesuffix("/v1")

native_client = httpx.Client(
    base_url=LM_SERVER_ROOT,
    timeout=CONFIG["request_timeout_seconds"],
)

# Image generation can target a different OpenAI-compatible server (e.g.
# LocalAI or a Stable Diffusion gateway); it defaults to the LM Studio URL.
IMAGE_API_BASE = (
    CONFIG["image_generation"]["api_base_url"] or CONFIG["lm_studio_base_url"]
).rstrip("/")

image_client = OpenAI(
    base_url=IMAGE_API_BASE,
    api_key=CONFIG["api_key"],
    timeout=CONFIG["image_generation"]["timeout_seconds"],
)

app = FastAPI(title="LocalMind", version="1.1.0")


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=CONFIG["defaults"]["temperature"], ge=0.0, le=2.0)
    max_tokens: int = Field(default=CONFIG["defaults"]["max_tokens"], ge=1, le=131072)


class LoadModelRequest(BaseModel):
    model: str = Field(min_length=1)
    ttl_seconds: Optional[int] = Field(default=None, ge=1, le=86400 * 7)
    context_length: Optional[int] = Field(default=None, ge=1, le=10_000_000)


class UnloadModelRequest(BaseModel):
    instance_id: str = Field(min_length=1)


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)
    size: Optional[str] = Field(default=None, pattern=r"^\d{2,4}x\d{2,4}$")
    n: int = Field(default=1, ge=1, le=4)
    # Chat model id used to rewrite the prompt before generation (optional).
    enhance_with: Optional[str] = Field(default=None, min_length=1)


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
    return {
        "defaults": CONFIG["defaults"],
        "model_management": CONFIG["model_management"],
        "image_generation": {"default_size": CONFIG["image_generation"]["default_size"]},
        "base_url": CONFIG["lm_studio_base_url"],
    }


# ---------- Image generation ----------

_image_capability: Optional[dict[str, Any]] = None

ENHANCE_SYSTEM_PROMPT = (
    "You are an expert image-prompt engineer. Rewrite the user's idea as one "
    "vivid, detailed prompt for a text-to-image model: subject, style, "
    "lighting, composition, mood, quality keywords. Reply with the prompt "
    "text only — no quotes, no preamble, no explanations."
)


def enhance_prompt(model: str, prompt: str) -> str:
    """Have a chat LLM (e.g. Gemma in LM Studio) rewrite an image prompt."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ENHANCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=500,
        # Reasoning models would otherwise spend the whole token budget
        # thinking; LM Studio supports turning that off per request.
        extra_body={"reasoning_effort": "none"},
    )
    text = (completion.choices[0].message.content or "").strip()
    # Some reasoning models emit a <think>...</think> block before the answer.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text or prompt


def probe_image_support() -> dict[str, Any]:
    """Detect whether the image API server offers /images/generations.

    An intentionally empty request is sent: a server that implements the
    endpoint answers with a validation error, while LM Studio (which currently
    has no image generation) answers 200/404 with an "Unexpected endpoint"
    error body.
    """
    try:
        response = httpx.post(f"{IMAGE_API_BASE}/images/generations", json={}, timeout=10)
    except httpx.HTTPError:
        return {
            "supported": False,
            "detail": f"The image API server at {IMAGE_API_BASE} is unreachable.",
        }
    if response.status_code == 404 or "Unexpected endpoint" in response.text[:500]:
        return {
            "supported": False,
            "detail": (
                "The connected server does not support image generation. "
                "LM Studio currently has no image endpoint — point "
                "image_generation.api_base_url in config.json at an "
                "OpenAI-compatible image server to enable this."
            ),
        }
    return {"supported": True, "detail": f"Image generation endpoint detected at {IMAGE_API_BASE}."}


@app.get("/api/images/capability")
def image_capability(refresh: bool = False) -> dict[str, Any]:
    global _image_capability
    if _image_capability is None or refresh:
        _image_capability = probe_image_support()
        logger.info("Image generation capability: %s", _image_capability)
    return _image_capability


@app.post("/api/images")
def generate_images(request: ImageRequest) -> JSONResponse:
    prompt = request.prompt
    enhancement_error: Optional[str] = None
    if request.enhance_with:
        try:
            prompt = enhance_prompt(request.enhance_with, request.prompt)
            logger.info("Prompt enhanced by %s: %r -> %r",
                        request.enhance_with, request.prompt, prompt)
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            enhancement_error = getattr(exc, "message", None) or str(exc)
            logger.warning("Prompt enhancement failed, using original prompt: %s",
                           enhancement_error)

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "n": request.n,
        "response_format": "b64_json",
    }
    if request.size:
        kwargs["size"] = request.size
    if CONFIG["image_generation"]["model"]:
        kwargs["model"] = CONFIG["image_generation"]["model"]
    try:
        result = image_client.images.generate(**kwargs)
    except (APIConnectionError, APITimeoutError):
        logger.exception("Image API unreachable during generation")
        return JSONResponse(
            status_code=502,
            content={"error": f"Cannot reach the image API server at {IMAGE_API_BASE}."},
        )
    except APIStatusError as exc:
        logger.exception("Image API returned an error during generation")
        return JSONResponse(status_code=502, content={"error": f"Image API error: {exc.message}"})

    images: list[str] = []
    for item in result.data or []:
        if getattr(item, "b64_json", None):
            images.append(f"data:image/png;base64,{item.b64_json}")
        elif getattr(item, "url", None):
            images.append(item.url)
    if not images:
        capability = probe_image_support()
        message = capability["detail"] if not capability["supported"] \
            else "The image server returned no images."
        return JSONResponse(status_code=502, content={"error": message})
    return JSONResponse(content={
        "images": images,
        "prompt_used": prompt,
        "enhanced": prompt != request.prompt,
        "enhancement_error": enhancement_error,
    })


def fetch_native_models() -> list[dict[str, Any]]:
    """Return the raw model list from LM Studio's native REST API."""
    response = native_client.get("/api/v1/models")
    response.raise_for_status()
    return response.json().get("models", [])


def native_error_message(response: httpx.Response) -> str:
    """Extract a human-readable message from a native API error response."""
    try:
        error = response.json().get("error", {})
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    except (json.JSONDecodeError, ValueError):
        pass
    return f"LM Studio returned HTTP {response.status_code}."


@app.get("/api/models")
def list_models() -> JSONResponse:
    """Models selectable for chat: loaded LLM instances (chat dropdown source).

    Prefers the native API so embedding models are excluded; falls back to the
    OpenAI-compatible listing if the native API is unavailable.
    """
    try:
        models = sorted(
            instance["id"]
            for model in fetch_native_models()
            if model.get("type") != "embedding"
            for instance in model.get("loaded_instances", [])
            if instance.get("id")
        )
        return JSONResponse(content={"models": models})
    except (httpx.HTTPError, ValueError):
        logger.warning("Native model list unavailable, falling back to OpenAI-compatible API.")
    try:
        models = sorted(model.id for model in client.models.list())
    except (APIConnectionError, APITimeoutError):
        logger.exception("LM Studio unreachable while listing models")
        return JSONResponse(status_code=502, content=connection_error_payload())
    except APIStatusError as exc:
        logger.exception("LM Studio returned an error while listing models")
        return JSONResponse(status_code=502, content={"error": f"LM Studio error: {exc.message}"})
    return JSONResponse(content={"models": models})


@app.get("/api/models/manage")
def manage_models() -> JSONResponse:
    """All downloaded models with their load state, for the management panel."""
    try:
        raw_models = fetch_native_models()
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.exception("LM Studio unreachable while fetching model status")
        return JSONResponse(status_code=502, content=connection_error_payload())
    except httpx.HTTPStatusError as exc:
        logger.exception("LM Studio returned an error while fetching model status")
        return JSONResponse(status_code=502, content={"error": native_error_message(exc.response)})

    models = [
        {
            "key": model.get("key"),
            "display_name": model.get("display_name") or model.get("key"),
            "type": model.get("type"),
            "params": model.get("params_string"),
            "quantization": (model.get("quantization") or {}).get("name"),
            "size_bytes": model.get("size_bytes"),
            "max_context_length": model.get("max_context_length"),
            "loaded_instances": [
                {
                    "id": instance.get("id"),
                    "context_length": (instance.get("config") or {}).get("context_length"),
                }
                for instance in model.get("loaded_instances", [])
            ],
        }
        for model in raw_models
        if model.get("key")
    ]
    return JSONResponse(content={"models": models})


@app.post("/api/models/load")
def load_model(request: LoadModelRequest) -> JSONResponse:
    payload: dict[str, Any] = {"model": request.model}
    if request.ttl_seconds is not None:
        payload["ttl_seconds"] = request.ttl_seconds
    if request.context_length is not None:
        payload["context_length"] = request.context_length
    try:
        response = native_client.post(
            "/api/v1/models/load",
            json=payload,
            timeout=CONFIG["model_management"]["load_timeout_seconds"],
        )
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.exception("LM Studio unreachable while loading %s", request.model)
        return JSONResponse(status_code=502, content=connection_error_payload())
    if response.is_error:
        logger.error("Failed to load %s: %s", request.model, response.text)
        return JSONResponse(status_code=502, content={"error": native_error_message(response)})
    logger.info("Loaded model %s (ttl=%s, context_length=%s)",
                request.model, request.ttl_seconds, request.context_length)
    return JSONResponse(content=response.json())


def unload_instance(instance_id: str) -> Optional[str]:
    """Unload one instance; return an error message or None on success."""
    try:
        response = native_client.post("/api/v1/models/unload", json={"instance_id": instance_id})
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.exception("LM Studio unreachable while unloading %s", instance_id)
        return connection_error_payload()["error"]
    if response.is_error:
        logger.error("Failed to unload %s: %s", instance_id, response.text)
        return native_error_message(response)
    logger.info("Unloaded model instance %s", instance_id)
    return None


@app.post("/api/models/unload")
def unload_model(request: UnloadModelRequest) -> JSONResponse:
    error = unload_instance(request.instance_id)
    if error:
        return JSONResponse(status_code=502, content={"error": error})
    return JSONResponse(content={"status": "unloaded", "instance_id": request.instance_id})


@app.post("/api/models/unload-all")
def unload_all_models() -> JSONResponse:
    try:
        raw_models = fetch_native_models()
    except (httpx.HTTPError, ValueError):
        logger.exception("LM Studio unreachable while listing models for unload-all")
        return JSONResponse(status_code=502, content=connection_error_payload())

    unloaded: list[str] = []
    errors: list[str] = []
    for model in raw_models:
        for instance in model.get("loaded_instances", []):
            instance_id = instance.get("id")
            if not instance_id:
                continue
            error = unload_instance(instance_id)
            if error:
                errors.append(f"{instance_id}: {error}")
            else:
                unloaded.append(instance_id)
    return JSONResponse(content={"unloaded": unloaded, "errors": errors})


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
