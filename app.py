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

Text-to-speech (Kokoro-82M, fully local & offline, CPU-only so it never takes
VRAM from the LLM):
  * POST /api/tts                -> stream synthesized speech as raw float32 PCM

Document upload (text is extracted server-side and injected into the chat):
  * POST /api/upload             -> extract text from an uploaded PDF

Conversation persistence (SQLite, server-side):
  * GET    /api/conversations        -> list conversations (newest first)
  * POST   /api/conversations        -> create an empty conversation
  * GET    /api/conversations/search -> full-text search across all conversations
  * GET    /api/conversations/{id}   -> full message history
  * PUT    /api/conversations/{id}   -> save messages and/or rename
  * DELETE /api/conversations/{id}   -> delete

System prompt presets (SQLite, server-side — reusable named personas):
  * GET    /api/system-prompts       -> list presets (newest first)
  * POST   /api/system-prompts       -> create a named preset
  * PUT    /api/system-prompts/{id}  -> rename and/or edit content
  * DELETE /api/system-prompts/{id}  -> delete
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import as_completed
from contextlib import closing
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator, Optional, Union
from urllib.parse import parse_qsl, urldefrag, urljoin, urlsplit

import httpx
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("localmind")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
# LOCALMIND_CONFIG overrides the config path (useful for Docker and tests).
CONFIG_FILE = Path(os.environ.get("LOCALMIND_CONFIG") or BASE_DIR / "config.json")
CONFIG_TEMPLATE_FILE = BASE_DIR / "config.template.json"
CERTS_DIR = BASE_DIR / "certs"
DATA_DIR = BASE_DIR / "data"
DB_FILE = DATA_DIR / "localmind.db"

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
        # Path to LM Studio's `lms` CLI. The REST load endpoint can't set a TTL,
        # so manual loads shell out to `lms load --ttl` when this binary is
        # found; null/missing falls back to a REST load (context length only).
        "lms_cli_path": None,
    },
    "image_generation": {
        "api_base_url": None,
        "model": None,
        "default_size": "1024x1024",
        "timeout_seconds": 300,
    },
    "documents": {
        "max_file_size_mb": 25,
        "max_text_chars": 20000,
    },
    "rag": {
        "enabled": True,
        "embedding_model": "text-embedding-nomic-embed-text-v1.5",
        # Documents longer than this (chars) are retrieved instead of inlined.
        "min_chars_for_rag": 8000,
        "chunk_chars": 1200,
        "chunk_overlap": 200,
        "top_k": 5,
    },
    "web_search": {
        # Master switch; when False the composer toggle is hidden entirely.
        "enabled": True,
        # "duckduckgo" works with zero setup (no API key, via the ddgs
        # package); "searxng" queries a self-hosted SearXNG instance and
        # requires searxng_base_url (the instance must allow format=json).
        "provider": "duckduckgo",
        "searxng_base_url": None,
        # ddgs engine order ("auto" rotates randomly, which mixes in weak
        # engines — fuzzy Wikipedia title matches and the like). Must use
        # names ddgs knows, or it silently falls back to "auto". bing is
        # deliberately absent: it returns fuzzy keyword matches that poison
        # the context (e.g. beer-company pages for an "Asahi Linux" query).
        "backends": "duckduckgo, mojeek, brave",
        # Search region/language as "country-language" (DuckDuckGo format,
        # e.g. "ch-de", "de-de", "us-en"). Also sets the Accept-Language for
        # page fetches and the SearXNG language. "wt-wt" = no region.
        "region": "wt-wt",
        "max_results": 5,
        "timeout_seconds": 15,
        # Distill the conversation into a search query with the chat model
        # first; the raw chat message makes a poor query ("In one sentence,
        # what …?") and follow-ups lack context entirely.
        "rewrite_query": True,
        # Download the top result pages and embed-rank their text against the
        # query (the RAG pipeline); off injects only the search snippets.
        "fetch_pages": True,
        # How many page chunks to inject when fetch_pages is on.
        "top_k": 6,
        # Extracted text per page is capped at this many characters.
        "max_page_chars": 20000,
        # Pasted URLs: when the message asks for a deep dive into a site (or
        # restricts sources to the link), same-site links found on the pasted
        # pages are crawled too, up to this many pages in total.
        "crawl_max_pages": 8,
        # How many ranked excerpts from linked/crawled pages to inject.
        "linked_top_k": 10,
        # Fetched pages are reused for this long, so follow-up questions
        # about the same site don't re-download it every turn. 0 disables.
        "page_cache_ttl_seconds": 900,
        # Research mode: the question is decomposed into sub-queries, each is
        # searched and read, then the model reviews what was found and
        # proposes follow-up queries for the gaps — repeated up to
        # research_max_rounds times (the loop is code-bounded; the model only
        # proposes queries). 1 round = plain fan-out without reflection.
        "research_max_rounds": 2,
        "research_queries_per_round": 3,
        # How many ranked page excerpts the synthesis prompt receives. Mind
        # the model's context window: ~12 excerpts ≈ 4k tokens.
        "research_top_k": 12,
    },
    "tts": {
        # Master switch; when False the message speaker button is hidden and
        # /api/tts returns 503. Kokoro-82M runs fully offline on the CPU.
        "enabled": True,
        # Kokoro language code: 'a' American English, 'b' British English,
        # 'e' Spanish, 'f' French, 'h' Hindi, 'i' Italian, 'p' Portuguese,
        # 'j' Japanese, 'z' Mandarin (some need extra packages / espeak-ng).
        "lang_code": "a",
        # Default voice (see Kokoro's voice list, e.g. af_heart, af_bella,
        # am_michael, bf_emma). Its locale must match lang_code.
        "voice": "af_heart",
        # Speech rate multiplier (1.0 = natural).
        "speed": 1.0,
        # Kokoro's native output rate. The client reads this from a response
        # header; only change it if you change Kokoro's output.
        "sample_rate": 24000,
        # Reject over-long requests so one call can't tie up the CPU for ages.
        "max_chars": 5000,
        # Preload the model in a background thread at startup so the first
        # request isn't slow. Set False to defer the load until first use.
        "warmup": True,
    },
    "tls": {
        "enabled": False,
        "cert_file": None,
        "key_file": None,
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

# ---------- Text-to-speech (Kokoro-82M, fully local & offline) ----------
#
# Kokoro synthesizes speech entirely on-device. The pipeline is pinned to the
# CPU so it never competes with the LLM for VRAM, and it is built once and
# reused — importing torch and loading the model is slow, but only on first
# use. Synthesis is serialized by a lock: a single KPipeline is not safe to
# drive from several requests at once, and one CPU TTS job already wants every
# core. Everything degrades gracefully when the optional `kokoro` package is
# absent (the UI hides the speaker button; the endpoint returns 503).


class TTSUnavailable(RuntimeError):
    """Raised when Kokoro cannot be imported or initialized."""


_tts_pipeline: Any = None
_tts_init_lock = threading.Lock()   # guards one-time pipeline construction
_tts_synth_lock = threading.Lock()  # serializes synthesis across requests
# Latched ONLY when the package is genuinely missing (permanent until restart);
# a transient construction failure (e.g. a network blip during the model
# download) is not latched so the next request retries.
_tts_init_error: Optional[str] = None


def _positive_int(value: Any, default: int) -> int:
    """Coerce a config value to a positive int, falling back on junk input."""
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced > 0 else default


# Frozen once at import (CONFIG is loaded once too). Coerced defensively so a
# malformed max_chars in config.json can't crash the TTSRequest field below.
TTS_MAX_CHARS = _positive_int(CONFIG["tts"]["max_chars"], 5000)


def tts_supported() -> bool:
    """Cheap check (no model load) for whether the TTS feature is usable.

    True only when enabled in config *and* the kokoro package is importable,
    so a missing dependency keeps the UI button hidden instead of erroring on
    every click. A prior failed init also flips this off.
    """
    if not CONFIG["tts"]["enabled"] or _tts_init_error:
        return False
    return importlib.util.find_spec("kokoro") is not None


def _configure_espeak_fallback() -> bool:
    """Point phonemizer at a bundled espeak-ng before Kokoro builds its G2P.

    Kokoro's English G2P only covers dictionary words; out-of-vocabulary words
    (proper names, foreign words) need an espeak-ng fallback. misaki can't build
    that fallback unless a usable espeak-ng library is loadable, so without one
    Kokoro silently drops those words — names just go missing from the speech.
    The `espeakng-loader` package ships a prebuilt library for every platform,
    so we wire it in here and no system install is required.

    Best-effort: if the loader or phonemizer is absent we log and carry on (TTS
    still works, it just skips unknown names as before). An espeak-ng the user
    already configured is left untouched.
    """
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except Exception as exc:
        logger.warning("espeak-ng fallback unavailable (%s); Kokoro will skip "
                       "out-of-vocabulary words such as proper names. Install "
                       "'espeakng-loader' to enable it.", exc)
        return False
    try:
        if not getattr(EspeakWrapper, "_ESPEAK_LIBRARY", None):
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
        # espeak-ng resolves its phoneme data from this env var; set before the
        # library is initialized (i.e. before KPipeline construction).
        os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
        return True
    except Exception:
        logger.warning("Could not configure the espeak-ng fallback", exc_info=True)
        return False


def get_tts_pipeline() -> Any:
    """Return the shared Kokoro pipeline, building it on first use.

    Construction (torch import + model download/load) can take several seconds,
    so it happens once under a lock and is cached. Raises TTSUnavailable with a
    user-facing message when Kokoro is missing or fails to initialize.
    """
    global _tts_pipeline, _tts_init_error
    if _tts_pipeline is not None:
        return _tts_pipeline
    if not CONFIG["tts"]["enabled"]:
        raise TTSUnavailable("Text-to-speech is disabled in the server configuration.")
    with _tts_init_lock:
        if _tts_pipeline is not None:
            return _tts_pipeline
        # Must run before kokoro is imported so misaki picks up the library.
        _configure_espeak_fallback()
        try:
            from kokoro import KPipeline
        except Exception as exc:  # ImportError or a transitive import failure
            _tts_init_error = (
                "The 'kokoro' package is not installed. Install it with "
                "`pip install kokoro` to enable text-to-speech."
            )
            logger.warning("TTS unavailable: %s (%s)", _tts_init_error, exc)
            raise TTSUnavailable(_tts_init_error) from exc
        try:
            lang_code = CONFIG["tts"]["lang_code"]
            logger.info("Initializing Kokoro TTS pipeline (lang_code=%s, device=cpu)…",
                        lang_code)
            # device='cpu' keeps TTS off the GPU so it never steals VRAM from the LLM.
            _tts_pipeline = KPipeline(lang_code=lang_code, device="cpu")
            logger.info("Kokoro TTS pipeline ready.")
        except Exception as exc:
            # Construction (model download / load) can fail transiently. Do NOT
            # latch _tts_init_error here, so the next request retries instead of
            # disabling TTS until restart — unlike a missing package (above),
            # which is permanent.
            message = f"Failed to initialize the Kokoro TTS pipeline: {exc}"
            logger.exception("TTS initialization failed (will retry on next request)")
            raise TTSUnavailable(message) from exc
    return _tts_pipeline


def _segment_to_pcm_bytes(audio: Any) -> bytes:
    """Convert one Kokoro audio segment to little-endian float32 PCM bytes.

    Kokoro yields float32 samples in [-1, 1] (a torch tensor or a numpy array);
    we ship them as raw 32-bit float so the browser's Web Audio API can play
    each chunk with no decode step. '<f4' pins little-endian regardless of host.
    """
    import numpy as np

    if audio is None:
        return b""
    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach().to("cpu").numpy()
    arr = np.ascontiguousarray(audio, dtype="<f4").reshape(-1)
    return arr.tobytes()


def stream_tts(pipeline: Any, request: "TTSRequest") -> Iterator[bytes]:
    """Yield raw float32 PCM as Kokoro produces each sentence/segment.

    The synthesis lock is held only around each segment's CPU work — never
    across a ``yield`` — so a slow or disconnected client can neither block
    other TTS requests nor strand the lock, while the single non-reentrant
    pipeline is still driven by at most one thread at a time.

    Text is pre-split on newlines so a G2P failure in one paragraph only skips
    that paragraph rather than aborting the entire stream.
    """
    cfg = CONFIG["tts"]
    voice = request.voice or cfg["voice"]
    speed = request.speed or cfg["speed"]
    text = request.text.strip()
    lines = [line for line in re.split(r"\n+", text) if line.strip()]
    for line in lines:
        # split_pattern=None: we already split by newline above; let Kokoro's
        # tokenizer handle sentence chunking within each line. Creating the
        # generator is lazy (Kokoro's __call__ yields), so the model isn't
        # touched until the first next() below, which runs under the lock.
        generator = pipeline(line, voice=voice, speed=speed, split_pattern=None)
        while True:
            with _tts_synth_lock:
                try:
                    result = next(generator)
                except StopIteration:
                    break
                except Exception:
                    logger.warning("TTS skipped a segment due to synthesis error",
                                   exc_info=True)
                    break
            # Kokoro yields (graphemes, phonemes, audio); tolerate either a
            # tuple/list or an object exposing .audio across kokoro versions.
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)):
                audio = result[-1]
            chunk = _segment_to_pcm_bytes(audio)
            if chunk:
                yield chunk


def _warm_tts_pipeline() -> None:
    """Best-effort background preload so the first /api/tts call isn't slow."""
    try:
        get_tts_pipeline()
    except TTSUnavailable:
        pass  # already logged; the feature just stays unavailable
    except Exception:
        logger.exception("Unexpected error while warming the TTS pipeline")


if CONFIG["tts"].get("warmup", True) and tts_supported():
    threading.Thread(target=_warm_tts_pipeline, name="tts-warmup", daemon=True).start()


app = FastAPI(title="LocalMind", version="1.2.0")


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    # Either plain text or OpenAI content parts (text + image_url) for
    # multimodal messages sent to vision models.
    content: Union[str, list[dict[str, Any]]]


class ChatRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=CONFIG["defaults"]["temperature"], ge=0.0, le=2.0)
    max_tokens: int = Field(default=CONFIG["defaults"]["max_tokens"], ge=1, le=131072)
    # LM Studio reasoning control; None leaves the model's default untouched.
    reasoning_effort: Optional[str] = Field(
        default=None, pattern="^(none|minimal|low|medium|high|xhigh)$")
    # RAG document ids; relevant chunks are retrieved and prepended to the
    # last user turn instead of inlining whole documents.
    doc_ids: Optional[list[str]] = Field(default=None)
    # Web grounding mode: "search" injects results for one rewritten query;
    # "research" runs the multi-round plan/search/reflect pipeline. Either
    # way the findings are prepended to the last user turn like RAG.
    web_search: Optional[str] = Field(default=None, pattern="^(search|research)$")
    # Idle TTL (seconds) for LM Studio to apply when this request JIT-loads the
    # model. None means "use the configured default"; 0 disables auto-unload.
    ttl_seconds: Optional[int] = Field(default=None, ge=0, le=86400 * 7)


class LoadModelRequest(BaseModel):
    model: str = Field(min_length=1)
    # TTL is honored only when loading via the `lms` CLI; a REST load ignores it
    # (the native endpoint has no TTL field). None = use the configured default.
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


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=TTS_MAX_CHARS)
    # Optional per-request overrides; fall back to the configured defaults.
    # The voice pattern matches Kokoro's identifiers (e.g. "af_heart").
    voice: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_]{1,80}$")
    speed: Optional[float] = Field(default=None, ge=0.5, le=2.0)


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
        "documents": CONFIG["documents"],
        # Lets the UI estimate how many tokens retrieved chunks will add.
        "rag": {
            "enabled": CONFIG["rag"]["enabled"],
            "top_k": CONFIG["rag"]["top_k"],
            "chunk_chars": CONFIG["rag"]["chunk_chars"],
        },
        # Lets the UI hide the composer toggle when web search is turned off.
        "web_search": {"enabled": CONFIG["web_search"]["enabled"]},
        # Lets the UI hide the speaker button when TTS is off or unavailable.
        "tts": {"enabled": tts_supported()},
        "base_url": CONFIG["lm_studio_base_url"],
    }


# ---------- Conversation persistence ----------

def db_connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def message_text(message: dict[str, Any]) -> str:
    """Best-effort plain text of a stored message (display field, text, or part)."""
    display = message.get("display")
    if isinstance(display, str) and display.strip():
        return display
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return str(part.get("text") or "")
    return ""


def searchable_body(messages: list[dict[str, Any]]) -> str:
    """Plain-text body of a conversation for the search index.

    Uses the display text of each message, so inlined document dumps and
    base64 image payloads never pollute search results.
    """
    return "\n".join(filter(None, (message_text(message) for message in messages)))


# False when this Python's SQLite was built without FTS5; search then falls
# back to a full scan, which is fine at personal-use scale.
FTS_AVAILABLE = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db() -> None:
    global FTS_AVAILABLE
    with closing(db_connect()) as connection, connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                   id TEXT PRIMARY KEY,
                   title TEXT NOT NULL,
                   created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL,
                   messages_json TEXT NOT NULL DEFAULT '[]'
               )""")
        try:
            connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts
                   USING fts5(conversation_id UNINDEXED, title, body)""")
        except sqlite3.OperationalError:
            FTS_AVAILABLE = False
            logger.warning("SQLite FTS5 is unavailable — conversation search "
                           "will scan conversations instead.")
            return
        # Index conversations that predate the search feature.
        missing = connection.execute(
            "SELECT id, title, messages_json FROM conversations WHERE id NOT IN "
            "(SELECT conversation_id FROM conversation_fts)").fetchall()
        for row in missing:
            try:
                messages = json.loads(row["messages_json"])
            except json.JSONDecodeError:
                messages = []
            connection.execute(
                "INSERT INTO conversation_fts (conversation_id, title, body) "
                "VALUES (?, ?, ?)",
                (row["id"], row["title"], searchable_body(messages)))
        if missing:
            logger.info("Backfilled %s conversation(s) into the search index.", len(missing))
        connection.execute(
            """CREATE TABLE IF NOT EXISTS system_prompts (
                   id TEXT PRIMARY KEY,
                   name TEXT NOT NULL,
                   content TEXT NOT NULL,
                   created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL
               )""")
        seed_system_prompts(connection)


# Starter personas, seeded once into an empty system_prompts table. The first
# entry mirrors the configured default so the picker always has a baseline.
def starter_system_prompts() -> list[tuple[str, str]]:
    return [
        ("Default", CONFIG["defaults"]["system_prompt"]),
        ("Code reviewer",
         "You are a meticulous senior software engineer reviewing code. "
         "Point out bugs, edge cases, security issues, and style problems. "
         "Be concise, cite specific lines, and suggest concrete fixes."),
        ("Translator",
         "You are a professional translator. Translate the user's text "
         "faithfully, preserving tone and meaning. If the target language is "
         "ambiguous, ask. Output only the translation unless asked otherwise."),
        ("Socratic tutor",
         "You are a Socratic tutor. Never give the answer outright. Guide the "
         "user to it with probing questions, hints, and small steps, checking "
         "their understanding as you go."),
    ]


def seed_system_prompts(connection: sqlite3.Connection) -> None:
    """Populate the presets table with starters the first time it is empty."""
    existing = connection.execute("SELECT COUNT(*) FROM system_prompts").fetchone()[0]
    if existing:
        return
    now = utc_now()
    for name, content in starter_system_prompts():
        connection.execute(
            "INSERT INTO system_prompts (id, name, content, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, name, content, now, now))
    logger.info("Seeded %s starter system prompt(s).", len(starter_system_prompts()))


init_db()


def update_search_index(connection: sqlite3.Connection, conversation_id: str,
                        title: str, messages: list[dict[str, Any]]) -> None:
    if not FTS_AVAILABLE:
        return
    connection.execute(
        "DELETE FROM conversation_fts WHERE conversation_id = ?", (conversation_id,))
    connection.execute(
        "INSERT INTO conversation_fts (conversation_id, title, body) VALUES (?, ?, ?)",
        (conversation_id, title, searchable_body(messages)))


# Bound the persisted payload so an unauthenticated PUT can't OOM the process
# or fill the disk; base64 image attachments are stored inline, so this is
# generous but finite.
MAX_CONVERSATION_BYTES = 32 * 1024 * 1024
MAX_CONVERSATION_MESSAGES = 2000


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    messages: Optional[list[dict[str, Any]]] = Field(
        default=None, max_length=MAX_CONVERSATION_MESSAGES)


# Generous but finite, so an unauthenticated POST can't store an unbounded blob.
MAX_SYSTEM_PROMPT_CHARS = 20000


class SystemPromptCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=MAX_SYSTEM_PROMPT_CHARS)


class SystemPromptUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    content: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_SYSTEM_PROMPT_CHARS)


def derive_title(messages: list[dict[str, Any]]) -> Optional[str]:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = " ".join(message_text(message).split())
        if text:
            return text[:60] + ("…" if len(text) > 60 else "")
    return None


@app.get("/api/conversations")
def list_conversations() -> dict[str, Any]:
    with closing(db_connect()) as connection:
        rows = connection.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC").fetchall()
    return {"conversations": [dict(row) for row in rows]}


@app.post("/api/conversations")
def create_conversation() -> dict[str, Any]:
    conversation_id = uuid.uuid4().hex
    now = utc_now()
    with closing(db_connect()) as connection, connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, 'New chat', ?, ?)",
            (conversation_id, now, now))
    logger.info("Created conversation %s", conversation_id)
    return {"id": conversation_id, "title": "New chat"}


def fts_match_expression(query: str) -> str:
    """Turn free-form user input into a safe FTS5 prefix query.

    Every term is quoted (so FTS5 operators and punctuation are literal) and
    given a trailing *, which makes search-as-you-type match partial words.
    """
    terms = (term.replace('"', '""') for term in query.split())
    return " ".join(f'"{term}"*' for term in terms)


def make_snippet(body: str, query: str, radius: int = 60) -> str:
    """A short context window around the first matched term, in plain text."""
    text = " ".join(body.split())
    lowered = text.lower()
    positions = [position for position in
                 (lowered.find(term.lower()) for term in query.split())
                 if position != -1]
    if not positions:
        return text[: radius * 2] + ("…" if len(text) > radius * 2 else "")
    start = max(0, min(positions) - radius)
    end = min(len(text), min(positions) + radius)
    return (("…" if start > 0 else "") + text[start:end]
            + ("…" if end < len(text) else ""))


@app.get("/api/conversations/search")
def search_conversations(q: str = "") -> dict[str, Any]:
    """Full-text search over titles and message text of all conversations."""
    query = " ".join(q.split())
    if not query:
        return {"results": []}
    with closing(db_connect()) as connection:
        if FTS_AVAILABLE:
            rows = connection.execute(
                "SELECT c.id, c.title, c.updated_at, f.body FROM conversation_fts f "
                "JOIN conversations c ON c.id = f.conversation_id "
                "WHERE conversation_fts MATCH ? ORDER BY rank LIMIT 30",
                (fts_match_expression(query),)).fetchall()
            matches = [(row["id"], row["title"], row["updated_at"], row["body"])
                       for row in rows]
        else:
            # FTS5 not compiled in: scan and substring-match every conversation.
            matches = []
            terms = [term.lower() for term in query.split()]
            for row in connection.execute(
                    "SELECT id, title, updated_at, messages_json FROM conversations "
                    "ORDER BY updated_at DESC").fetchall():
                try:
                    messages = json.loads(row["messages_json"])
                except json.JSONDecodeError:
                    messages = []
                body = searchable_body(messages)
                haystack = f"{row['title']}\n{body}".lower()
                if all(term in haystack for term in terms):
                    matches.append((row["id"], row["title"], row["updated_at"], body))
                if len(matches) >= 30:
                    break
    return {"results": [
        {"id": conversation_id, "title": title, "updated_at": updated_at,
         "snippet": make_snippet(body, query)}
        for conversation_id, title, updated_at, body in matches
    ]}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> JSONResponse:
    with closing(db_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"error": "Conversation not found."})
    try:
        messages = json.loads(row["messages_json"])
    except json.JSONDecodeError:
        logger.error("Corrupt messages_json in conversation %s", conversation_id)
        messages = []
    return JSONResponse(content={
        "id": row["id"], "title": row["title"], "messages": messages,
    })


@app.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, http_request: Request) -> JSONResponse:
    raw = await http_request.body()
    if len(raw) > MAX_CONVERSATION_BYTES:
        return JSONResponse(status_code=413, content={
            "error": f"Conversation too large "
                     f"(limit {MAX_CONVERSATION_BYTES // 1024 // 1024} MB)."})
    try:
        request = ConversationUpdate.model_validate_json(raw)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    with closing(db_connect()) as connection, connection:
        row = connection.execute(
            "SELECT title, messages_json FROM conversations WHERE id = ?",
            (conversation_id,)).fetchone()
        if row is None:
            return JSONResponse(status_code=404, content={"error": "Conversation not found."})
        title = request.title or row["title"]
        if request.messages is not None:
            # Auto-title untitled conversations from their first user message.
            if not request.title and row["title"] == "New chat":
                title = derive_title(request.messages) or title
            connection.execute(
                "UPDATE conversations SET messages_json = ?, title = ?, updated_at = ? "
                "WHERE id = ?",
                (json.dumps(request.messages), title, utc_now(), conversation_id))
            update_search_index(connection, conversation_id, title, request.messages)
        else:
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, utc_now(), conversation_id))
            try:
                stored = json.loads(row["messages_json"])
            except json.JSONDecodeError:
                stored = []
            update_search_index(connection, conversation_id, title, stored)
    return JSONResponse(content={"id": conversation_id, "title": title})


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> JSONResponse:
    with closing(db_connect()) as connection, connection:
        deleted = connection.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount
        if FTS_AVAILABLE:
            connection.execute(
                "DELETE FROM conversation_fts WHERE conversation_id = ?",
                (conversation_id,))
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "Conversation not found."})
    logger.info("Deleted conversation %s", conversation_id)
    return JSONResponse(content={"status": "deleted", "id": conversation_id})


# ---------- System prompt presets ----------

@app.get("/api/system-prompts")
def list_system_prompts() -> dict[str, Any]:
    with closing(db_connect()) as connection:
        rows = connection.execute(
            "SELECT id, name, content, created_at, updated_at FROM system_prompts "
            "ORDER BY updated_at DESC").fetchall()
    return {"presets": [dict(row) for row in rows]}


@app.post("/api/system-prompts")
async def create_system_prompt(http_request: Request) -> JSONResponse:
    try:
        request = SystemPromptCreate.model_validate_json(await http_request.body())
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    preset_id = uuid.uuid4().hex
    now = utc_now()
    with closing(db_connect()) as connection, connection:
        connection.execute(
            "INSERT INTO system_prompts (id, name, content, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (preset_id, request.name, request.content, now, now))
    logger.info("Created system prompt %s (%s)", preset_id, request.name)
    return JSONResponse(content={
        "id": preset_id, "name": request.name, "content": request.content,
        "created_at": now, "updated_at": now,
    })


@app.put("/api/system-prompts/{preset_id}")
async def update_system_prompt(preset_id: str, http_request: Request) -> JSONResponse:
    try:
        request = SystemPromptUpdate.model_validate_json(await http_request.body())
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    if request.name is None and request.content is None:
        return JSONResponse(status_code=422, content={
            "error": "Provide a name and/or content to update."})
    with closing(db_connect()) as connection, connection:
        row = connection.execute(
            "SELECT name, content FROM system_prompts WHERE id = ?",
            (preset_id,)).fetchone()
        if row is None:
            return JSONResponse(status_code=404, content={"error": "Preset not found."})
        name = request.name if request.name is not None else row["name"]
        content = request.content if request.content is not None else row["content"]
        now = utc_now()
        connection.execute(
            "UPDATE system_prompts SET name = ?, content = ?, updated_at = ? "
            "WHERE id = ?",
            (name, content, now, preset_id))
    return JSONResponse(content={
        "id": preset_id, "name": name, "content": content, "updated_at": now,
    })


@app.delete("/api/system-prompts/{preset_id}")
def delete_system_prompt(preset_id: str) -> JSONResponse:
    with closing(db_connect()) as connection, connection:
        deleted = connection.execute(
            "DELETE FROM system_prompts WHERE id = ?", (preset_id,)).rowcount
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "Preset not found."})
    logger.info("Deleted system prompt %s", preset_id)
    return JSONResponse(content={"status": "deleted", "id": preset_id})


# ---------- Document upload ----------

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".conf", ".log", ".xml", ".html", ".css",
    ".js", ".ts", ".py", ".java", ".c", ".cpp", ".h", ".hpp", ".go",
    ".rs", ".rb", ".php", ".sh", ".sql",
}


def extract_pdf_text(data: bytes) -> Union[tuple[str, int], JSONResponse]:
    """Return (text, page_count) or an error response."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Try an empty password; many PDFs are only owner-locked.
            if not reader.decrypt(""):
                return JSONResponse(status_code=400, content={
                    "error": "This PDF is password-protected and cannot be read."})
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        logger.warning("Failed to parse PDF: %s", exc)
        return JSONResponse(status_code=400, content={
            "error": "This file could not be parsed as a PDF."})
    except Exception:  # noqa: BLE001 — malformed PDFs raise various errors
        logger.exception("Unexpected error while parsing PDF")
        return JSONResponse(status_code=400, content={
            "error": "This file could not be parsed as a PDF."})

    text = "\n\n".join(
        f"[Page {number}]\n{content.strip()}"
        for number, content in enumerate(pages, start=1)
        if content.strip()
    )
    if not text:
        return JSONResponse(status_code=400, content={
            "error": "No extractable text found — this PDF appears to contain "
                     "only scanned images. OCR is not supported."})
    return text, len(pages)


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Extract text from an uploaded PDF or plain-text file for chat context."""
    filename = file.filename or "document"
    extension = Path(filename).suffix.lower()

    data = await file.read()
    max_bytes = CONFIG["documents"]["max_file_size_mb"] * 1024 * 1024
    if len(data) > max_bytes:
        return JSONResponse(status_code=400, content={
            "error": f"File is too large "
                     f"({len(data) / 1024 / 1024:.1f} MB, "
                     f"limit {CONFIG['documents']['max_file_size_mb']} MB)."})

    pages: Optional[int] = None
    if extension == ".pdf":
        result = extract_pdf_text(data)
        if isinstance(result, JSONResponse):
            return result
        text, pages = result
    elif extension in TEXT_EXTENSIONS:
        try:
            text = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            return JSONResponse(status_code=400, content={
                "error": f"{filename} is not valid UTF-8 text."})
        if not text:
            return JSONResponse(status_code=400, content={"error": f"{filename} is empty."})
    else:
        return JSONResponse(status_code=400, content={
            "error": f"Unsupported file type '{extension or 'none'}'. "
                     "Supported: PDF, plain-text, and code files. "
                     "Images can be attached directly in the chat."})

    max_chars = CONFIG["documents"]["max_text_chars"]
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n\n[Document truncated]"
    logger.info("Extracted %s chars from %s%s%s",
                len(text), filename,
                f" ({pages} pages)" if pages else "",
                ", truncated" if truncated else "")

    rag_config = CONFIG["rag"]
    if rag_config["enabled"] and len(text) >= rag_config["min_chars_for_rag"]:
        indexed = index_document(filename, text, pages)
        if indexed is not None:
            return JSONResponse(content=indexed)
        # Embedding failed — fall back to inlining the (truncated) text.
        logger.warning("RAG indexing failed for %s; falling back to inline text.", filename)

    return JSONResponse(content={
        "filename": filename,
        "pages": pages,
        "chars": len(text),
        "truncated": truncated,
        "text": text,
        "rag": False,
    })


# ---------- Retrieval-augmented generation over uploads ----------

# In-memory chunk store, keyed by doc_id. Cleared on restart, which is fine:
# the frontend re-uploads documents per session and degrades gracefully on miss.
DOC_STORE: dict[str, dict[str, Any]] = {}


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    chunks: list[str] = []
    start = 0
    length = len(text)
    # Overlap must stay below the window or chunks wouldn't advance.
    overlap = max(0, min(overlap, size - 1))
    step = max(1, size - overlap)
    while start < length:
        end = min(start + size, length)
        # Prefer to break on a paragraph/newline near the window edge.
        if end < length:
            window = text.rfind("\n", start + step, end)
            if window != -1:
                end = window
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length or end <= start:
            break
        # Step back by the overlap, but always advance by at least `step`.
        start = max(end - overlap, start + step)
    return chunks


def ensure_embedding_model_loaded() -> None:
    """Pre-load the embedding model with the configured TTL if it isn't loaded.

    The /v1/embeddings endpoint ignores a "ttl" field, so letting it JIT-load the
    model leaves it resident with no idle timeout. Loading it ourselves via `lms`
    first gives it the default TTL. Only loads when not already loaded — a second
    `lms load` would spawn a duplicate instance rather than refresh the TTL.

    Best-effort: any failure (CLI missing, lms error) is logged and we fall back
    to letting the embedding request JIT-load the model the old way.
    """
    mgmt = CONFIG["model_management"]
    if not (mgmt.get("auto_unload_by_default") and mgmt.get("default_ttl_seconds")):
        return  # auto-unload off — nothing to set, leave loading to the request
    model = CONFIG["rag"]["embedding_model"]
    try:
        already_loaded = any(
            m.get("key") == model and m.get("loaded_instances")
            for m in fetch_native_models()
        )
    except httpx.HTTPError:
        return  # can't tell; let the embedding request handle loading
    if already_loaded:
        return
    try:
        load_via_lms(model, mgmt["default_ttl_seconds"], None)
        logger.info("Pre-loaded embedding model %s with ttl=%s",
                    model, mgmt["default_ttl_seconds"])
    except FileNotFoundError:
        logger.warning(
            "lms CLI not found; embedding model %s will load without a TTL. "
            "Set model_management.lms_cli_path to enable it.", model)
    except RuntimeError as exc:
        logger.warning("Could not pre-load embedding model %s with a TTL: %s",
                       model, exc)


# Embedding cache (text -> vector): follow-up questions about a pasted site
# re-rank largely the same page chunks every turn, and re-embedding them is
# the slowest part of an otherwise cache-served crawl. FIFO-evicted.
EMBED_CACHE_MAX_ENTRIES = 2048
_embed_cache: dict[str, list[float]] = {}
_embed_cache_lock = threading.Lock()


def embed_texts(inputs: list[str]) -> Optional[list[list[float]]]:
    """Embed inputs via LM Studio; returns None if the embedding API fails."""
    with _embed_cache_lock:
        vectors = {text: _embed_cache[text] for text in inputs if text in _embed_cache}
    missing = list(dict.fromkeys(text for text in inputs if text not in vectors))
    if missing:
        # The /v1/embeddings endpoint can't set a TTL, so when the model isn't
        # loaded yet, pre-load it via `lms` with the configured default TTL —
        # otherwise this request would leave it resident with no idle timeout.
        ensure_embedding_model_loaded()
        try:
            response = client.embeddings.create(
                model=CONFIG["rag"]["embedding_model"], input=missing)
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            logger.warning("Embedding request failed: %s", getattr(exc, "message", exc))
            return None
        vectors.update(zip(missing, (item.embedding for item in response.data)))
        with _embed_cache_lock:
            for text in missing:
                _embed_cache[text] = vectors[text]
            while len(_embed_cache) > EMBED_CACHE_MAX_ENTRIES:
                _embed_cache.pop(next(iter(_embed_cache)))
    return [vectors[text] for text in inputs]


def index_document(filename: str, text: str, pages: Optional[int]) -> Optional[dict[str, Any]]:
    """Chunk + embed a document, store it, and return upload metadata (or None)."""
    config = CONFIG["rag"]
    chunks = chunk_text(text, config["chunk_chars"], config["chunk_overlap"])
    if not chunks:
        return None
    embeddings = embed_texts([f"search_document: {chunk}" for chunk in chunks])
    if embeddings is None:
        return None
    doc_id = uuid.uuid4().hex
    DOC_STORE[doc_id] = {"filename": filename, "chunks": chunks, "embeddings": embeddings}
    logger.info("Indexed %s for RAG: %s chunks, doc_id=%s", filename, len(chunks), doc_id)
    return {
        "filename": filename,
        "pages": pages,
        "chars": len(text),
        "truncated": False,
        "rag": True,
        "doc_id": doc_id,
        "chunks": len(chunks),
    }


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def retrieve_context(doc_ids: list[str], query: str) -> str:
    """Return the top-k most relevant chunks across the given documents."""
    pool: list[tuple[str, str, list[float]]] = []
    for doc_id in doc_ids:
        doc = DOC_STORE.get(doc_id)
        if not doc:
            continue
        for chunk, embedding in zip(doc["chunks"], doc["embeddings"]):
            pool.append((doc["filename"], chunk, embedding))
    if not pool:
        return ""
    query_embedding = embed_texts([f"search_query: {query}"])
    if query_embedding is None:
        # Embedding the query failed: fall back to the first few chunks so the
        # model still sees something relevant rather than nothing.
        ranked = pool[: CONFIG["rag"]["top_k"]]
    else:
        q = query_embedding[0]
        qnorm = math.sqrt(_dot(q, q)) or 1.0
        scored = sorted(
            pool,
            key=lambda item: _dot(q, item[2]) / ((math.sqrt(_dot(item[2], item[2])) or 1.0) * qnorm),
            reverse=True,
        )
        ranked = scored[: CONFIG["rag"]["top_k"]]
    blocks = [f"[From {filename}]\n{chunk}" for filename, chunk, _ in ranked]
    return "\n\n".join(blocks)


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


def model_max_context_length(model_key: str) -> Optional[int]:
    """The model's own maximum context length, or None if it can't be determined."""
    try:
        for model in fetch_native_models():
            if model.get("key") == model_key:
                max_context = model.get("max_context_length")
                return int(max_context) if max_context else None
    except (httpx.HTTPError, ValueError, TypeError):
        logger.warning("Could not resolve max context length for %s", model_key)
    return None


def resolve_lms_cli() -> Optional[str]:
    """Locate LM Studio's `lms` CLI, or None if it can't be found.

    Prefers the configured path, then $PATH, then LM Studio's default install
    location. The CLI is needed because the REST load endpoint can't set a TTL.
    """
    configured = CONFIG["model_management"].get("lms_cli_path")
    if configured:
        return configured if os.path.isfile(configured) else None
    found = shutil.which("lms")
    if found:
        return found
    default = os.path.expanduser("~/.lmstudio/bin/lms")
    return default if os.path.isfile(default) else None


def load_via_lms(model: str, ttl: Optional[int], context_length: Optional[int]) -> None:
    """Load a model through `lms load`, raising RuntimeError on failure.

    Unlike the REST endpoint, `lms load` can set both a TTL and a context length
    on the loaded instance.
    """
    lms = resolve_lms_cli()
    if lms is None:
        raise FileNotFoundError("lms CLI not found")
    cmd = [lms, "load", model, "--yes"]
    if ttl:
        cmd += ["--ttl", str(ttl)]
    if context_length:
        cmd += ["--context-length", str(context_length)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CONFIG["model_management"]["load_timeout_seconds"],
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"lms load timed out after {exc.timeout:.0f}s") from exc
    if result.returncode != 0:
        raise RuntimeError(parse_lms_error(result.stderr or result.stdout or ""))


def parse_lms_error(output: str) -> str:
    """Pull the human-readable error out of `lms` output.

    `lms` prints the error first, then boilerplate suggestions ("To see a list…",
    indented command examples). Keep the leading message lines and drop the rest.
    """
    message_lines: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("To ") or raw.startswith("    "):  # suggestion boilerplate
            break
        message_lines.append(line)
    return " ".join(message_lines) or "lms load failed"


def lms_remaining_ttl_seconds() -> dict[str, int]:
    """Map loaded-instance identifier -> remaining idle TTL (seconds), via `lms ps`.

    LM Studio's REST model list omits remaining_ttl_seconds for embedding
    instances even when a TTL is set, so the UI can't show it. `lms ps --json`
    reports ttlMs + lastUsedTime for every type; we derive the live countdown.
    Best-effort: returns {} if the CLI is missing or anything goes wrong.
    """
    lms = resolve_lms_cli()
    if lms is None:
        return {}
    try:
        result = subprocess.run(
            [lms, "ps", "--json"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return {}
        entries = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return {}
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    remaining: dict[str, int] = {}
    for entry in entries if isinstance(entries, list) else []:
        identifier = entry.get("identifier")
        ttl_ms = entry.get("ttlMs")
        last_used = entry.get("lastUsedTime")
        if not identifier or not ttl_ms or last_used is None:
            continue
        seconds = round((ttl_ms - (now_ms - last_used)) / 1000)
        remaining[identifier] = max(0, seconds)
    return remaining


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

    # The REST list omits remaining_ttl_seconds for embedding instances, so fill
    # the gap from `lms ps` (which reports a TTL for every instance type). Only
    # shell out when some loaded instance is actually missing its TTL, to avoid a
    # subprocess on every poll when there's nothing to backfill.
    needs_ttl_lookup = any(
        instance.get("remaining_ttl_seconds") is None
        for model in raw_models
        for instance in model.get("loaded_instances", [])
    )
    lms_ttls = lms_remaining_ttl_seconds() if needs_ttl_lookup else {}

    models = [
        {
            "key": model.get("key"),
            "display_name": model.get("display_name") or model.get("key"),
            "type": model.get("type"),
            "params": model.get("params_string"),
            "quantization": (model.get("quantization") or {}).get("name"),
            "size_bytes": model.get("size_bytes"),
            "max_context_length": model.get("max_context_length"),
            "capabilities": model.get("capabilities") or {},
            "loaded_instances": [
                {
                    "id": instance.get("id"),
                    "context_length": (instance.get("config") or {}).get("context_length"),
                    # LM Studio reports the live countdown to auto-eviction; absent
                    # when the instance has no TTL, and always absent for embedding
                    # instances — so fall back to the value derived from `lms ps`.
                    "remaining_ttl_seconds": (
                        instance.get("remaining_ttl_seconds")
                        if instance.get("remaining_ttl_seconds") is not None
                        else lms_ttls.get(instance.get("id"))
                    ),
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
    # Resolve the context length to use. An explicit request wins; otherwise fall
    # back to the configured default, and finally to the model's own maximum.
    # Without this, LM Studio loads every model at a hardcoded 8192 rather than
    # the model's real default context window.
    context_length = request.context_length
    if context_length is None:
        context_length = CONFIG["model_management"].get("default_context_length")
    if context_length is None:
        context_length = model_max_context_length(request.model)

    # Resolve the idle TTL: explicit request wins, else the configured default
    # (only when auto-unload is on). 0/None means no TTL.
    mgmt = CONFIG["model_management"]
    ttl = request.ttl_seconds
    if ttl is None and mgmt.get("auto_unload_by_default"):
        ttl = mgmt.get("default_ttl_seconds")

    # The REST load endpoint can't set a TTL (it rejects a "ttl" key), so when a
    # TTL is wanted we load through the `lms` CLI, which sets both TTL and context
    # length. Without a TTL — or if the CLI is missing — fall back to REST.
    if ttl:
        try:
            load_via_lms(request.model, ttl, context_length)
        except FileNotFoundError:
            logger.warning(
                "lms CLI not found; loading %s via REST without a TTL. "
                "Set model_management.lms_cli_path to enable TTL on manual loads.",
                request.model)
        except RuntimeError as exc:
            logger.error("Failed to load %s via lms: %s", request.model, exc)
            return JSONResponse(status_code=502, content={"error": str(exc)})
        else:
            logger.info("Loaded model %s via lms (ttl=%s, context_length=%s)",
                        request.model, ttl, context_length)
            return JSONResponse(content={
                "model": request.model,
                "ttl_seconds": ttl,
                "context_length": context_length,
            })

    payload: dict[str, Any] = {"model": request.model}
    if context_length is not None:
        payload["context_length"] = context_length
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
    logger.info("Loaded model %s via REST (context_length=%s, no ttl)",
                request.model, context_length)
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


def last_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text") or "")
    return ""


def prepend_to_last_user_turn(
    messages: list[dict[str, Any]], preamble: str
) -> list[dict[str, Any]]:
    """Return a copy of `messages` with `preamble` prepended to the last user turn."""
    augmented = [dict(message) for message in messages]
    for message in reversed(augmented):
        if message.get("role") != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            message["content"] = preamble + content
        elif isinstance(content, list):
            message["content"] = [{"type": "text", "text": preamble}, *content]
        break
    return augmented


def apply_rag(messages: list[dict[str, Any]], doc_ids: list[str]) -> list[dict[str, Any]]:
    """Prepend retrieved chunks to the final user turn for the given documents."""
    query = last_user_query(messages)
    context = retrieve_context(doc_ids, query) if query else ""
    if not context:
        return messages
    preamble = ("Use the following excerpts from the user's uploaded "
                "documents to answer.\n\n" + context + "\n\n")
    return prepend_to_last_user_turn(messages, preamble)


# ---------- Web search ----------

class WebSearchError(Exception):
    """Raised when a web search cannot be performed; the chat continues without it."""


def search_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
    """Search DuckDuckGo via the ddgs package (no API key required)."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise WebSearchError(
            "the ddgs package is not installed — run: pip install ddgs"
        ) from exc
    raw = None
    last_error: Optional[Exception] = None
    # ddgs rotates between backends, some of which fail transiently; one
    # retry picks a different backend and resolves most hiccups. A slow
    # failure was a timeout, not a hiccup — retrying would double the stall.
    for _ in range(2):
        started = time.monotonic()
        try:
            with DDGS(timeout=CONFIG["web_search"]["timeout_seconds"]) as ddgs:
                raw = ddgs.text(query, max_results=max_results,
                                backend=CONFIG["web_search"]["backends"],
                                region=CONFIG["web_search"]["region"])
            break
        except Exception as exc:  # noqa: BLE001 — ddgs raises library-specific errors
            last_error = exc
            if time.monotonic() - started > 5:
                break
    if raw is None:
        raise WebSearchError(f"DuckDuckGo search failed: {last_error}") from last_error
    return [
        {
            "title": item.get("title") or "",
            "url": item.get("href") or "",
            "snippet": item.get("body") or "",
        }
        for item in raw or []
    ]


def search_searxng(query: str, max_results: int) -> list[dict[str, str]]:
    """Search a self-hosted SearXNG instance (its settings must allow format=json)."""
    base_url = (CONFIG["web_search"]["searxng_base_url"] or "").rstrip("/")
    if not base_url:
        raise WebSearchError("searxng_base_url is not configured")
    params = {"q": query, "format": "json"}
    language = _region_language()
    if language:
        params["language"] = language
    try:
        response = httpx.get(
            f"{base_url}/search",
            params=params,
            timeout=CONFIG["web_search"]["timeout_seconds"],
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except (httpx.HTTPError, ValueError) as exc:
        raise WebSearchError(f"SearXNG request failed: {exc}") from exc
    return [
        {
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "snippet": item.get("content") or "",
        }
        for item in results[:max_results]
    ]


def run_web_search(query: str, max_results: Optional[int] = None) -> list[dict[str, str]]:
    """Dispatch to the configured search provider."""
    provider = CONFIG["web_search"]["provider"]
    max_results = max_results or CONFIG["web_search"]["max_results"]
    if provider == "searxng":
        return search_searxng(query, max_results)
    if provider == "duckduckgo":
        return search_duckduckgo(query, max_results)
    raise WebSearchError(f"unknown web search provider: {provider}")


SEARCH_QUERY_SYSTEM_PROMPT = (
    "You turn chat messages into web search queries. Given a conversation, "
    "write one concise search query (a few keywords, no quotes, no boolean "
    "operators) that would find the information needed to answer the user's "
    "last message. Resolve pronouns and follow-up references from the "
    "conversation so the query stands on its own. Write the query in the "
    "language of the user's question (English is fine for technical topics). "
    "If the question is time-sensitive and names no date, include the "
    "current year. Reply with the query text only — no explanations."
)


def quick_completion(model: str, system: str, user: str, max_tokens: int = 300) -> Optional[str]:
    """One short, non-reasoning helper completion; None on failure.

    Used for the auxiliary LLM calls around web search (query rewriting,
    research planning/reflection) — callers must fall back gracefully.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            # Reasoning models would otherwise spend the budget thinking.
            extra_body={"reasoning_effort": "none"},
        )
    except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
        logger.warning("Helper completion failed: %s", getattr(exc, "message", exc))
        return None
    text = (completion.choices[0].message.content or "").strip()
    # Some reasoning models emit a <think>...</think> block before the answer.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text or None


def rewrite_search_query(model: str, messages: list[dict[str, Any]], fallback: str) -> str:
    """Distill the conversation's last question into a short search query.

    Chat messages make poor queries verbatim, and follow-ups ("what about in
    winter?") are meaningless without the preceding turns. Falls back to the
    raw question on any failure — a usable query beats no search.
    """
    lines = []
    for message in messages[-6:]:
        if message.get("role") not in ("user", "assistant"):
            continue
        text = message_text(message).strip()
        if not text:
            continue
        # Keep the tail: attachment text is prepended, the question comes last.
        lines.append(f"{message['role']}: {text[-800:]}")
    # Helpers don't know the date; without it, "latest X" queries get the
    # model's training-era year appended and steer the engine to stale pages.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = quick_completion(model, SEARCH_QUERY_SYSTEM_PROMPT,
                            f"Today is {today}.\n\n" + "\n\n".join(lines),
                            max_tokens=200)
    text = " ".join(text.split()).strip("\"'") if text else ""
    if not text or len(text) > 200:
        return fallback
    return text


class _PageTextExtractor(HTMLParser):
    """Collect the readable text of an HTML page, its <title>, and link hrefs.

    Text fragments are buffered and joined into one line per block element,
    so inline markup ("costs <b>5</b> francs") can't split sentences — split
    sentences chunk poorly and embed worse. Only reliably-closed non-content
    containers are skipped; boilerplate like nav/footer text is left in
    because malformed pages often leave those tags unclosed (which would
    swallow the whole document) — the reranker sorts low-value chunks out.
    """

    SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "head"}
    BLOCK_TAGS = {
        "p", "div", "li", "ul", "ol", "br", "hr", "h1", "h2", "h3", "h4",
        "h5", "h6", "tr", "td", "th", "table", "section", "article",
        "header", "footer", "nav", "main", "aside", "blockquote", "pre",
        "form", "figure", "figcaption", "details", "summary",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._buffer: list[str] = []
        self.parts: list[str] = []
        self.links: list[str] = []
        self.title = ""

    def _flush(self) -> None:
        if self._buffer:
            self.parts.append(" ".join(self._buffer))
            self._buffer = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)
        if tag in self.BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in self.BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if not self._skip_depth and data.strip():
            self._buffer.append(" ".join(data.split()))

    def close(self) -> None:
        super().close()
        self._flush()


def _region_language() -> str:
    """The configured search region as an HTTP language tag ("de-CH"); "" if none."""
    region = CONFIG["web_search"].get("region") or "wt-wt"
    if region == "wt-wt" or "-" not in region:
        return ""
    country, _, language = region.partition("-")
    return f"{language}-{country.upper()}"


_LANGUAGE_TAG = _region_language()
BROWSER_HEADERS = {
    # Some sites reject the default httpx user agent outright.
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    # Content-negotiating sites should serve the configured region's language.
    "Accept-Language": (
        f"{_LANGUAGE_TAG},{_LANGUAGE_TAG.split('-')[0]};q=0.9,en;q=0.8"
        if _LANGUAGE_TAG else "en"),
}

# A pasted PDF link is read with the same pypdf path uploads use; this caps
# how much of one is downloaded.
MAX_PDF_BYTES = 20 * 1024 * 1024


# Fetched-page cache (url -> (timestamp, fetch_page result)): follow-up
# questions about a pasted site re-crawl the same pages every turn; serving
# them from a short-lived cache makes those turns near-instant. Successful
# fetches only, so transient failures retry naturally.
PAGE_CACHE_MAX_ENTRIES = 256
_page_cache: dict[str, tuple[float, tuple[str, list[str], str, str]]] = {}
_page_cache_lock = threading.Lock()


def _fetch_pdf_text(response: httpx.Response, max_chars: int, deadline: float) -> str:
    """Read a streamed PDF response (size- and time-bounded) and extract its text."""
    data = bytearray()
    if int(response.headers.get("content-length") or 0) > MAX_PDF_BYTES:
        return ""
    for piece in response.iter_bytes():
        data.extend(piece)
        # A byte-trickling server resets httpx's per-read timeout forever; the
        # wall-clock deadline is what actually bounds the download.
        if len(data) > MAX_PDF_BYTES or time.monotonic() > deadline:
            return ""
    extracted = extract_pdf_text(bytes(data))
    if not isinstance(extracted, tuple):
        return ""
    return extracted[0][:max_chars]


def fetch_page(url: str) -> tuple[str, list[str], str, str]:
    """Fetch one page; returns (readable text, hrefs, final URL after redirects, title).

    HTML bodies are streamed with both a size cap and a wall-clock deadline,
    so a link to a large download or a byte-trickling server can't exhaust
    memory or stall the batch; the size cap leaves generous headroom for
    markup around max_page_chars of extractable text. PDF links are read via
    the same pypdf path uploads use.
    """
    config = CONFIG["web_search"]
    ttl = config["page_cache_ttl_seconds"]
    if ttl:
        with _page_cache_lock:
            entry = _page_cache.get(url)
            if entry and time.monotonic() - entry[0] < ttl:
                return entry[1]
    parser = _PageTextExtractor()
    started = time.monotonic()
    timed_out = False
    with httpx.stream("GET", url, timeout=config["timeout_seconds"],
                      follow_redirects=True, headers=BROWSER_HEADERS) as response:
        response.raise_for_status()
        final_url = str(response.url)
        content_type = response.headers.get("content-type", "html")
        if "pdf" in content_type or urlsplit(final_url).path.lower().endswith(".pdf"):
            try:
                text = _fetch_pdf_text(response, config["max_page_chars"],
                                       started + config["timeout_seconds"])
            except httpx.HTTPError:
                text = ""
            result = (text, [], final_url, "")
        elif "html" not in content_type:
            result = ("", [], final_url, "")
        else:
            max_html_chars = config["max_page_chars"] * 20
            read = 0
            try:
                for piece in response.iter_text():
                    parser.feed(piece)
                    read += len(piece)
                    if read >= max_html_chars:
                        break
                    if time.monotonic() - started > config["timeout_seconds"]:
                        timed_out = True
                        break
                parser.close()
            except Exception:  # noqa: BLE001 — never let a weird page kill the fetch
                pass
            title = " ".join(parser.title.split())[:200]
            result = ("\n".join(parser.parts)[: config["max_page_chars"]],
                      parser.links, final_url, title)
    # Cache only successful, complete fetches: a deadline-truncated or empty
    # result (size-capped PDF, JS shell, transient content-type miss) should
    # be retried on the next turn, not served from cache for the whole TTL.
    if ttl and not timed_out and result[0]:
        with _page_cache_lock:
            _page_cache[url] = (time.monotonic(), result)
            while len(_page_cache) > PAGE_CACHE_MAX_ENTRIES:
                _page_cache.pop(next(iter(_page_cache)))
    return result


def fetch_page_texts(results: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    """Download the result pages concurrently; returns {url: (text, title)}.

    Failures are skipped, and the whole batch is bounded by a deadline so one
    slow page can't hold the answer hostage — whatever finished in time is
    used. Stragglers keep downloading in the background and land in the page
    cache for the next turn.
    """
    texts: dict[str, tuple[str, str]] = {}
    urls = [item["url"] for item in results if item["url"]]
    if not urls:
        return texts
    pool = ThreadPoolExecutor(max_workers=min(5, len(urls)))
    futures = {pool.submit(fetch_page, url): url for url in urls}
    try:
        deadline = CONFIG["web_search"]["timeout_seconds"] + 5
        for future in as_completed(futures, timeout=deadline):
            url = futures[future]
            try:
                text, _, _, title = future.result()
                texts[url] = (text, title)
            except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
                logger.info("Skipping web result %s: %s", url, exc)
    except FuturesTimeoutError:
        # concurrent.futures.TimeoutError, a separate class from the builtin
        # on Python 3.10 — catch it explicitly so the deadline degrades
        # gracefully instead of escaping and aborting the whole answer.
        logger.info("Page fetch deadline hit; continuing with %d of %d pages",
                    len(texts), len(urls))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return texts


def web_search_context(
    model: str, query: str, rank_query: str,
    only_hosts: Optional[set[str]] = None,
) -> str:
    """Search the web and build a context block for prompt injection.

    The engine is asked for twice the configured results and a quick LLM
    relevance check drops the off-topic ones (engines fuzzy-match keywords)
    before any page is fetched. Page chunks are embed-ranked against
    `rank_query` — the user's actual question — not the keyword query sent
    to the engine. `only_hosts` drops results from other sites: engines
    don't always honor a site: operator strictly, so it's enforced here too.
    """
    config = CONFIG["web_search"]
    results = run_web_search(query, config["max_results"] * 2)
    if only_hosts:
        results = [item for item in results if _same_site(item["url"], only_hosts)]
    if len(results) > config["max_results"]:
        # A NONE verdict ([]) shouldn't wipe a successful search down to no
        # context — keep the engine's own top results in that case.
        results = filter_relevant_results(model, rank_query, results) or results
    results = results[: config["max_results"]]
    if not results:
        return ""
    # The snippets always go in: for weather/news-style queries they often
    # carry the direct answer, while the pages behind them are JS shells
    # whose extracted text contains nothing useful.
    snippets = "Search results:\n\n" + "\n\n".join(
        f"[{index}] [{item['title']}]({item['url']})\n{item['snippet']}"
        for index, item in enumerate(results, 1)
    )
    if not config["fetch_pages"]:
        return snippets
    page_texts = fetch_page_texts(results)
    rag = CONFIG["rag"]
    pool: list[tuple[dict[str, str], str]] = []
    for item in results:
        text = page_texts.get(item["url"], ("", ""))[0]
        for chunk in chunk_text(text, rag["chunk_chars"], rag["chunk_overlap"]):
            pool.append((item, chunk))
    if not pool:
        return snippets
    top = rank_chunks_by_query(rank_query, pool, config["top_k"])
    if top is None:
        return snippets
    excerpts = "\n\n".join(
        f"[{item['title']}]({item['url']})\n{chunk}" for item, chunk in top)
    return snippets + "\n\nRelevant excerpts from the result pages:\n\n" + excerpts


def score_chunks_by_query(
    query: str, pool: list[tuple[dict[str, str], str]]
) -> Optional[list[tuple[dict[str, str], str]]]:
    """All (result, chunk) pairs sorted by similarity to the query; None if embeddings fail."""
    chunk_embeddings = embed_texts([f"search_document: {chunk}" for _, chunk in pool])
    query_embedding = embed_texts([f"search_query: {query}"]) if chunk_embeddings else None
    if not chunk_embeddings or not query_embedding:
        return None
    q = query_embedding[0]
    qnorm = math.sqrt(_dot(q, q)) or 1.0
    scored = sorted(
        zip(pool, chunk_embeddings),
        key=lambda entry: _dot(q, entry[1]) / ((math.sqrt(_dot(entry[1], entry[1])) or 1.0) * qnorm),
        reverse=True,
    )
    return [pair for pair, _ in scored]


def select_diverse_chunks(
    ranked: list[tuple[dict[str, str], str]], top_k: int
) -> list[tuple[dict[str, str], str]]:
    """Greedy top_k from ranked, skipping near-duplicates of already-picked chunks.

    Pages repeat taglines, teasers and shared boilerplate; without this the
    top slots fill with copies of the same paragraph instead of new facts.
    """
    if top_k <= 0:
        return []
    selected: list[tuple[dict[str, str], str]] = []
    picked_words: list[set[str]] = []
    for item, chunk in ranked:
        words = set(chunk.lower().split())
        if words and any(
                len(words & seen) / len(words | seen) > 0.8 for seen in picked_words):
            continue
        selected.append((item, chunk))
        picked_words.append(words)
        if len(selected) == top_k:
            break
    return selected


def rank_chunks_by_query(
    query: str, pool: list[tuple[dict[str, str], str]], top_k: int
) -> Optional[list[tuple[dict[str, str], str]]]:
    """Embed-rank (result, chunk) pairs against the query; None if embeddings fail."""
    ranked = score_chunks_by_query(query, pool)
    return None if ranked is None else select_diverse_chunks(ranked, top_k)


def apply_web_search(
    messages: list[dict[str, Any]], model: str, query: str, user_query: str,
    only_hosts: Optional[set[str]] = None, supplementary: bool = False,
    fallback_query: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Prepend web search results for `query` to the final user turn.

    `user_query` is the user's actual question (used to judge relevance and
    rank page chunks); it is passed explicitly because RAG may already have
    prepended document context to the message. `supplementary` marks the
    results as secondary to pages the user linked themselves. When the query
    finds nothing and a `fallback_query` is given, that is searched once —
    an over-specific rewrite or an unsupported site: operator shouldn't kill
    the whole search.
    """
    # Pasted URLs and instruction words are noise for relevance judgments.
    rank_query = " ".join(URL_PATTERN.sub(" ", user_query).split()) or user_query
    # Search engines cap query length; the first sentences carry the intent.
    context = web_search_context(model, query[:400], rank_query, only_hosts)
    if not context and fallback_query and fallback_query.strip() != query.strip():
        logger.info("Search found nothing for %r; retrying with %r",
                    query, fallback_query)
        context = web_search_context(model, fallback_query[:400], rank_query, only_hosts)
    if not context:
        return messages
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    precedence = (
        " The pages the user linked (provided separately) are the primary "
        "sources; use these results only to fill gaps." if supplementary else ""
    )
    preamble = (
        f"Web search results retrieved {today} (UTC). Use them to answer when "
        "relevant and cite sources inline as markdown links." + precedence
        + "\n\n" + context + "\n\n"
    )
    return prepend_to_last_user_turn(messages, preamble)


# ---------- Linked pages (user-pasted URLs) ----------

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")

# Pages up to this size are inlined whole; larger ones are embed-ranked
# against the question and only the best excerpts are injected.
LINKED_PAGE_INLINE_CHARS = 6000


def extract_urls(text: str) -> list[str]:
    """URLs pasted in the message — deduped, order-preserving, capped at 5."""
    urls: list[str] = []
    for match in URL_PATTERN.findall(text):
        url = match.rstrip(".,;:!?")
        # A trailing ")" is usually prose punctuation — "(see https://x.ch)" —
        # unless the URL itself contains an opening paren (Wikipedia titles).
        if url.endswith(")") and "(" not in url:
            url = url[:-1]
        # Drop URLs urlsplit can't parse (stray brackets and the like) — they
        # would raise ValueError deep inside the pipeline otherwise.
        if not _host_of(url):
            continue
        if url not in urls:
            urls.append(url)
    return urls[:5]


def _host_of(url: str) -> str:
    """Lowercased hostname without port or a www. prefix; "" when unparsable."""
    try:
        return (urlsplit(url).hostname or "").removeprefix("www.")
    except ValueError:
        return ""


def _same_site(url: str, hosts: set[str]) -> bool:
    """Whether `url` belongs to one of the (www-stripped) hosts, subdomains included."""
    host = _host_of(url)
    return bool(host) and any(host == h or host.endswith("." + h) for h in hosts)


# Link targets that can't contain readable HTML — skipped during a crawl so
# they don't burn fetch attempts (the content-type check is the backstop).
NON_HTML_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".heic", ".svg",
    ".ico", ".css", ".js", ".pdf", ".zip", ".gz", ".mp3", ".m4a", ".wav",
    ".mp4", ".mov", ".webm", ".xml", ".rss", ".atom", ".json", ".txt",
    ".docx", ".xlsx", ".pptx",
)

# Query parameters that never change the content behind a link: share
# buttons, comment-reply anchors, click tracking.
JUNK_QUERY_PARAMS = ("share", "replytocom", "fbclid", "gclid", "ref")


def crawl_dedup_key(url: str) -> str:
    """Canonical URL form for crawl dedup.

    Collapses scheme, www., trailing-slash and share/tracking-parameter
    variants, which all serve the page already fetched (WordPress-style
    ?share=twitter / ?replytocom= / ?utm_* links) and would otherwise burn
    the page budget on duplicates.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").removeprefix("www.")
    params = sorted(
        (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not (key.startswith("utm_") or key in JUNK_QUERY_PARAMS))
    query = "&".join(f"{key}={value}" for key, value in params)
    return f"{host}{parts.path.rstrip('/')}" + (f"?{query}" if query else "")


def collect_site_links(page_url: str, hrefs: list[str], hosts: set[str]) -> list[str]:
    """Absolute same-site links from a page's hrefs, in document order."""
    links: list[str] = []
    for href in hrefs:
        try:
            absolute = urldefrag(urljoin(page_url, href)).url
            parts = urlsplit(absolute)
        except ValueError:
            # Malformed href (stray brackets and the like) — skip it.
            continue
        if parts.scheme not in ("http", "https"):
            continue
        if not _same_site(absolute, hosts):
            continue
        if parts.path.lower().endswith(NON_HTML_EXTENSIONS):
            continue
        links.append(absolute)
    # Shallow paths first (stable, so document order breaks ties): top-level
    # pages like /about and /episodes describe a site, while deep paths are
    # leaves — single posts, share/like action pages.
    links.sort(key=lambda link: urlsplit(link).path.count("/"))
    return links


def crawl_site_texts(start_urls: list[str], max_pages: int) -> dict[str, tuple[str, str]]:
    """Crawl pasted pages + same-site links breadth-first; {url: (text, title)}.

    A single linked page rarely answers questions about a site as a whole —
    the "what is this podcast about?" answer lives on /about or the episode
    pages, not the homepage — so a deep dive follows the site's own links.
    """
    hosts = {h for h in (_host_of(url) for url in start_urls) if h}
    seen = {crawl_dedup_key(url) for url in start_urls}
    frontier = list(start_urls)
    texts: dict[str, tuple[str, str]] = {}
    attempts = 0
    # Only pages that yielded text fill the page budget; the attempts cap
    # bounds the crawl when many links error out or turn out not to be HTML.
    while frontier and len(texts) < max_pages and attempts < max_pages * 4:
        batch = frontier[: max_pages - len(texts)]
        frontier = frontier[len(batch):]
        attempts += len(batch)
        with ThreadPoolExecutor(max_workers=min(5, len(batch))) as pool:
            futures = {pool.submit(fetch_page, url): url for url in batch}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    text, hrefs, final_url, title = future.result()
                except Exception as exc:  # noqa: BLE001 — a weird link must not kill the crawl
                    logger.info("Skipping crawled page %s: %s", url, exc)
                    continue
                if url in start_urls:
                    # Redirects off a pasted page (http→https, a shortlink)
                    # define the site the user actually meant.
                    if _host_of(final_url):
                        hosts.add(_host_of(final_url))
                elif not _same_site(final_url, hosts):
                    # A same-site link that redirected off-site — don't let
                    # foreign content masquerade as the linked site's.
                    logger.info("Skipping off-site redirect %s -> %s", url, final_url)
                    continue
                seen.add(crawl_dedup_key(final_url))
                if text:
                    texts[url] = (text, title)
                # Resolve hrefs against the final URL: after a redirect (say
                # /blog -> /blog/), relative links are wrong otherwise.
                for link in collect_site_links(final_url, hrefs, hosts):
                    key = crawl_dedup_key(link)
                    if key not in seen:
                        seen.add(key)
                        frontier.append(link)
    return texts


LINKED_USAGE_SYSTEM_PROMPT = (
    "You classify how a chat message wants its web links used. Reply with two "
    "words: ONLY or ANY, then SITE or PAGE.\n"
    "ONLY = the user restricts sources to the linked pages (\"use no other "
    "pages\", \"only this site as a source\"). ANY = no such restriction.\n"
    "SITE = the linked website as a whole matters (deep dive, what is this "
    "site/podcast/company about). PAGE = the linked page alone is enough.\n"
    "The message may be in any language.\n\n"
    "Examples:\n"
    "\"summarize https://x.ch/article\" -> ANY PAGE\n"
    "\"deep dive through https://x.ch use no other pages as a source\" -> ONLY SITE\n"
    "\"what is this podcast about? https://pod.ch\" -> ANY SITE\n"
    "\"only use https://shop.ch as source: what do they sell?\" -> ONLY SITE\n"
    "\"tell me what https://firm.ch offers\" -> ANY SITE\n"
    "\"is https://a.ch/post consistent with current research?\" -> ANY PAGE\n"
    "Reply with the two words only."
)


def classify_linked_usage(model: str, query: str) -> tuple[bool, bool]:
    """(restrict to linked pages, crawl the site) for a message with URLs.

    An explicit source restriction implies crawling: when the linked site is
    the only allowed source, one page of it is rarely enough. Falls back to
    (False, False) — fetch the pasted pages, keep searching — on any failure,
    which is the pre-classification behavior.
    """
    text = quick_completion(model, LINKED_USAGE_SYSTEM_PROMPT, query[:2000],
                            max_tokens=200)
    if not text:
        return False, False
    upper = text.upper()
    # Anchor on the first "ONLY/ANY SITE/PAGE" pair, falling back to the
    # first occurrence of each label alone — the label words also appear in
    # explanatory prose ("the user did not say to use only…"), which must
    # not flip the result.
    pair = re.search(r"\b(ONLY|ANY)\b\W+(SITE|PAGE)\b", upper)
    if pair:
        restrict = pair.group(1) == "ONLY"
        return restrict, restrict or pair.group(2) == "SITE"
    scope = re.search(r"\b(ONLY|ANY)\b", upper)
    depth = re.search(r"\b(SITE|PAGE)\b", upper)
    restrict = scope is not None and scope.group(1) == "ONLY"
    return restrict, restrict or (depth is not None and depth.group(1) == "SITE")


def strip_repeated_lines(
    texts: dict[str, tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """Collapse site furniture across crawled pages of one site.

    Nav menus, footers and player controls repeat on most pages; the first
    page keeps one copy (it may carry signal — a tagline, the site name) and
    the rest drop it, so the chunk pool isn't crowded with duplicates.
    """
    if len(texts) < 4:
        return texts
    threshold = len(texts) // 2 + 1
    counts: Counter[str] = Counter()
    for text, _ in texts.values():
        counts.update(set(text.splitlines()))
    repeated = {line for line, count in counts.items() if count >= threshold}
    if not repeated:
        return texts
    kept_once: set[str] = set()
    cleaned: dict[str, tuple[str, str]] = {}
    for url, (text, title) in texts.items():
        lines = []
        for line in text.splitlines():
            if line in repeated:
                if line in kept_once:
                    continue
                kept_once.add(line)
            lines.append(line)
        cleaned[url] = ("\n".join(lines), title)
    return cleaned


def linked_pages_context(query: str, urls: list[str], crawl: bool) -> tuple[str, list[str]]:
    """Fetch user-pasted URLs; returns (context block, pasted URLs that failed).

    Search engines never index small or new sites, so pasted links are
    fetched directly instead of searched for. Short pasted pages are inlined
    whole; everything else — long pages, and with `crawl` the same-site pages
    linked from the pasted ones — is embed-ranked against the question
    (leading text of the pasted pages when embeddings are unavailable).
    """
    config = CONFIG["web_search"]
    rag = CONFIG["rag"]
    if crawl:
        texts = crawl_site_texts(urls, max(len(urls), int(config["crawl_max_pages"])))
        texts = strip_repeated_lines(texts)
    else:
        texts = fetch_page_texts([{"title": url, "url": url, "snippet": ""} for url in urls])
    failed = [url for url in urls if not texts.get(url, ("", ""))[0]]

    def label(url: str) -> str:
        # A real <title> makes a better citation link than a bare URL, which
        # for a crawled /episode-417 says nothing about the content.
        title = texts.get(url, ("", ""))[1]
        return f"[{title}]({url})" if title else f"[{url}]"

    blocks: list[str] = []
    pool: list[tuple[dict[str, str], str]] = []
    overflow: list[str] = []  # pasted pages too long to inline
    for url in urls:
        text = texts.get(url, ("", ""))[0]
        if not text:
            continue
        if len(text) <= LINKED_PAGE_INLINE_CHARS:
            blocks.append(f"{label(url)}\n{text}")
        else:
            overflow.append(url)
            pool.extend(({"url": url, "label": label(url)}, chunk) for chunk
                        in chunk_text(text, rag["chunk_chars"], rag["chunk_overlap"]))
    for url, (text, _) in texts.items():
        if url in urls or not text:
            continue
        pool.extend(({"url": url, "label": label(url)}, chunk) for chunk
                    in chunk_text(text, rag["chunk_chars"], rag["chunk_overlap"]))
    if pool:
        capped = pool[:RESEARCH_POOL_CAP]
        # The pasted URLs and instruction words ("use no other source") are
        # noise for similarity ranking; the topical words should match.
        rank_query = " ".join(URL_PATTERN.sub(" ", query).split()) or query
        ranked = score_chunks_by_query(rank_query, capped)
        if ranked is None:
            # No embeddings: keep the lead of the long pasted pages, as before.
            blocks.extend(f"{label(url)}\n{texts[url][0][:LINKED_PAGE_INLINE_CHARS]}"
                          for url in overflow)
        else:
            top = select_diverse_chunks(ranked, config["linked_top_k"])
            # Every page the user pasted deserves representation — don't let
            # one long page crowd the others out of the excerpt slots.
            extras = []
            for url in overflow:
                if any(item["url"] == url for item, _ in top):
                    continue
                best = next((pair for pair in ranked if pair[0]["url"] == url), None)
                if best:
                    extras.append(best)
            if extras:
                top = top[: max(config["linked_top_k"] - len(extras), 0)] + extras
            excerpts = "\n\n".join(f"{item['label']}\n{chunk}" for item, chunk in top)
            blocks.append("Most relevant excerpts from the linked pages:\n\n" + excerpts)
    return "\n\n".join(blocks), failed


# ---------- Web research (multi-round search) ----------

RESEARCH_PLAN_SYSTEM_PROMPT = (
    "You are a research planner. Given a question, reply with the distinct web "
    "search queries (each a few keywords) that together cover what is needed "
    "to answer it thoroughly. Write queries in the question's language "
    "(English is fine for technical topics). One query per line — no "
    "numbering, no explanations."
)

RESEARCH_REFLECT_SYSTEM_PROMPT = (
    "You are reviewing the sources a web research pass has collected so far. "
    "Decide what important information is still missing to answer the user's "
    "question. Reply with new web search queries that would fill the gaps, "
    "one per line (a few keywords each, no numbering, no explanations) — or "
    "reply with the single word DONE if the sources already cover the question."
)

RESEARCH_FILTER_SYSTEM_PROMPT = (
    "You are filtering web search results for relevance. Given a question and "
    "a numbered list of results, reply with only the numbers of the results "
    "that are actually about the question's topic, comma-separated "
    "(for example: 1, 3, 4). Reply with the single word NONE if no result is "
    "relevant."
)

# Backstop for the chunk pool across all rounds, so a misconfigured
# rounds/results combination can't produce an enormous embedding request.
RESEARCH_POOL_CAP = 400


def parse_query_lines(text: str, limit: int) -> list[str]:
    """Extract up to `limit` queries from LLM output, one per line.

    Tolerates the numbering/bullets models add despite instructions.
    """
    queries: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^[\s\-*•]*(?:\d+[.)])?\s*", "", line).strip().strip("\"'")
        if not line or line.upper() == "DONE" or len(line) > 200:
            continue
        if line.lower() not in (q.lower() for q in queries):
            queries.append(line)
    return queries[:limit]


def plan_research_queries(model: str, query: str, limit: int) -> list[str]:
    """Decompose the question into sub-queries; the raw question on failure."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = quick_completion(
        model, RESEARCH_PLAN_SYSTEM_PROMPT,
        f"Today is {today}.\nQuestion: {query}\n\n"
        f"Reply with at most {limit} queries.")
    return parse_query_lines(text or "", limit) or [query[:400]]


def filter_relevant_results(
    model: str, query: str, results: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Drop results that are off-topic for the question.

    Search engines fuzzy-match keywords (an "Asahi Linux" query returns beer
    company pages), and embeddings can't tell those apart — the shared brand
    token dominates the similarity. A quick LLM call separates them reliably.
    Returns the input unchanged when the call fails; an empty list when the
    model judges nothing relevant.
    """
    if len(results) <= 1:
        return results
    candidates = results[:30]
    listing = "\n".join(
        f"{index}. {item['title']} — {item['snippet'][:120]}"
        for index, item in enumerate(candidates, 1))
    text = quick_completion(
        model, RESEARCH_FILTER_SYSTEM_PROMPT,
        f"Question: {query}\n\nResults:\n{listing}")
    if not text:
        return results
    if text.strip().upper().startswith("NONE"):
        return []
    indices = {int(number) for number in re.findall(r"\d+", text)}
    if not indices:
        return results
    kept = [item for index, item in enumerate(candidates, 1) if index in indices]
    return kept or results


def reflect_research_gaps(
    model: str, query: str, results: list[dict[str, str]], limit: int
) -> list[str]:
    """Ask the model which follow-up queries would fill gaps; [] means stop."""
    sources = "\n".join(
        f"- {item['title']}: {item['snippet'][:150]}" for item in results[:25])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = quick_completion(
        model, RESEARCH_REFLECT_SYSTEM_PROMPT,
        f"Today is {today}.\nQuestion: {query}\n\n"
        f"Sources collected so far:\n{sources}\n\n"
        f"Reply with at most {limit} queries, or DONE.")
    if not text or text.strip().upper().startswith("DONE"):
        return []
    return parse_query_lines(text, limit)


def research_context(model: str, query: str) -> Iterator[str]:
    """Run the multi-round research pipeline, yielding progress statuses.

    Returns (via the generator's StopIteration value) the assembled context
    block, or "" when nothing usable was found. The loop is bounded by code —
    the model only proposes queries, it never decides how long to run.
    """
    config = CONFIG["web_search"]
    rag = CONFIG["rag"]
    max_rounds = max(1, int(config["research_max_rounds"]))
    per_round = max(1, int(config["research_queries_per_round"]))
    seen_queries: set[str] = set()
    seen_urls: set[str] = set()
    results_found: list[dict[str, str]] = []
    pool: list[tuple[dict[str, str], str]] = []
    yield "🔬 Planning research…"
    queries = plan_research_queries(model, query, per_round)
    for round_no in range(1, max_rounds + 1):
        queries = [q for q in queries if q.lower() not in seen_queries][:per_round]
        if not queries:
            break
        round_results: list[dict[str, str]] = []
        for sub_query in queries:
            seen_queries.add(sub_query.lower())
        listing = ", ".join(f"“{q}”" for q in queries)
        yield f"🔬 Round {round_no}/{max_rounds}: searching {listing}…"
        # The sub-queries are independent — search them concurrently (the
        # executor exit joins them; results merge in query order so the URL
        # dedup stays deterministic).
        with ThreadPoolExecutor(max_workers=min(3, len(queries))) as search_pool:
            searches = [(q, search_pool.submit(run_web_search, q[:400]))
                        for q in queries]
        for sub_query, search in searches:
            try:
                found = search.result()
            except WebSearchError as exc:
                logger.warning("Research query %r failed: %s", sub_query, exc)
                continue
            new_results = [r for r in found if r["url"] and r["url"] not in seen_urls]
            seen_urls.update(r["url"] for r in new_results)
            round_results.extend(new_results)
        # Filter before fetching so off-topic pages cost nothing and the
        # reflection step reasons over clean sources.
        if len(round_results) > 1:
            yield f"🔬 Round {round_no}/{max_rounds}: checking relevance…"
            kept = filter_relevant_results(model, query, round_results)
            if len(kept) < len(round_results):
                logger.info("Research relevance filter dropped %d of %d results",
                            len(round_results) - len(kept), len(round_results))
            round_results = kept
        results_found.extend(round_results)
        if config["fetch_pages"] and round_results and len(pool) < RESEARCH_POOL_CAP:
            yield f"🔬 Round {round_no}/{max_rounds}: reading {len(round_results)} pages…"
            texts = fetch_page_texts(round_results)
            for item in round_results:
                for chunk in chunk_text(texts.get(item["url"], ("", ""))[0],
                                        rag["chunk_chars"], rag["chunk_overlap"]):
                    pool.append((item, chunk))
            if len(pool) >= RESEARCH_POOL_CAP:
                logger.info("Research chunk pool capped at %d chunks", RESEARCH_POOL_CAP)
        if round_no < max_rounds:
            yield "🔬 Reviewing findings for gaps…"
            queries = reflect_research_gaps(model, query, results_found, per_round)
    if not results_found:
        return ""
    # The snippet list doubles as the source index for citations; cap it so a
    # large rounds × results configuration can't flood the prompt.
    snippets = "Search results:\n\n" + "\n\n".join(
        f"[{index}] [{item['title']}]({item['url']})\n{item['snippet']}"
        for index, item in enumerate(results_found[:15], 1))
    if not pool:
        return snippets
    yield "🔬 Ranking findings…"
    top = rank_chunks_by_query(query, pool[:RESEARCH_POOL_CAP], config["research_top_k"])
    if top is None:
        return snippets
    excerpts = "\n\n".join(
        f"[{item['title']}]({item['url']})\n{chunk}" for item, chunk in top)
    return snippets + "\n\nRelevant excerpts from the result pages:\n\n" + excerpts


def stream_completion(request: ChatRequest) -> Iterator[str]:
    """Yield Server-Sent Events with incremental completion tokens.

    Errors are reported as an SSE `error` field because the HTTP status
    is already committed once streaming has started.
    """
    stream = None
    try:
        outgoing = [message.model_dump() for message in request.messages]
        # Capture the user's question before RAG prepends document context to it.
        query = last_user_query(outgoing)
        web_config = CONFIG["web_search"]
        mode = request.web_search if (web_config["enabled"] and query) else None
        # URLs pasted into the message are fetched directly — search engines
        # don't index small sites, so searching for them finds nothing.
        have_linked_pages = False
        restrict_to_linked = False
        urls = extract_urls(query) if mode else []
        if urls:
            plural = "s" if len(urls) > 1 else ""
            yield f"data: {json.dumps({'status': f'🔗 Reading {len(urls)} linked page{plural}…'})}\n\n"
            if web_config["page_cache_ttl_seconds"]:
                # The pasted pages are needed whatever the classifier decides —
                # warm the page cache while it runs instead of after.
                with ThreadPoolExecutor(max_workers=2) as overlap:
                    verdict = overlap.submit(classify_linked_usage, request.model, query)
                    overlap.submit(fetch_page_texts,
                                   [{"title": u, "url": u, "snippet": ""} for u in urls])
                    restrict_to_linked, crawl_site = verdict.result()
            else:
                restrict_to_linked, crawl_site = classify_linked_usage(request.model, query)
            # A bare domain root rarely answers anything by itself — explore
            # the site regardless of how the classifier read the message.
            if all(urlsplit(u).path in ("", "/") and not urlsplit(u).query for u in urls):
                crawl_site = True
            if crawl_site:
                status = (f"🔗 Exploring {urlsplit(urls[0]).netloc} "
                          f"(up to {web_config['crawl_max_pages']} pages)…")
                yield f"data: {json.dumps({'status': status})}\n\n"
            page_block, failed = linked_pages_context(query, urls, crawl_site)
            if failed:
                notice = "Could not read: " + ", ".join(failed)
                yield f"data: {json.dumps({'notice': notice})}\n\n"
            if page_block:
                have_linked_pages = True
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if restrict_to_linked:
                    preamble = (
                        f"Content fetched {today} (UTC) from the pages linked "
                        "in the user's message. The user asked to use only "
                        "these pages as sources: answer strictly from this "
                        "content, cite the pages as markdown links, and when "
                        "the content does not cover something, say so instead "
                        "of drawing on other sources or prior knowledge.\n\n"
                        + page_block + "\n\n")
                else:
                    preamble = (
                        f"Content fetched {today} (UTC) from the pages linked in "
                        "the user's message. Use it to answer and cite the pages "
                        "as markdown links.\n\n" + page_block + "\n\n")
                outgoing = prepend_to_last_user_turn(outgoing, preamble)
        only_hosts: list[str] = []
        if restrict_to_linked:
            if have_linked_pages:
                # The user asked for the linked pages to be the only sources —
                # honor it by skipping the engine search entirely.
                yield f"data: {json.dumps({'status': '🔒 Using only the linked pages as sources…'})}\n\n"
                mode = None
            else:
                # The sites couldn't be read directly; a search restricted to
                # their domains is the closest available honoring of the request.
                only_hosts = list(dict.fromkeys(
                    h for h in (_host_of(u) for u in urls) if h))
                mode = "search"
        search_query = None
        fallback_query = None
        if mode == "search":
            search_query = query
            # The raw question (minus any pasted URL) is the last-resort query
            # when an over-specific rewrite or a site: operator finds nothing.
            clean_question = " ".join(URL_PATTERN.sub(" ", query).split()) or query
            fallback_query = clean_question
            if web_config["rewrite_query"]:
                yield f"data: {json.dumps({'status': '🌐 Preparing web search…'})}\n\n"
                # Rewrite from the pre-RAG messages so retrieved document
                # chunks don't leak into the search query.
                search_query = rewrite_search_query(request.model, outgoing, query)
                if search_query != query:
                    logger.info("Search query rewritten: %r", search_query)
            if only_hosts:
                # Engines don't reliably parse multiple site: operators; hint
                # with the first host and let web_search_context's host filter
                # enforce the full restriction. site: support is flaky on small
                # sites, so the fallback searches the host as a plain keyword.
                search_query = f"site:{only_hosts[0]} {search_query}"
                fallback_query = f"{only_hosts[0]} {clean_question}"
        if request.doc_ids:
            outgoing = apply_rag(outgoing, request.doc_ids)
        if mode == "search":
            searching = ("🌐 Searching the web & reading the top results…"
                         if web_config["fetch_pages"] else "🌐 Searching the web…")
            yield f"data: {json.dumps({'status': searching})}\n\n"
            before_search = outgoing
            try:
                outgoing = apply_web_search(
                    outgoing, request.model, search_query, query,
                    set(only_hosts) or None, supplementary=have_linked_pages,
                    fallback_query=fallback_query)
            except WebSearchError as exc:
                # Degrade gracefully: surface the failure as a toast and let
                # the model answer from its own knowledge. When the message's
                # linked pages were already fetched, a failed search on top of
                # them is routine (tiny sites aren't indexed) — log only.
                logger.warning("Web search failed: %s", exc)
                # The honesty preamble below speaks for the restricted case;
                # a "answering without web results" toast would contradict it.
                if not have_linked_pages and not only_hosts:
                    notice = f"Web search failed — {exc}. Answering without web results."
                    yield f"data: {json.dumps({'notice': notice})}\n\n"
            if only_hosts and outgoing is before_search:
                # A source restriction the pipeline couldn't satisfy: the
                # linked sites were unreadable AND the restricted search found
                # nothing. Tell the model rather than let it silently answer
                # from prior knowledge, which the restriction forbids.
                preamble = (
                    "The user restricted sources to "
                    f"{', '.join(urls)}, but those pages could not be "
                    "retrieved and a search restricted to them found nothing. "
                    "Tell the user the requested sources are unavailable; do "
                    "not answer from other sources or prior knowledge.\n\n")
                outgoing = prepend_to_last_user_turn(outgoing, preamble)
        elif mode == "research":
            # Drain the research generator, forwarding its progress statuses;
            # its return value is the assembled context block.
            research = research_context(request.model, query)
            research_block = ""
            while True:
                try:
                    status = next(research)
                except StopIteration as stop:
                    research_block = stop.value or ""
                    break
                yield f"data: {json.dumps({'status': status})}\n\n"
            if research_block:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                precedence = (
                    " The pages the user linked (provided separately) are the "
                    "primary sources; these findings supplement them."
                    if have_linked_pages else "")
                preamble = (
                    f"Research findings retrieved {today} (UTC) by a "
                    "multi-query web search. Write a thorough, well-structured "
                    "answer from them: use headings where helpful, cite "
                    "sources inline as markdown links, and note anything the "
                    "findings leave open." + precedence + "\n\n"
                    + research_block + "\n\n")
                outgoing = prepend_to_last_user_turn(outgoing, preamble)
            elif not have_linked_pages:
                notice = "Web research found no usable sources. Answering without web results."
                yield f"data: {json.dumps({'notice': notice})}\n\n"
        extra_body: dict[str, Any] = {}
        if request.reasoning_effort:
            extra_body["reasoning_effort"] = request.reasoning_effort
        # Attach the idle TTL so a model first loaded by this message auto-unloads.
        # LM Studio applies a request's ttl only when the request JIT-loads the
        # model — an already-loaded instance keeps whatever TTL it loaded with
        # (the load button handles that case via the lms CLI). The request's
        # ttl_seconds (from the UI's auto-unload control) wins; otherwise fall
        # back to the configured default. 0 means auto-unload off, so omit ttl.
        mgmt = CONFIG["model_management"]
        ttl = request.ttl_seconds
        if ttl is None and mgmt.get("auto_unload_by_default"):
            ttl = mgmt.get("default_ttl_seconds")
        if ttl:
            extra_body["ttl"] = ttl
        stream = client.chat.completions.create(
            model=request.model,
            messages=outgoing,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
            extra_body=extra_body or None,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # LM Studio streams thinking separately for reasoning models.
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield f"data: {json.dumps({'reasoning': reasoning})}\n\n"
            if delta.content:
                yield f"data: {json.dumps({'content': delta.content})}\n\n"
        yield "data: [DONE]\n\n"
    except GeneratorExit:
        # The client disconnected (stop button, closed tab). Re-raise so the
        # generator shuts down; `finally` closes the upstream connection,
        # which makes LM Studio abort the generation instead of completing
        # it in the background.
        logger.info("Chat stream cancelled by the client.")
        raise
    except (APIConnectionError, APITimeoutError):
        logger.exception("LM Studio unreachable during chat completion")
        yield f"data: {json.dumps(connection_error_payload())}\n\n"
    except APIStatusError as exc:
        logger.exception("LM Studio returned an error during chat completion")
        yield f"data: {json.dumps({'error': f'LM Studio error: {exc.message}'})}\n\n"
    except Exception:  # noqa: BLE001 — never leak a raw traceback into the stream
        logger.exception("Unexpected error during chat completion")
        yield f"data: {json.dumps({'error': 'Unexpected server error. Check the application logs.'})}\n\n"
    finally:
        if stream is not None:
            stream.close()


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_completion(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/tts")
def tts(request: TTSRequest) -> Response:
    """Stream Kokoro-synthesized speech as raw float32 PCM.

    Audio starts flowing as soon as the first sentence is rendered, so the
    client can begin playback without waiting for the whole text. The format is
    advertised via X-* headers (mono, 32-bit float, little-endian, at the
    configured sample rate) since raw PCM carries no header of its own.
    """
    try:
        pipeline = get_tts_pipeline()
    except TTSUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    if not request.text.strip():
        return JSONResponse(status_code=400, content={"error": "No text to read aloud."})

    chunks = stream_tts(pipeline, request)
    # Pull the first chunk here, in the handler, so an empty result (e.g. an
    # unknown voice, or a missing espeak-ng for out-of-vocabulary words) becomes
    # a JSON error instead of a committed 200 with a silent empty body — once
    # StreamingResponse begins, the status line can no longer change.
    try:
        first_chunk = next(chunks)
    except StopIteration:
        return JSONResponse(status_code=500,
                            content={"error": "Speech synthesis produced no audio."})

    def body() -> Iterator[bytes]:
        yield first_chunk
        yield from chunks

    sample_rate = int(CONFIG["tts"]["sample_rate"])
    headers = {
        "X-Sample-Rate": str(sample_rate),
        "X-Audio-Channels": "1",
        "X-Audio-Format": "f32le",
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",  # don't let a proxy buffer the live stream
    }
    return StreamingResponse(body(), media_type="application/octet-stream", headers=headers)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- TLS ----------

SNAKEOIL_CERT = CERTS_DIR / "snakeoil.crt"
SNAKEOIL_KEY = CERTS_DIR / "snakeoil.key"
SNAKEOIL_VALIDITY_DAYS = 825


def generate_snakeoil_cert(cert_path: Path, key_path: Path) -> None:
    """Create a self-signed certificate for localhost/LAN use."""
    import datetime
    import ipaddress
    import socket

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LocalMind (self-signed)"),
    ])
    san_entries: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address("::1")),
    ]
    hostname = socket.gethostname()
    if hostname and hostname != "localhost":
        san_entries.append(x509.DNSName(hostname))
        if not hostname.endswith(".local"):
            san_entries.append(x509.DNSName(f"{hostname}.local"))
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=SNAKEOIL_VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    os.chmod(key_path, 0o600)
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    logger.info("Generated self-signed certificate %s (valid %s days)",
                cert_path, SNAKEOIL_VALIDITY_DAYS)


def snakeoil_cert_is_usable() -> bool:
    """True if the existing snakeoil cert parses and is not about to expire."""
    import datetime

    from cryptography import x509

    if not (SNAKEOIL_CERT.is_file() and SNAKEOIL_KEY.is_file()):
        return False
    try:
        certificate = x509.load_pem_x509_certificate(SNAKEOIL_CERT.read_bytes())
    except (ValueError, OSError):
        return False
    expiry = getattr(certificate, "not_valid_after_utc", None)
    if expiry is None:
        expiry = certificate.not_valid_after.replace(tzinfo=datetime.timezone.utc)
    return expiry > datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)


def resolve_tls_files() -> Optional[tuple[str, str]]:
    """Return (cert_file, key_file) when TLS is enabled, else None.

    Explicitly configured paths are validated and never silently replaced;
    with no paths configured, a snakeoil certificate is (re)generated.
    """
    tls = CONFIG["tls"]
    if not tls["enabled"]:
        return None
    cert_file, key_file = tls["cert_file"], tls["key_file"]
    if cert_file or key_file:
        if not (cert_file and key_file):
            raise SystemExit("TLS misconfiguration: cert_file and key_file "
                             "must both be set (or both left null for a "
                             "self-signed certificate).")
        # Relative paths resolve against the project directory, not the cwd.
        cert_path = Path(cert_file) if Path(cert_file).is_absolute() else BASE_DIR / cert_file
        key_path = Path(key_file) if Path(key_file).is_absolute() else BASE_DIR / key_file
        if not (cert_path.is_file() and key_path.is_file()):
            raise SystemExit(f"TLS misconfiguration: {cert_path} or {key_path} "
                             "does not exist.")
        return str(cert_path), str(key_path)
    if not snakeoil_cert_is_usable():
        logger.warning("No TLS certificate configured — generating a "
                       "self-signed (snakeoil) certificate. Browsers will "
                       "show a security warning.")
        generate_snakeoil_cert(SNAKEOIL_CERT, SNAKEOIL_KEY)
    return str(SNAKEOIL_CERT), str(SNAKEOIL_KEY)


if __name__ == "__main__":
    import uvicorn

    ssl_kwargs: dict[str, Any] = {}
    tls_files = resolve_tls_files()
    if tls_files:
        ssl_kwargs = {"ssl_certfile": tls_files[0], "ssl_keyfile": tls_files[1]}
        logger.info("HTTPS enabled on https://%s:%s (cert: %s)",
                    CONFIG["host"], CONFIG["port"], tls_files[0])
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"], **ssl_kwargs)
