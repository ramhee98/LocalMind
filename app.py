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
import io
import json
import logging
import math
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import httpx
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
        "embedding_ttl_seconds": 600,
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

app = FastAPI(title="LocalMind", version="1.1.0")


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
        "documents": CONFIG["documents"],
        # Lets the UI estimate how many tokens retrieved chunks will add.
        "rag": {
            "enabled": CONFIG["rag"]["enabled"],
            "top_k": CONFIG["rag"]["top_k"],
            "chunk_chars": CONFIG["rag"]["chunk_chars"],
        },
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
        if end <= start:
            break
        start = end if end > start + step else start + step
    return chunks


def embed_texts(inputs: list[str]) -> Optional[list[list[float]]]:
    """Embed inputs via LM Studio; returns None if the embedding API fails."""
    # When LM Studio JIT-loads the embedding model for this request, tell it the
    # configured idle TTL so the model auto-unloads like UI-loaded models do.
    # Without this LM Studio keeps the JIT-loaded model resident indefinitely.
    extra_body: dict[str, Any] = {}
    ttl_seconds = CONFIG["rag"].get("embedding_ttl_seconds")
    if ttl_seconds is not None:
        extra_body["ttl"] = ttl_seconds
    try:
        response = client.embeddings.create(
            model=CONFIG["rag"]["embedding_model"], input=inputs,
            extra_body=extra_body or None)
    except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
        logger.warning("Embedding request failed: %s", getattr(exc, "message", exc))
        return None
    return [item.embedding for item in response.data]


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
            "capabilities": model.get("capabilities") or {},
            "loaded_instances": [
                {
                    "id": instance.get("id"),
                    "context_length": (instance.get("config") or {}).get("context_length"),
                    # LM Studio reports the live countdown to auto-eviction; absent
                    # when the instance was loaded without a TTL.
                    "remaining_ttl_seconds": instance.get("remaining_ttl_seconds"),
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

    # Resolve the context length to use. An explicit request wins; otherwise fall
    # back to the configured default, and finally to the model's own maximum.
    # Without this, LM Studio loads every model at a hardcoded 8192 rather than
    # the model's real default context window.
    context_length = request.context_length
    if context_length is None:
        context_length = CONFIG["model_management"].get("default_context_length")
    if context_length is None:
        context_length = model_max_context_length(request.model)
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
    logger.info("Loaded model %s (ttl=%s, context_length=%s)",
                request.model, request.ttl_seconds, context_length)
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


def apply_rag(messages: list[dict[str, Any]], doc_ids: list[str]) -> list[dict[str, Any]]:
    """Prepend retrieved chunks to the final user turn for the given documents."""
    query = last_user_query(messages)
    context = retrieve_context(doc_ids, query) if query else ""
    if not context:
        return messages
    augmented = [dict(message) for message in messages]
    for message in reversed(augmented):
        if message.get("role") != "user":
            continue
        preamble = ("Use the following excerpts from the user's uploaded "
                    "documents to answer.\n\n" + context + "\n\n")
        content = message["content"]
        if isinstance(content, str):
            message["content"] = preamble + content
        elif isinstance(content, list):
            message["content"] = [{"type": "text", "text": preamble}, *content]
        break
    return augmented


def stream_completion(request: ChatRequest) -> Iterator[str]:
    """Yield Server-Sent Events with incremental completion tokens.

    Errors are reported as an SSE `error` field because the HTTP status
    is already committed once streaming has started.
    """
    stream = None
    try:
        outgoing = [message.model_dump() for message in request.messages]
        if request.doc_ids:
            outgoing = apply_rag(outgoing, request.doc_ids)
        extra_body: dict[str, Any] = {}
        if request.reasoning_effort:
            extra_body["reasoning_effort"] = request.reasoning_effort
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
