# LocalMind

A lightweight, self-hosted chat interface for a local [LM Studio](https://lmstudio.ai/) server.

LocalMind is a small FastAPI application with a vanilla HTML/CSS/JS frontend. It talks to LM Studio's OpenAI-compatible API and gives you:

- **Dynamic model selection** — the dropdown is populated with the models currently loaded in LM Studio.
- **Real-time streaming** — responses render token-by-token via Server-Sent Events.
- **Stop generation** — the Send button turns into ⏹ Stop while a reply streams; stopping keeps the partial answer in the conversation and aborts the upstream LM Studio request so the model stops burning tokens.
- **Markdown rendering** — assistant replies render as sanitized Markdown (GFM) live while streaming: headings, lists, tables, blockquotes, links, and fenced code blocks with a hover copy button. Uses vendored [marked](https://github.com/markedjs/marked) + [DOMPurify](https://github.com/cure53/DOMPurify), so it works fully offline.
- **Parameter controls** — adjust `temperature` and `max_tokens` from a settings panel before sending.
- **System prompt presets** — save, name, and switch between reusable system prompts (e.g. *Code reviewer*, *Translator*, *Socratic tutor*) from the ⚙ settings panel. Presets are persisted server-side (SQLite); edit the prompt inline for a one-off, or save it back to a preset. The last-used preset is remembered across reloads.
- **Model management** — load and unload models straight from the UI (📦 panel), with:
  - **idle TTL (auto-unload)** — automatically unload a model after it has been idle for a configurable number of seconds,
  - **context length override** — load a model with a custom context window,
  - **live status** — loaded/not-loaded badges, parameter count, quantization, on-disk size, max context, and an estimate of memory in use, auto-refreshed while the panel is open,
  - **multiple instances** — load a model more than once and unload each instance individually,
  - **unload all** — free all memory with one click.
- **Image generation (capability-detected)** — a 🎨 mode in the composer that generates images from text prompts via an OpenAI-compatible `/images/generations` endpoint. The UI detects automatically whether the connected server supports it; generated images render in the chat with click-to-zoom and a download link, and the image size is adjustable in settings.
- **LLM prompt enhancement** — optionally let the selected chat model (e.g. Gemma in LM Studio) rewrite your short idea into a rich, detailed image prompt before generation; the enhanced prompt is shown under the image. If enhancement fails, the original prompt is used and you are notified.
- **File attachments (📎 / drag & drop)** — attach PDFs, plain-text, and code files; their text is extracted server-side and injected into the chat context so you can ask questions about them. Limits are configurable, with clear errors for password-protected or scanned PDFs.
- **Image attachments & pasted screenshots** — attach images via the picker, drag & drop, or simply paste a screenshot (Cmd+V); they are sent to vision-capable models (like Gemma) as multimodal input.
- **Thinking control** — a settings option to turn model reasoning off/low/high per conversation; when a model thinks, the thought stream renders live in a collapsible 💭 block that folds away once the answer starts.
- **Saved conversations** — every chat is persisted server-side (SQLite) and listed in a sidebar; reload-safe, multi-device over the LAN, with rename and delete. The user turn is saved before streaming begins, so nothing is lost mid-generation.
- **Conversation search** — a search box in the sidebar runs full-text search (SQLite FTS5, search-as-you-type) over the titles and message text of all saved conversations and shows matching snippets; press Escape to clear.
- **Edit & regenerate** — hover any of your messages to ✏️ edit it (attachments are preserved) or any answer to ↻ regenerate it; the conversation is truncated at that point and re-streamed, with a confirmation if later messages would be discarded.
- **Context-window meter** — a gauge in the composer estimates the prompt size (system prompt, history, attachments, retrieved RAG chunks, and your draft) against the loaded model's actual context window, turning yellow at 80% and red at 95%.
- **Document retrieval (RAG)** — long uploads are chunked, embedded with a local embedding model, and queried by relevance per question (only the top matches enter the prompt) — so you can chat about a big PDF without blowing the context window. Small docs are still inlined.
- **Web search & research (🌐)** — a Web control in the composer grounds answers in live web results, in two modes. *Search*: the chat model distills your question (with conversation context, so follow-ups work) into a search query, the top result pages are downloaded and embed-ranked against it with the local embedding model, and the best excerpts are injected into the prompt with sources cited inline as links. *Research*: a bounded multi-round pipeline — the model plans sub-queries, results are LLM-filtered for relevance and their pages read, the model reviews the findings for gaps and searches again, and everything is synthesized into a structured, cited answer (slower, deeper). In either mode, URLs pasted into the message are fetched directly and their content injected, so you can ask about pages search engines don't index; ask for a deep dive and same-site links are crawled too, or ask to use only the linked pages and the engine search is skipped entirely. Works out of the box via DuckDuckGo (no API key), or point it at a self-hosted [SearXNG](https://docs.searxng.org/) instance to keep queries on your own infrastructure. If a search fails, you get a toast and the model answers without web results.
- **Text-to-speech (🔊)** — every message gets a speaker button that reads it aloud using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), a small, high-quality voice model that runs **fully offline** with no external API. It is pinned to the **CPU** so it never takes VRAM from your LLM, and the audio **streams sentence-by-sentence** (raw PCM played via the Web Audio API), so playback starts on the first sentence instead of waiting for the whole reply. Optional: if the `kokoro` package isn't installed the button simply stays hidden.
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
│   ├── script.js           # Frontend logic (models, settings, SSE streaming)
│   └── vendor/             # Vendored libraries (marked, DOMPurify)
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
| `documents.max_file_size_mb` | Maximum upload size for attached files | `25` |
| `documents.max_text_chars` | Inlined (non-RAG) text is truncated beyond this length | `20000` |
| `rag.enabled` | Retrieve large documents by relevance instead of inlining them | `true` |
| `rag.embedding_model` | LM Studio embedding model used to index/search documents | `text-embedding-nomic-embed-text-v1.5` |
| `rag.min_chars_for_rag` | Documents at least this long are indexed for retrieval | `8000` |
| `rag.chunk_chars` | Target size of each retrieved chunk | `1200` |
| `rag.chunk_overlap` | Overlap between consecutive chunks | `200` |
| `rag.top_k` | Number of chunks injected into the prompt per question | `5` |
| `tls.enabled` | Serve the app over HTTPS | `false` |
| `tls.cert_file` | Path to a PEM certificate (`null` = auto-generate self-signed) | `null` |
| `tls.key_file` | Path to the matching PEM private key | `null` |

The config file path itself can be overridden with the `LOCALMIND_CONFIG` environment variable (useful for Docker or running multiple instances).

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

### Conversations (saved automatically)

Conversations are persisted server-side in a SQLite database under `data/` (gitignored), so they survive reloads, crashes, and closing the tab — and because the app can bind `0.0.0.0`, the same history is reachable from any device on your LAN.

- The ☰ button toggles a sidebar listing past conversations, newest first.
- The first exchange of a new chat auto-creates and auto-titles a conversation (from your first message); the user turn is saved *before* the model starts streaming, so nothing is lost if you reload mid-generation.
- Click a conversation to resume it (messages, image thumbnails, and document tags are restored); **double-click** to rename; the 🗑 on each row deletes it (with confirmation).
- The topbar 🗑 starts a fresh chat without deleting anything.

> The conversation endpoints are unauthenticated, like the rest of the app. On a trusted home LAN this is usually fine; if your network is shared, bind to `127.0.0.1` (set `host` in `config.json`) or put LocalMind behind an authenticating reverse proxy. Stored conversations include uploaded document text and pasted images.

### Document retrieval (RAG)

Small attached documents are inlined into the prompt as before. Documents at least `rag.min_chars_for_rag` characters long are instead **indexed for retrieval**: on upload they are chunked and embedded with a local LM Studio embedding model (`text-embedding-nomic-embed-text-v1.5` by default, loaded on demand), and the attachment chip shows “🔍 searchable”. For each question, only the most relevant chunks (`rag.top_k`) are embedded-matched and injected — so you can chat about a long PDF without it consuming the whole context window or being truncated. Follow-up questions keep using the same documents for the rest of the session. If the embedding model is unavailable, the upload falls back to inlining the (truncated) text, and retrieval misses degrade to answering without the document rather than erroring.

### Web search & research

Set the **🌐 Web** control in the composer to *Search* or *Research* to ground answers in live web results.

**Pasted links are fetched directly** (in either mode): URLs in your message are downloaded and their content injected into the prompt — search engines don't index small or new sites, so asking about one would otherwise find nothing. A quick LLM call classifies what you want from the links: if you ask for a **deep dive into a site** (or paste a bare domain), same-site links found on the page are crawled too — breadth-first, up to `crawl_max_pages` pages — so "what is this podcast about?" finds the answer on /about or the episode pages, not just the homepage. If you ask to **use only the linked pages as sources** ("use no other pages"), the engine search is skipped entirely and the model is told to answer strictly from the fetched content (if the site can't be fetched, a `site:`-restricted search is the fallback). Otherwise the search still runs alongside, with the results marked as secondary to your linked pages. Short pasted pages are inlined whole; everything else is embed-ranked against your question and the best `linked_top_k` excerpts are injected. Pasted PDF links are read too (via the same extractor uploads use). Fetched pages and their crawled neighbours are cached for `page_cache_ttl_seconds`, so follow-up questions about the same site answer near-instantly. Links that can't be fetched are reported in a toast.

**Search** runs a quick pipeline (~5–10s):

1. **Query rewriting** — the selected chat model distills your message and the recent turns into a concise search query (one quick `reasoning_effort: none` call), so conversational phrasing and follow-ups like *"and what about the next one?"* still search well. The helper is told today's date and to write the query in your language, so *"latest …"* questions aren't anchored to the model's training year and German questions aren't silently translated to English.
2. **Search** — the query goes to the configured provider: DuckDuckGo via the [ddgs](https://pypi.org/project/ddgs/) package (default, no API key) or a self-hosted SearXNG instance, under the configured `region`. Twice `max_results` candidates are requested and a quick LLM relevance check drops off-topic ones before any page is downloaded (search engines fuzzy-match keywords). If the rewritten query finds nothing, the raw question is retried once.
3. **Read & rank** — the surviving result pages are downloaded in parallel (bounded by a batch deadline so one slow page can't stall the answer), stripped to text, chunked, and ranked against **your actual question** with the same local embedding model RAG uses; near-duplicate excerpts are skipped so the slots hold distinct facts. The result snippets plus the most relevant page excerpts are injected into the prompt, and the model is asked to cite sources as inline links.

**Research** runs a deeper, multi-round pipeline (typically 1–3 minutes) for questions that deserve more than one query:

1. **Plan** — the model decomposes your question into up to `research_queries_per_round` distinct sub-queries.
2. **Search & filter** — each sub-query is searched; a quick LLM call then drops off-topic results (search engines fuzzy-match keywords — an "Asahi Linux" query happily returns beer-company pages — and that junk would otherwise poison the answer).
3. **Read** — the pages behind the relevant results are downloaded and chunked.
4. **Reflect** — the model reviews what was found and proposes follow-up queries for the gaps; steps 2–4 repeat up to `research_max_rounds` times. The loop is bounded by code — the model only proposes queries, it never decides how long to run.
5. **Synthesize** — the best `research_top_k` excerpts (embed-ranked against your original question) are injected, and the model is asked to write a thorough, structured answer with inline citations.

Live progress ("Round 1/2: searching …", "checking relevance…", "reading 14 pages…") shows in the reply bubble while either mode runs, and the control's state is remembered across reloads.

> Tip: research answers are long — raise **Max tokens** in settings (4096+) so the synthesis isn't cut off. With `research_max_rounds: 1`, Research becomes a single fan-out pass without reflection (faster, still multi-query).

Configured under `web_search` in `config.json`:

- `provider` — `duckduckgo` (default, no API key or setup needed) or `searxng`.
- `searxng_base_url` — your [SearXNG](https://docs.searxng.org/) instance, e.g. `http://localhost:8888`, if you prefer to keep search queries on infrastructure you control. The instance must allow the JSON output format (`formats: [html, json]` in its `settings.yml`).
- `backends` — engine order for ddgs (default `duckduckgo, mojeek, brave`). Unknown names make ddgs fall back to its full random rotation, which includes weaker engines; `bing` is left out deliberately because its fuzzy keyword matches poison the context.
- `region` — search region as `country-language` (e.g. `ch-de`, `de-de`, `us-en`; default `wt-wt` = no region). Also sets the `Accept-Language` header for page fetches and the SearXNG language, so content-negotiating sites serve the right language. Set this (e.g. `ch-de`) if you mostly search in a non-English language.
- `rewrite_query` — set `false` to search with the raw chat message instead of the LLM-rewritten query (faster, noticeably worse for conversational questions).
- `fetch_pages` — set `false` to inject only the search snippets without downloading pages (faster, shallower answers). `top_k` controls how many ranked page excerpts are injected and `max_page_chars` how much text per page is considered.
- `max_results` / `timeout_seconds` — how many search results to use per query and how long to wait.
- `crawl_max_pages` / `linked_top_k` — pasted-URL handling: how many same-site pages a deep dive may fetch in total, and how many ranked excerpts from linked/crawled pages are injected.
- `page_cache_ttl_seconds` — how long fetched pages are reused (default 15 minutes), so follow-up questions about the same site don't re-download it every turn; `0` disables the cache.
- `research_max_rounds` / `research_queries_per_round` / `research_top_k` — research depth: how many plan-search-reflect rounds to run, how many sub-queries each round may search, and how many ranked excerpts the synthesis prompt receives (~12 ≈ 4k tokens; mind the model's context window).
- `enabled: false` hides the Web control entirely.

If a search fails (offline, rate-limited, misconfigured), a toast explains why and the model answers from its own knowledge instead of erroring. The same applies piecewise further down the pipeline: if the query rewrite fails the raw question is searched, and if page downloads or the embedding model are unavailable the snippets alone are injected.

> Tip: ranking page excerpts uses the RAG embedding model (`rag.embedding_model`), which defaults to an English-trained model. If you mostly search in another language, point it at a multilingual embedding model available in LM Studio (e.g. `text-embedding-nomic-embed-text-v2-moe`) so the right excerpts rank above boilerplate.

> Note: web search sends your (rewritten) question to the search provider (DuckDuckGo, or your SearXNG instance and the engines it queries), and the result pages are fetched from their sites. Leave the control off for conversations that should stay fully local.

> Note: fetching modern sites requires a Python built with TLS 1.3 support. Apple's CommandLineTools Python is built against LibreSSL 2.8 (no TLS 1.3) and fails on many sites with `TLSV1_ALERT_PROTOCOL_VERSION`; use a python.org or Homebrew build (which also satisfies the 3.10+ requirement) for the virtualenv.

### Attaching files and images

Use the 📎 button, drag & drop anywhere onto the page, or paste directly from the clipboard:

- **Documents** (PDF, `.txt`, `.md`, `.csv`, `.json`, and common code files): the text is extracted on the server (PDFs via `pypdf`, with page markers) and prepended invisibly to your next message — the chat shows only a small 📄 tag, but the model receives the full text and can answer follow-up questions about it. Long documents are truncated to `documents.max_text_chars`. Scanned (image-only) and password-protected PDFs are rejected with a clear message; OCR is not supported.
- **Images / screenshots**: attached images (or a pasted screenshot, Cmd+V) are sent as multimodal `image_url` content parts — this requires a vision-capable model (the Gemma models report `vision: true`). Thumbnails appear in the composer and in your message bubble.

Attachments ride along with the *next* message you send and can be removed (✕) before sending.

### Thinking (reasoning) control

The bar above the message box has two independent controls:

- **💭 Thinking** — *Default* (leave the model's own behavior untouched), *On*, or *Off*.
- **Effort** — *Minimal* to *Extra high*; applies only when Thinking is *On*.

Together they map onto LM Studio's `reasoning_effort` (*Off* → `none`, *On* → the chosen effort). While a reasoning model thinks, the thoughts stream into a collapsible 💭 block above the answer, which folds closed when the actual answer begins. Thoughts are display-only — they are not sent back as context in later turns. The controls hide in image mode, where they don't apply.

> Tip: reasoning consumes your `max_tokens` budget. If answers come back empty with thinking enabled, raise **Max tokens** in settings.

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

With **✨ Enhance prompt** enabled (visible in image mode), your idea is first rewritten into a detailed image prompt by the chat model selected in the dropdown — this is how your LM Studio LLM participates in image creation even though it cannot render images itself. The enhanced prompt is displayed under the generated image. Reasoning is disabled for this call (`reasoning_effort: none`) so reasoning models answer directly.

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

### Text-to-speech

Hover any message and click **🔊 Play** to hear it read aloud; the button becomes **⏹ Stop** while it plays. Speech is synthesized locally by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on the **CPU** (so it never competes with your LLM for VRAM) and **streams sentence-by-sentence** — `/api/tts` returns raw float32 PCM as each sentence is rendered, and the browser schedules the chunks gaplessly through the Web Audio API, so playback begins almost immediately instead of after the whole paragraph.

It is **optional and fully offline** — no external API is ever contacted. To enable it:

```bash
pip install kokoro espeakng-loader   # already in requirements.txt
```

Both ship with `requirements.txt`; `kokoro` pulls in torch + numpy. `espeakng-loader` bundles a prebuilt **espeak-ng** library (no system install needed) that Kokoro uses to pronounce out-of-vocabulary words — proper names, foreign words — that aren't in its dictionary. Without it those words are silently dropped, so it's recommended; if you'd rather use a system espeak-ng you already have, that is picked up automatically and the loader is skipped.

On first use the ~330 MB model is downloaded once from Hugging Face and cached; the pipeline is then warmed in the background at startup so the first click is fast. If `kokoro` isn't installed, the speaker button stays hidden and everything else works normally. Configure it under `tts` in `config.json`:

```json
"tts": {
  "enabled": true,
  "lang_code": "a",
  "voice": "af_heart",
  "speed": 1.0,
  "sample_rate": 24000,
  "max_chars": 5000,
  "warmup": true
}
```

`lang_code` selects the language (`a` American / `b` British English, `e` Spanish, `f` French, `i` Italian, `p` Portuguese, `h` Hindi, `j` Japanese, `z` Mandarin) and must match the locale of `voice` (see [Kokoro's voice list](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)). `max_chars` caps a single request so one synthesis can't tie up the CPU; set `warmup` to `false` to defer the model load until the first click.

## HTTPS

Set `tls.enabled` to `true` and restart:

- **With your own certificate** — point `tls.cert_file` and `tls.key_file` at your PEM files (e.g. from Let's Encrypt or an internal CA). Relative paths resolve against the project directory. If the files are missing or only one is set, the app refuses to start with a clear error rather than silently falling back.
- **Without a certificate** — a self-signed ("snakeoil") certificate is generated automatically under `certs/` (gitignored), covering `localhost`, `127.0.0.1`, `::1`, and your machine's hostname, valid for 825 days. It is reused across restarts and regenerated shortly before expiry. Browsers will show a one-time security warning you must accept — fine for home-lab use; use a real certificate (or a reverse proxy like Caddy/Traefik) for anything beyond that.

```json
"tls": {
  "enabled": true,
  "cert_file": null,
  "key_file": null
}
```

Then open <https://localhost:8000>. HTTPS applies when starting via `python app.py`; if you run uvicorn directly, pass the flags yourself: `uvicorn app:app --ssl-certfile certs/snakeoil.crt --ssl-keyfile certs/snakeoil.key`.

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
| `POST` | `/api/chat` | Streaming chat completion (SSE; supports multimodal content, `reasoning_effort`, and `doc_ids` for RAG) |
| `POST` | `/api/upload` | Extract/index a PDF/text/code file (multipart form); large docs return a `doc_id` for retrieval |
| `GET` | `/api/conversations` | List saved conversations (newest first) |
| `POST` | `/api/conversations` | Create an empty conversation |
| `GET` | `/api/conversations/{id}` | Full message history of a conversation |
| `PUT` | `/api/conversations/{id}` | Save messages and/or rename: `{"messages"?, "title"?}` |
| `DELETE` | `/api/conversations/{id}` | Delete a conversation |
| `GET` | `/api/system-prompts` | List saved system prompt presets (newest first) |
| `POST` | `/api/system-prompts` | Create a preset: `{"name", "content"}` |
| `PUT` | `/api/system-prompts/{id}` | Rename and/or edit a preset: `{"name"?, "content"?}` |
| `DELETE` | `/api/system-prompts/{id}` | Delete a preset |
| `GET` | `/api/models/manage` | All downloaded models with load state and details |
| `POST` | `/api/models/load` | Load a model: `{"model", "ttl_seconds"?, "context_length"?}` |
| `POST` | `/api/models/unload` | Unload one instance: `{"instance_id"}` |
| `POST` | `/api/models/unload-all` | Unload every loaded instance |
| `GET` | `/api/images/capability` | Whether the image API supports generation (`?refresh=true` re-probes) |
| `POST` | `/api/images` | Generate image(s): `{"prompt", "size"?, "n"?, "enhance_with"?}` |
| `POST` | `/api/tts` | Stream speech as raw float32 PCM (Kokoro): `{"text", "voice"?, "speed"?}`; `503` when TTS is unavailable |

## License

See [LICENSE](LICENSE).
