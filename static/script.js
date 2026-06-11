"use strict";

(() => {
  const modelSelect = document.getElementById("model-select");
  const settingsToggle = document.getElementById("settings-toggle");
  const settingsPanel = document.getElementById("settings-panel");
  const exportToggle = document.getElementById("export-toggle");
  const exportMenu = document.getElementById("export-menu");
  const printRoot = document.getElementById("print-root");
  const clearChatButton = document.getElementById("clear-chat");
  const errorBanner = document.getElementById("error-banner");
  const errorText = document.getElementById("error-text");
  const errorRetry = document.getElementById("error-retry");
  const errorDismiss = document.getElementById("error-dismiss");
  const temperatureInput = document.getElementById("temperature");
  const temperatureValue = document.getElementById("temperature-value");
  const maxTokensInput = document.getElementById("max-tokens");
  const presetSelect = document.getElementById("preset-select");
  const presetNewButton = document.getElementById("preset-new");
  const presetRenameButton = document.getElementById("preset-rename");
  const presetDeleteButton = document.getElementById("preset-delete");
  const presetSaveButton = document.getElementById("preset-save");
  const presetStatus = document.getElementById("preset-status");
  const systemPromptInput = document.getElementById("system-prompt");
  const chatWindow = document.getElementById("chat-window");
  const emptyState = document.getElementById("empty-state");
  const messageInput = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");
  const modelsToggle = document.getElementById("models-toggle");
  const modelsPanel = document.getElementById("models-panel");
  const modelsList = document.getElementById("models-list");
  const modelsSummary = document.getElementById("models-summary");
  const modelsRefresh = document.getElementById("models-refresh");
  const unloadAllButton = document.getElementById("unload-all");
  const ttlEnabledInput = document.getElementById("ttl-enabled");
  const ttlSecondsInput = document.getElementById("ttl-seconds");
  const contextLengthInput = document.getElementById("context-length");
  const toastContainer = document.getElementById("toast-container");
  const composer = document.getElementById("composer");
  const imageToggle = document.getElementById("image-toggle");
  const imageSizeSelect = document.getElementById("image-size");
  const imageOptions = document.getElementById("image-options");
  const enhancePromptInput = document.getElementById("enhance-prompt");
  const enhanceModelLabel = document.getElementById("enhance-model-label");
  const attachButton = document.getElementById("attach-button");
  const fileInput = document.getElementById("file-input");
  const attachmentsBar = document.getElementById("attachments-bar");
  const appRoot = document.querySelector(".app");
  const composerControls = document.getElementById("composer-controls");
  const thinkingModeSelect = document.getElementById("thinking-mode");
  const effortLevelSelect = document.getElementById("effort-level");
  const webSearchToggle = document.getElementById("web-search-toggle");
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebarBackdrop = document.getElementById("sidebar-backdrop");
  const conversationList = document.getElementById("conversation-list");
  const newChatButton = document.getElementById("new-chat");
  const conversationSearch = document.getElementById("conversation-search");
  const contextMeter = document.getElementById("context-meter");
  const contextMeterFill = document.getElementById("context-meter-fill");
  const contextMeterText = document.getElementById("context-meter-text");

  /** Conversation history sent to the backend on every request. */
  let messages = [];
  let systemPrompt = "";
  /** Saved system prompt presets, newest first; mirrors the server table. */
  let presets = [];
  /** Id of the selected preset, or "" when the prompt is an unsaved edit. */
  let activePresetId = "";
  /** localStorage key remembering the last-selected preset across reloads. */
  const ACTIVE_PRESET_KEY = "localmind.activePreset";
  let isStreaming = false;
  /** Abort handle for the in-flight chat stream; null when not streaming. */
  let chatAbortController = null;
  let statusRefreshSeconds = 10;
  let statusRefreshTimer = null;
  let imageMode = false;
  let imageGenSupported = false;
  let imageGenDetail = "";
  /** Uploaded documents waiting to be sent with the next message. */
  let attachments = [];
  let attachmentCounter = 0;
  /** Server-side conversation id; created lazily on the first exchange. */
  let currentConversationId = null;
  /** Title of the open conversation, used to name exported files. */
  let currentTitle = "";
  /** RAG document ids attached this session, retrieved on each chat turn. */
  let docIds = new Set();
  /** Model keys / instance ids with an in-flight load or unload request. */
  const busyModels = new Set();
  /** Loaded-instance id -> context window in tokens, for the context meter. */
  let contextWindows = {};
  /** RAG retrieval parameters from the server, for token estimation. */
  let ragEstimate = { top_k: 0, chunk_chars: 0 };
  /** Whether to augment the next message with web search results. */
  let webSearchEnabled = false;
  /** localStorage key remembering the web search toggle across reloads. */
  const WEB_SEARCH_KEY = "localmind.webSearch";

  // ---------- Error banner ----------

  function showError(text) {
    errorText.textContent = text;
    errorBanner.classList.remove("hidden");
  }

  function hideError() {
    errorBanner.classList.add("hidden");
  }

  errorDismiss.addEventListener("click", hideError);
  errorRetry.addEventListener("click", () => {
    hideError();
    loadModels();
    if (!modelsPanel.classList.contains("hidden")) refreshManagedModels();
  });

  // ---------- Toasts ----------

  function toast(text, type = "success", timeoutMs = 5000) {
    const node = document.createElement("div");
    node.className = `toast ${type}`;
    node.textContent = text;
    toastContainer.appendChild(node);
    setTimeout(() => node.remove(), timeoutMs);
  }

  // ---------- Settings ----------

  settingsToggle.addEventListener("click", () => {
    settingsPanel.classList.toggle("hidden");
  });

  temperatureInput.addEventListener("input", () => {
    temperatureValue.textContent = Number(temperatureInput.value).toFixed(2);
  });

  async function loadDefaults() {
    try {
      const response = await fetch("/api/config");
      if (!response.ok) return;
      const data = await response.json();
      const { defaults, model_management: management } = data;
      temperatureInput.value = defaults.temperature;
      temperatureValue.textContent = Number(defaults.temperature).toFixed(2);
      maxTokensInput.value = defaults.max_tokens;
      // Provisional until loadPresets() selects the active preset; kept as the
      // fallback if the presets request fails.
      systemPrompt = defaults.system_prompt || "";
      if (management) {
        ttlEnabledInput.checked = Boolean(management.auto_unload_by_default);
        ttlSecondsInput.value = management.default_ttl_seconds ?? 600;
        if (management.default_context_length) {
          contextLengthInput.value = management.default_context_length;
        }
        statusRefreshSeconds = management.status_refresh_seconds || 10;
      }
      if (data.rag?.top_k) {
        ragEstimate = { top_k: data.rag.top_k, chunk_chars: data.rag.chunk_chars || 0 };
      }
      if (data.web_search && !data.web_search.enabled) {
        webSearchToggle.classList.add("hidden");
        if (webSearchEnabled) setWebSearch(false);
      }
      const defaultSize = data.image_generation?.default_size;
      if (defaultSize) {
        if (![...imageSizeSelect.options].some((o) => o.value === defaultSize)) {
          const option = document.createElement("option");
          option.value = defaultSize;
          option.textContent = defaultSize.replace("x", " × ");
          imageSizeSelect.appendChild(option);
        }
        imageSizeSelect.value = defaultSize;
      }
      updateContextMeter();
    } catch {
      /* Non-fatal: the UI falls back to its hardcoded defaults. */
    }
  }

  // ---------- System prompt presets ----------

  /** True when the textarea differs from the selected preset's saved content. */
  function presetIsDirty() {
    const active = presets.find((p) => p.id === activePresetId);
    if (!active) return systemPromptInput.value.trim().length > 0;
    return systemPromptInput.value !== active.content;
  }

  /** Sync the Save button, status text, and Rename/Delete enablement to state. */
  function refreshPresetControls() {
    const active = presets.find((p) => p.id === activePresetId);
    const dirty = presetIsDirty();
    presetSaveButton.disabled = !dirty || !systemPromptInput.value.trim();
    presetRenameButton.disabled = !active;
    presetDeleteButton.disabled = !active;
    if (dirty) {
      presetStatus.textContent = active ? "Unsaved edits" : "Unsaved prompt";
      presetStatus.classList.add("dirty");
    } else {
      presetStatus.textContent = active ? `Using “${active.name}”` : "";
      presetStatus.classList.remove("dirty");
    }
  }

  /** Repopulate the dropdown from `presets`, keeping the active selection. */
  function renderPresetOptions() {
    presetSelect.innerHTML = "";
    for (const preset of presets) {
      const option = document.createElement("option");
      option.value = preset.id;
      option.textContent = preset.name;
      presetSelect.appendChild(option);
    }
    const custom = document.createElement("option");
    custom.value = "";
    custom.textContent = "Custom (unsaved)";
    presetSelect.appendChild(custom);
    presetSelect.value = activePresetId;
  }

  /** Make `presetId` active, loading its content into the textarea + prompt. */
  function applyPreset(presetId, { persist = true } = {}) {
    const preset = presets.find((p) => p.id === presetId);
    activePresetId = preset ? preset.id : "";
    if (preset) {
      systemPromptInput.value = preset.content;
      systemPrompt = preset.content;
    }
    presetSelect.value = activePresetId;
    if (persist) {
      try {
        if (activePresetId) localStorage.setItem(ACTIVE_PRESET_KEY, activePresetId);
        else localStorage.removeItem(ACTIVE_PRESET_KEY);
      } catch { /* localStorage may be unavailable (private mode); ignore. */ }
    }
    refreshPresetControls();
    updateContextMeter();
  }

  async function loadPresets() {
    try {
      const response = await fetch("/api/system-prompts");
      if (!response.ok) return;
      const data = await response.json();
      presets = Array.isArray(data.presets) ? data.presets : [];
      renderPresetOptions();
      let stored = "";
      try { stored = localStorage.getItem(ACTIVE_PRESET_KEY) || ""; } catch { /* ignore */ }
      const initial = presets.find((p) => p.id === stored) || presets[0];
      if (initial) {
        applyPreset(initial.id, { persist: false });
      } else {
        // No presets at all: keep the config fallback as a custom prompt.
        systemPromptInput.value = systemPrompt;
        refreshPresetControls();
      }
    } catch {
      /* Non-fatal: leave the config-default prompt in place. */
      systemPromptInput.value = systemPrompt;
      refreshPresetControls();
    }
  }

  // Typing in the textarea takes effect immediately for this session; it
  // becomes a "Custom (unsaved)" prompt until saved back to a preset.
  systemPromptInput.addEventListener("input", () => {
    systemPrompt = systemPromptInput.value;
    refreshPresetControls();
    updateContextMeter();
  });

  presetSelect.addEventListener("change", () => {
    if (presetSelect.value) {
      applyPreset(presetSelect.value);
    } else {
      // Re-selecting "Custom" just detaches from the active preset.
      activePresetId = "";
      try { localStorage.removeItem(ACTIVE_PRESET_KEY); } catch { /* ignore */ }
      refreshPresetControls();
    }
  });

  presetSaveButton.addEventListener("click", async () => {
    const active = presets.find((p) => p.id === activePresetId);
    if (!active) return;
    const content = systemPromptInput.value;
    try {
      const response = await fetch(`/api/system-prompts/${active.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not save the preset.");
      active.content = data.content;
      systemPrompt = data.content;
      refreshPresetControls();
      toast(`Saved “${active.name}”.`);
    } catch (error) {
      showError(error.message || "Could not save the preset.");
    }
  });

  presetNewButton.addEventListener("click", async () => {
    const content = systemPromptInput.value.trim();
    if (!content) {
      showError("Enter a system prompt before saving it as a preset.");
      return;
    }
    const name = window.prompt("Name for this preset:");
    if (name === null) return;
    const trimmed = name.trim();
    if (!trimmed) return;
    try {
      const response = await fetch("/api/system-prompts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: trimmed, content }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not create the preset.");
      presets.unshift(data);
      renderPresetOptions();
      applyPreset(data.id);
      toast(`Created “${data.name}”.`);
    } catch (error) {
      showError(error.message || "Could not create the preset.");
    }
  });

  presetRenameButton.addEventListener("click", async () => {
    const active = presets.find((p) => p.id === activePresetId);
    if (!active) return;
    const name = window.prompt("Rename preset:", active.name);
    if (name === null) return;
    const trimmed = name.trim();
    if (!trimmed || trimmed === active.name) return;
    try {
      const response = await fetch(`/api/system-prompts/${active.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: trimmed }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not rename the preset.");
      active.name = data.name;
      renderPresetOptions();
      refreshPresetControls();
      toast(`Renamed to “${data.name}”.`);
    } catch (error) {
      showError(error.message || "Could not rename the preset.");
    }
  });

  presetDeleteButton.addEventListener("click", async () => {
    const active = presets.find((p) => p.id === activePresetId);
    if (!active) return;
    if (!window.confirm(`Delete the preset “${active.name}”?`)) return;
    try {
      const response = await fetch(`/api/system-prompts/${active.id}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || "Could not delete the preset.");
      }
      presets = presets.filter((p) => p.id !== active.id);
      renderPresetOptions();
      if (presets.length) {
        applyPreset(presets[0].id);
      } else {
        activePresetId = "";
        presetSelect.value = "";
        refreshPresetControls();
      }
      toast(`Deleted “${active.name}”.`);
    } catch (error) {
      showError(error.message || "Could not delete the preset.");
    }
  });

  // ---------- Models ----------

  async function loadModels({ quiet = false } = {}) {
    const previousSelection = modelSelect.value;
    modelSelect.disabled = true;
    if (!quiet) modelSelect.innerHTML = "<option value=''>Loading models…</option>";
    try {
      const response = await fetch("/api/models");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to load models.");
      if (!data.models.length) {
        modelSelect.innerHTML = "<option value=''>No model loaded</option>";
        if (!quiet) {
          showError("LM Studio is reachable but no models are loaded. " +
                    "Use the 📦 panel to load one.");
        }
        updateSendState();
        return;
      }
      modelSelect.innerHTML = "";
      for (const id of data.models) {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = id;
        modelSelect.appendChild(option);
      }
      if (data.models.includes(previousSelection)) {
        modelSelect.value = previousSelection;
      }
      modelSelect.disabled = false;
      updateSendState();
      refreshContextWindows();
    } catch (error) {
      modelSelect.innerHTML = "<option value=''>No models available</option>";
      if (!quiet) showError(error.message || "Could not reach the backend.");
      updateSendState();
    }
  }

  // ---------- Model management ----------

  function formatBytes(bytes) {
    if (!bytes && bytes !== 0) return "";
    const gb = bytes / 1024 ** 3;
    return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  }

  function formatContext(tokens) {
    if (!tokens) return "";
    return tokens >= 1000 ? `${Math.round(tokens / 1000)}k` : String(tokens);
  }

  /** Seconds -> "m:ss" (or "h:mm:ss"), for the live TTL countdown. */
  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) return "";
    const total = Math.max(0, Math.round(seconds));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const pad = (n) => String(n).padStart(2, "0");
    return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
  }

  function loadOptions() {
    // TTL is applied by the backend via the `lms` CLI (the REST load endpoint
    // can't set one). Sent only when auto-unload is checked.
    const options = {};
    if (ttlEnabledInput.checked) {
      const ttl = Math.floor(Number(ttlSecondsInput.value));
      if (ttl >= 1) options.ttl_seconds = ttl;
    }
    const contextLength = Math.floor(Number(contextLengthInput.value));
    if (contextLengthInput.value.trim() && contextLength >= 1) {
      options.context_length = contextLength;
    }
    return options;
  }

  function renderModels(models) {
    modelsList.innerHTML = "";
    if (!models.length) {
      modelsList.innerHTML = "<li class='models-placeholder'>No models downloaded in LM Studio.</li>";
      modelsSummary.textContent = "";
      return;
    }

    let loadedCount = 0;
    let loadedBytes = 0;

    // Loaded models first (they're what you're managing), then largest first.
    const ordered = [...models].sort((a, b) => {
      const aLoaded = a.loaded_instances.length > 0;
      const bLoaded = b.loaded_instances.length > 0;
      if (aLoaded !== bLoaded) return aLoaded ? -1 : 1;
      return (b.size_bytes || 0) - (a.size_bytes || 0);
    });

    for (const model of ordered) {
      const isLoaded = model.loaded_instances.length > 0;
      const isBusy = busyModels.has(model.key);
      if (isLoaded) {
        loadedCount += 1;
        loadedBytes += (model.size_bytes || 0) * model.loaded_instances.length;
      }

      const row = document.createElement("li");
      row.className = "model-row";

      const info = document.createElement("div");
      info.className = "model-info";

      const name = document.createElement("div");
      name.className = "model-name";
      name.textContent = model.key;
      const badge = document.createElement("span");
      if (model.type === "embedding") {
        badge.className = "badge embedding";
        badge.textContent = "embedding";
      } else {
        badge.className = `badge ${isLoaded ? "loaded" : "not-loaded"}`;
        badge.textContent = isLoaded ? "loaded" : "not loaded";
      }
      name.appendChild(badge);

      // Capability badges from LM Studio's reported model capabilities.
      const caps = model.capabilities || {};
      const capBadges = [];
      if (caps.vision) capBadges.push("vision");
      if (caps.trained_for_tool_use) capBadges.push("tools");
      if (caps.reasoning) capBadges.push("reasoning");
      for (const cap of capBadges) {
        const capBadge = document.createElement("span");
        capBadge.className = "badge capability";
        capBadge.textContent = cap;
        name.appendChild(capBadge);
      }

      const meta = document.createElement("div");
      meta.className = "model-meta";
      meta.textContent = [
        model.params,
        model.quantization,
        formatBytes(model.size_bytes),
        model.max_context_length ? `max ctx ${formatContext(model.max_context_length)}` : "",
      ].filter(Boolean).join(" · ");

      info.append(name, meta);

      const actions = document.createElement("div");
      const loadButton = document.createElement("button");
      loadButton.type = "button";
      loadButton.className = "small-button primary";
      if (isBusy) {
        loadButton.disabled = true;
        loadButton.innerHTML = "<span class='spinner'></span>Working…";
      } else {
        loadButton.textContent = isLoaded ? "Load another" : "Load";
        loadButton.addEventListener("click", () => loadModel(model.key));
      }
      actions.appendChild(loadButton);

      row.append(info, actions);

      if (isLoaded) {
        const instances = document.createElement("div");
        instances.className = "model-instances";
        for (const instance of model.loaded_instances) {
          const line = document.createElement("div");
          line.className = "model-instance";
          const label = document.createElement("span");
          // Token usage is only meaningful for the model the chat is pointed at,
          // since that's the prompt we can measure.
          const isActiveChatModel = instance.id === modelSelect.value;
          const ctxText = instance.context_length
            ? `ctx ${formatContext(instance.context_length)}` +
              (isActiveChatModel ? ` (≈${formatTokens(estimateUsedTokens())} used)` : "")
            : "";
          const details = [
            ctxText,
            instance.remaining_ttl_seconds != null
              ? `ttl ${formatDuration(instance.remaining_ttl_seconds)}`
              : "no ttl",
          ].filter(Boolean);
          label.textContent = details.length
            ? `${instance.id} — ${details.join(" · ")}`
            : instance.id;
          const unloadButton = document.createElement("button");
          unloadButton.type = "button";
          unloadButton.className = "small-button danger";
          if (busyModels.has(instance.id)) {
            unloadButton.disabled = true;
            unloadButton.innerHTML = "<span class='spinner'></span>Unloading…";
          } else {
            unloadButton.textContent = "Unload";
            unloadButton.addEventListener("click", () => unloadModelInstance(instance.id));
          }
          line.append(label, unloadButton);
          instances.appendChild(line);
        }
        row.appendChild(instances);
      }

      modelsList.appendChild(row);
    }

    // LM Studio doesn't report a loaded instance's resident memory, only the
    // model's on-disk size — so label this as a disk-footprint estimate rather
    // than implying it's the exact RAM/VRAM in use.
    modelsSummary.textContent =
      `${loadedCount} of ${models.length} models loaded` +
      (loadedBytes ? ` · ~${formatBytes(loadedBytes)} on disk` : "");
  }

  async function refreshManagedModels({ quiet = false } = {}) {
    try {
      const response = await fetch("/api/models/manage");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to fetch model status.");
      renderModels(data.models);
    } catch (error) {
      if (!quiet) {
        modelsList.innerHTML = "<li class='models-placeholder'>Model status unavailable.</li>";
        modelsSummary.textContent = "";
        showError(error.message || "Could not fetch model status.");
      }
    }
  }

  async function loadModel(modelKey) {
    busyModels.add(modelKey);
    refreshManagedModels({ quiet: true });
    try {
      const response = await fetch("/api/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelKey, ...loadOptions() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to load the model.");
      const ttlNote = data.ttl_seconds ? `, auto-unloads after ${data.ttl_seconds}s idle` : "";
      const seconds = data.load_time_seconds ? ` in ${data.load_time_seconds.toFixed(1)}s` : "";
      toast(`Loaded ${data.instance_id || modelKey}${seconds}${ttlNote}.`);
    } catch (error) {
      toast(error.message || "Failed to load the model.", "error", 8000);
    } finally {
      busyModels.delete(modelKey);
      await Promise.all([refreshManagedModels({ quiet: true }), loadModels({ quiet: true })]);
    }
  }

  async function unloadModelInstance(instanceId) {
    busyModels.add(instanceId);
    refreshManagedModels({ quiet: true });
    try {
      const response = await fetch("/api/models/unload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance_id: instanceId }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to unload the model.");
      toast(`Unloaded ${instanceId}.`);
    } catch (error) {
      toast(error.message || "Failed to unload the model.", "error", 8000);
    } finally {
      busyModels.delete(instanceId);
      await Promise.all([refreshManagedModels({ quiet: true }), loadModels({ quiet: true })]);
    }
  }

  async function unloadAll() {
    unloadAllButton.disabled = true;
    try {
      const response = await fetch("/api/models/unload-all", { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to unload models.");
      if (data.errors.length) {
        toast(`Unloaded ${data.unloaded.length}, failed: ${data.errors.join("; ")}`, "error", 8000);
      } else if (data.unloaded.length) {
        toast(`Unloaded ${data.unloaded.length} model instance(s).`);
      } else {
        toast("Nothing to unload — no models are loaded.");
      }
    } catch (error) {
      toast(error.message || "Failed to unload models.", "error", 8000);
    } finally {
      unloadAllButton.disabled = false;
      await Promise.all([refreshManagedModels({ quiet: true }), loadModels({ quiet: true })]);
    }
  }

  function setModelsPanelOpen(open) {
    modelsPanel.classList.toggle("hidden", !open);
    clearInterval(statusRefreshTimer);
    statusRefreshTimer = null;
    if (open) {
      refreshManagedModels();
      statusRefreshTimer = setInterval(
        () => refreshManagedModels({ quiet: true }),
        statusRefreshSeconds * 1000,
      );
    }
  }

  modelsToggle.addEventListener("click", () => {
    setModelsPanelOpen(modelsPanel.classList.contains("hidden"));
  });

  modelsRefresh.addEventListener("click", () => {
    refreshManagedModels();
    loadModels({ quiet: true });
  });

  unloadAllButton.addEventListener("click", unloadAll);

  // ---------- Context-window meter ----------

  // Rough heuristic: ~4 characters per token, plus a flat cost per image.
  const CHARS_PER_TOKEN = 4;
  const IMAGE_TOKEN_ESTIMATE = 800;

  function estimateTokens(text) {
    return Math.ceil((text || "").length / CHARS_PER_TOKEN);
  }

  function estimateMessageTokens(message) {
    let tokens = 4; // per-message formatting overhead
    const content = message.content;
    if (typeof content === "string") {
      tokens += estimateTokens(content);
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (part?.type === "text") tokens += estimateTokens(part.text);
        else if (part?.type === "image_url") tokens += IMAGE_TOKEN_ESTIMATE;
      }
    }
    return tokens;
  }

  function formatTokens(tokens) {
    return tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : String(tokens);
  }

  /** Map loaded instances to their context windows (instance override or model max). */
  async function refreshContextWindows() {
    try {
      const response = await fetch("/api/models/manage");
      if (!response.ok) return;
      const { models } = await response.json();
      contextWindows = {};
      for (const model of models || []) {
        for (const instance of model.loaded_instances || []) {
          if (instance.id) {
            contextWindows[instance.id] =
              instance.context_length || model.max_context_length || 0;
          }
        }
      }
    } catch {
      /* Native API unavailable: the meter simply stays hidden. */
    }
    updateContextMeter();
  }

  /** Estimated tokens in the current prompt: system, history, draft, attachments, RAG. */
  function estimateUsedTokens() {
    // The system prompt and the in-progress draft each become full messages in
    // the API call, so they need the same per-message overhead (+4 tokens) that
    // estimateMessageTokens applies to the history entries.
    let used = systemPrompt
      ? estimateMessageTokens({ role: "system", content: systemPrompt })
      : 0;
    for (const message of messages) used += estimateMessageTokens(message);
    if (messageInput.value)
      used += estimateMessageTokens({ role: "user", content: messageInput.value });
    for (const doc of attachments) {
      if (doc.uploading) continue;
      if (doc.kind === "image") used += IMAGE_TOKEN_ESTIMATE;
      else if (!doc.rag) used += estimateTokens(doc.text);
    }
    if (docIds.size && ragEstimate.top_k) {
      used += Math.ceil((ragEstimate.top_k * ragEstimate.chunk_chars) / CHARS_PER_TOKEN);
    }
    return used;
  }

  function updateContextMeter() {
    const limit = contextWindows[modelSelect.value];
    if (!limit) {
      contextMeter.classList.add("hidden");
      return;
    }
    const used = estimateUsedTokens();
    const percent = Math.min(100, Math.round((used / limit) * 100));
    contextMeterFill.style.width = `${percent}%`;
    contextMeterFill.classList.toggle("warn", percent >= 80 && percent < 95);
    contextMeterFill.classList.toggle("danger", percent >= 95);
    contextMeterText.textContent = `≈${formatTokens(used)} / ${formatTokens(limit)}`;
    contextMeter.title =
      `Estimated prompt size: ~${used.toLocaleString()} of ` +
      `${limit.toLocaleString()} tokens (${percent}%).\n` +
      "Covers the system prompt, history, attachments, and your draft — " +
      "a character-based estimate, not an exact token count.";
    contextMeter.classList.remove("hidden");
  }

  // ---------- Markdown rendering ----------

  // Vendored libraries; if either failed to load, fall back to plain text.
  const markdownReady =
    typeof window.marked !== "undefined" && typeof window.DOMPurify !== "undefined";
  if (markdownReady) {
    marked.use({ gfm: true, breaks: true });
  }

  function copyText(text) {
    if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(text);
    // Plain-http LAN pages have no clipboard API: fall back to execCommand.
    const scratch = document.createElement("textarea");
    scratch.value = text;
    scratch.style.position = "fixed";
    scratch.style.opacity = "0";
    document.body.appendChild(scratch);
    scratch.select();
    try {
      return document.execCommand("copy")
        ? Promise.resolve()
        : Promise.reject(new Error("copy rejected"));
    } finally {
      scratch.remove();
    }
  }

  function addCodeCopyButtons(target) {
    for (const pre of target.querySelectorAll("pre")) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "code-copy";
      button.textContent = "⧉ Copy";
      button.addEventListener("click", async () => {
        const code = pre.querySelector("code");
        try {
          await copyText((code ?? pre).innerText.replace(/\n$/, ""));
          button.textContent = "✓ Copied";
        } catch {
          button.textContent = "Copy failed";
        }
        setTimeout(() => { button.textContent = "⧉ Copy"; }, 1500);
      });
      pre.appendChild(button);
    }
  }

  /**
   * Render assistant markdown into `target`, sanitized. Model output is
   * untrusted input, so everything goes through DOMPurify. Copy buttons are
   * skipped during streaming (the subtree is replaced on every chunk).
   */
  function renderMarkdown(target, text, { withCopyButtons = true } = {}) {
    if (!markdownReady) {
      target.textContent = text;
      return;
    }
    target.classList.add("md");
    target.innerHTML = DOMPurify.sanitize(marked.parse(text));
    for (const link of target.querySelectorAll("a[href]")) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
    if (withCopyButtons) addCodeCopyButtons(target);
  }

  // ---------- Chat rendering ----------

  function appendMessage(role, content) {
    emptyState?.remove();
    const bubble = document.createElement("div");
    bubble.className = `message ${role}`;
    bubble.textContent = content;
    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return bubble;
  }

  // True when the view is at (or within a small slack of) the bottom. Lets us
  // follow streaming output without yanking the user back down if they scrolled
  // up to read earlier content.
  function isPinnedToBottom() {
    const slack = 80;
    return (
      chatWindow.scrollHeight - chatWindow.scrollTop - chatWindow.clientHeight <=
      slack
    );
  }

  function updateSendState() {
    // While a chat reply streams, the button becomes an always-enabled Stop.
    if (isStreaming && chatAbortController) {
      sendButton.disabled = false;
      sendButton.textContent = "⏹ Stop";
      sendButton.classList.add("stop");
      updateContextMeter();
      return;
    }
    sendButton.classList.remove("stop");
    const uploading = attachments.some((doc) => doc.uploading);
    sendButton.disabled =
      isStreaming || uploading || !messageInput.value.trim() ||
      (!imageMode && !modelSelect.value);
    sendButton.textContent = imageMode ? "Generate" : "Send";
    updateEnhanceLabel();
    updateContextMeter();
  }

  // ---------- Document attachments ----------

  function renderAttachments() {
    attachmentsBar.innerHTML = "";
    attachmentsBar.classList.toggle("hidden", !attachments.length);
    for (const doc of attachments) {
      const chip = document.createElement("span");
      chip.className = `attachment-chip${doc.uploading ? " uploading" : ""}`;

      const name = document.createElement("span");
      name.className = "chip-name";
      if (doc.kind === "image" && doc.dataUrl) {
        const thumb = document.createElement("img");
        thumb.className = "chip-thumb";
        thumb.src = doc.dataUrl;
        thumb.alt = doc.name;
        chip.prepend(thumb);
        name.textContent = doc.name;
      } else {
        name.textContent = `${doc.kind === "image" ? "🖼" : "📄"} ${doc.name}`;
      }
      chip.appendChild(name);

      const meta = document.createElement("span");
      meta.className = "chip-meta";
      if (doc.uploading) {
        meta.innerHTML = "<span class='spinner'></span>";
      } else if (doc.kind === "image") {
        meta.textContent = "image";
      } else {
        const base = doc.pages
          ? `${doc.pages} page${doc.pages === 1 ? "" : "s"}`
          : `${doc.chars} chars`;
        meta.textContent = doc.rag
          ? `${base} · 🔍 searchable`
          : base + (doc.truncated ? " · truncated" : "");
      }
      chip.appendChild(meta);

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "chip-remove";
      remove.setAttribute("aria-label", `Remove ${doc.name}`);
      remove.textContent = "✕";
      remove.addEventListener("click", () => {
        attachments = attachments.filter((entry) => entry.id !== doc.id);
        renderAttachments();
        updateSendState();
      });
      chip.appendChild(remove);

      attachmentsBar.appendChild(chip);
    }
  }

  function attachImage(file) {
    if (file.size > 15 * 1024 * 1024) {
      toast(`${file.name}: image is too large (limit 15 MB).`, "error", 6000);
      return;
    }
    const name = file.name || `screenshot-${++attachmentCounter}.png`;
    const doc = { id: ++attachmentCounter, name, kind: "image", uploading: true };
    attachments.push(doc);
    renderAttachments();
    updateSendState();
    const reader = new FileReader();
    reader.onload = () => {
      Object.assign(doc, { dataUrl: reader.result, uploading: false });
      renderAttachments();
      updateSendState();
    };
    reader.onerror = () => {
      attachments = attachments.filter((entry) => entry.id !== doc.id);
      toast(`${name}: could not read the image.`, "error", 6000);
      renderAttachments();
      updateSendState();
    };
    reader.readAsDataURL(file);
  }

  async function uploadFiles(fileList) {
    for (const file of fileList) {
      if (file.type.startsWith("image/")) {
        attachImage(file);
        continue;
      }
      const doc = { id: ++attachmentCounter, name: file.name, kind: "document", uploading: true };
      attachments.push(doc);
      renderAttachments();
      updateSendState();
      try {
        const form = new FormData();
        form.append("file", file);
        const response = await fetch("/api/upload", { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Upload failed.");
        Object.assign(doc, data, { uploading: false });
        if (data.truncated) {
          toast(`${doc.name} is long — only the first part of the text is used.`,
                "error", 7000);
        }
      } catch (error) {
        attachments = attachments.filter((entry) => entry.id !== doc.id);
        toast(`${doc.name}: ${error.message || "upload failed."}`, "error", 8000);
      }
      renderAttachments();
      updateSendState();
    }
  }

  attachButton.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    uploadFiles([...fileInput.files]);
    fileInput.value = "";
  });

  document.addEventListener("dragover", (event) => {
    event.preventDefault();
    appRoot.classList.add("drag-over");
  });
  document.addEventListener("dragleave", (event) => {
    if (!event.relatedTarget) appRoot.classList.remove("drag-over");
  });
  document.addEventListener("drop", (event) => {
    event.preventDefault();
    appRoot.classList.remove("drag-over");
    if (event.dataTransfer?.files?.length) uploadFiles([...event.dataTransfer.files]);
  });

  // Paste screenshots/images straight from the clipboard.
  document.addEventListener("paste", (event) => {
    const items = [...(event.clipboardData?.items || [])];
    const images = items
      .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter(Boolean);
    if (images.length) {
      event.preventDefault();
      uploadFiles(images);
    }
  });

  // ---------- Image generation ----------

  async function checkImageCapability() {
    try {
      const response = await fetch("/api/images/capability");
      const data = await response.json();
      imageGenSupported = Boolean(data.supported);
      imageGenDetail = data.detail || "";
    } catch {
      imageGenSupported = false;
      imageGenDetail = "Could not determine image generation support.";
    }
    imageToggle.classList.remove("hidden");
    imageToggle.classList.toggle("unsupported", !imageGenSupported);
    imageToggle.title = imageGenSupported
      ? "Image generation mode"
      : `Image generation unavailable — ${imageGenDetail}`;
  }

  function updateEnhanceLabel() {
    enhanceModelLabel.textContent = modelSelect.value || "the selected model";
  }

  // ---------- Thinking controls ----------

  /** Map the two bottom-bar selects onto LM Studio's reasoning_effort. */
  function reasoningEffortValue() {
    if (thinkingModeSelect.value === "off") return "none";
    if (thinkingModeSelect.value === "on") return effortLevelSelect.value;
    return null; // Default: leave the model's own behavior untouched.
  }

  thinkingModeSelect.addEventListener("change", () => {
    effortLevelSelect.disabled = thinkingModeSelect.value !== "on";
  });

  // ---------- Web search toggle ----------

  function setWebSearch(on) {
    webSearchEnabled = on;
    webSearchToggle.classList.toggle("active", on);
    webSearchToggle.setAttribute("aria-pressed", String(on));
    try {
      if (on) localStorage.setItem(WEB_SEARCH_KEY, "1");
      else localStorage.removeItem(WEB_SEARCH_KEY);
    } catch { /* localStorage may be unavailable (private mode); ignore. */ }
  }

  webSearchToggle.addEventListener("click", () => setWebSearch(!webSearchEnabled));

  // Restore the last toggle state; the server config may still hide it.
  try {
    if (localStorage.getItem(WEB_SEARCH_KEY) === "1") setWebSearch(true);
  } catch { /* ignore */ }

  function setImageMode(on) {
    imageMode = on;
    imageToggle.classList.toggle("active", on);
    composer.classList.toggle("image-mode", on);
    imageOptions.classList.toggle("hidden", !on);
    // Thinking applies to text chat only; hide it while generating images.
    composerControls.classList.toggle("hidden", on);
    updateEnhanceLabel();
    messageInput.placeholder = on
      ? "Describe the image to generate… (Enter to generate)"
      : "Type a message… (Enter to send, Shift+Enter for a new line)";
    updateSendState();
  }

  imageToggle.addEventListener("click", () => {
    if (!imageGenSupported) {
      toast(imageGenDetail || "Image generation is not supported by the connected server.",
            "error", 9000);
      return;
    }
    setImageMode(!imageMode);
  });

  async function sendImageMessage(prompt) {
    const enhanceWith =
      enhancePromptInput.checked && modelSelect.value ? modelSelect.value : null;
    appendMessage("user", prompt);
    const bubble = appendMessage(
      "assistant",
      enhanceWith ? `Enhancing prompt with ${enhanceWith}, then generating…` : "Generating image…",
    );
    bubble.classList.add("pending");
    try {
      const response = await fetch("/api/images", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          size: imageSizeSelect.value,
          n: 1,
          ...(enhanceWith ? { enhance_with: enhanceWith } : {}),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Image generation failed.");
      if (data.enhancement_error) {
        toast(`Prompt enhancement failed (${data.enhancement_error}) — used your original prompt.`,
              "error", 8000);
      }
      bubble.textContent = "";
      bubble.classList.add("image");
      data.images.forEach((src, index) => {
        const img = document.createElement("img");
        img.src = src;
        img.alt = prompt;
        img.addEventListener("click", () => img.classList.toggle("expanded"));
        const download = document.createElement("a");
        download.href = src;
        download.download = `localmind-${Date.now()}-${index + 1}.png`;
        download.textContent = "⬇ Download";
        download.className = "image-download";
        bubble.append(img, download);
      });
      if (data.enhanced && data.prompt_used) {
        const caption = document.createElement("div");
        caption.className = "image-caption";
        caption.textContent = `✨ ${data.prompt_used}`;
        bubble.appendChild(caption);
      }
      chatWindow.scrollTop = chatWindow.scrollHeight;
    } catch (error) {
      bubble.remove();
      showError(error.message || "Image generation failed.");
    } finally {
      bubble.classList.remove("pending");
      isStreaming = false;
      updateSendState();
      messageInput.focus();
    }
  }

  messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 180)}px`;
    updateSendState();
  });

  messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!sendButton.disabled) sendMessage();
    }
  });

  sendButton.addEventListener("click", () => {
    if (isStreaming && chatAbortController) {
      chatAbortController.abort();
      return;
    }
    sendMessage();
  });
  modelSelect.addEventListener("change", updateSendState);

  // ---------- Conversations (server-side persistence) ----------

  function clearChatWindow() {
    chatWindow.querySelectorAll(".message").forEach((node) => node.remove());
  }

  function startNewChat() {
    if (isStreaming) return;
    messages = [];
    attachments = [];
    docIds = new Set();
    currentConversationId = null;
    currentTitle = "";
    renderAttachments();
    clearChatWindow();
    highlightActiveConversation();
    updateContextMeter();
    updateExportState();
  }

  clearChatButton.addEventListener("click", startNewChat);
  newChatButton.addEventListener("click", () => {
    startNewChat();
    closeSidebarOnMobile();
  });

  function highlightActiveConversation() {
    conversationList.querySelectorAll(".conversation-item").forEach((item) => {
      item.classList.toggle("active", item.dataset.id === currentConversationId);
    });
  }

  function closeSidebarOnMobile() {
    if (window.matchMedia("(max-width: 900px)").matches) {
      sidebar.classList.remove("open");
      sidebarBackdrop.classList.add("hidden");
    }
  }

  sidebarToggle.addEventListener("click", () => {
    if (window.matchMedia("(max-width: 900px)").matches) {
      const opening = !sidebar.classList.contains("open");
      sidebar.classList.toggle("open", opening);
      sidebarBackdrop.classList.toggle("hidden", !opening);
    } else {
      sidebar.classList.toggle("collapsed");
    }
  });

  sidebarBackdrop.addEventListener("click", closeSidebarOnMobile);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && sidebar.classList.contains("open")) closeSidebarOnMobile();
  });

  async function loadConversations() {
    // While a search is active, refreshes re-run the search instead of
    // replacing the filtered list with all conversations.
    if (conversationSearch.value.trim()) {
      runConversationSearch();
      return;
    }
    try {
      const response = await fetch("/api/conversations");
      if (!response.ok) return;
      const { conversations } = await response.json();
      renderConversationList(conversations);
    } catch {
      /* Non-fatal: the sidebar just stays empty. */
    }
  }

  // ---------- Conversation search ----------

  let searchDebounceTimer = null;

  async function runConversationSearch() {
    const query = conversationSearch.value.trim();
    if (!query) {
      loadConversations();
      return;
    }
    try {
      const response = await fetch(
        `/api/conversations/search?q=${encodeURIComponent(query)}`);
      if (!response.ok) return;
      const { results } = await response.json();
      // The user may have kept typing while this request was in flight.
      if (conversationSearch.value.trim() !== query) return;
      renderSearchResults(results);
    } catch {
      /* Non-fatal: the previous list stays visible. */
    }
  }

  function renderSearchResults(results) {
    conversationList.innerHTML = "";
    if (!results.length) {
      const empty = document.createElement("li");
      empty.className = "conversation-empty";
      empty.textContent = "No matching conversations.";
      conversationList.appendChild(empty);
      return;
    }
    for (const result of results) {
      const item = document.createElement("li");
      item.className = "conversation-item search-result";
      item.dataset.id = result.id;

      const title = document.createElement("span");
      title.className = "conversation-title";
      title.textContent = result.title;
      title.title = result.title;
      item.appendChild(title);

      if (result.snippet) {
        const snippet = document.createElement("span");
        snippet.className = "conversation-snippet";
        snippet.textContent = result.snippet;
        item.appendChild(snippet);
      }

      item.addEventListener("click", () => openConversation(result.id));
      conversationList.appendChild(item);
    }
    highlightActiveConversation();
  }

  conversationSearch.addEventListener("input", () => {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(runConversationSearch, 250);
  });

  conversationSearch.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && conversationSearch.value) {
      event.stopPropagation();
      conversationSearch.value = "";
      runConversationSearch();
    }
  });

  function renderConversationList(conversations) {
    conversationList.innerHTML = "";
    for (const conversation of conversations) {
      const item = document.createElement("li");
      item.className = "conversation-item";
      item.dataset.id = conversation.id;

      const title = document.createElement("span");
      title.className = "conversation-title";
      title.textContent = conversation.title;
      title.title = conversation.title;
      item.appendChild(title);

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "conversation-delete";
      remove.setAttribute("aria-label", `Delete ${conversation.title}`);
      remove.textContent = "🗑";
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteConversation(conversation.id);
      });
      item.appendChild(remove);

      item.addEventListener("click", () => {
        if (item.classList.contains("renaming")) return;
        openConversation(conversation.id);
      });
      item.addEventListener("dblclick", (event) => {
        event.preventDefault();
        beginRename(item, conversation, title);
      });
      conversationList.appendChild(item);
    }
    highlightActiveConversation();
  }

  function beginRename(item, conversation, titleNode) {
    if (item.classList.contains("renaming")) return;
    item.classList.add("renaming");
    const input = document.createElement("input");
    input.value = conversation.title;
    input.maxLength = 200;
    // Keep clicks/dblclicks inside the field from reaching the row handlers.
    input.addEventListener("click", (event) => event.stopPropagation());
    input.addEventListener("dblclick", (event) => event.stopPropagation());
    titleNode.textContent = "";
    titleNode.appendChild(input);
    input.focus();
    input.select();
    let finished = false;
    const finish = async (save) => {
      if (finished) return;
      finished = true;
      const newTitle = input.value.trim();
      if (save && newTitle && newTitle !== conversation.title) {
        try {
          await fetch(`/api/conversations/${conversation.id}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: newTitle }),
          });
        } catch {
          toast("Could not rename the conversation.", "error", 6000);
        }
      }
      loadConversations();
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") finish(true);
      if (event.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(true));
  }

  async function deleteConversation(id) {
    if (isStreaming && id === currentConversationId) {
      toast("Can't delete a conversation while it's still generating.", "error", 6000);
      return;
    }
    if (!window.confirm("Delete this conversation? This cannot be undone.")) return;
    try {
      const response = await fetch(`/api/conversations/${id}`, { method: "DELETE" });
      // A 404 means it's already gone — treat as success.
      if (!response.ok && response.status !== 404) throw new Error();
      if (id === currentConversationId) {
        currentConversationId = null;
        startNewChat();
      }
    } catch {
      toast("Could not delete the conversation.", "error", 6000);
    }
    loadConversations();
  }

  function extractMessageText(content) {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      const part = content.find((entry) => entry?.type === "text");
      return part?.text || "";
    }
    return "";
  }

  function renderConversationHistory() {
    clearChatWindow();
    messages.forEach((message, index) => {
      if (message.role === "user") {
        const bubble = appendMessage("user", message.display ?? extractMessageText(message.content));
        const meta = Array.isArray(message.attachments_meta) ? message.attachments_meta : [];
        if (meta.length) {
          const tags = document.createElement("div");
          tags.className = "message-attachments";
          for (const entry of meta) {
            if (entry.kind === "image" && typeof entry.dataUrl === "string"
                && entry.dataUrl.startsWith("data:image/")) {
              const thumb = document.createElement("img");
              thumb.className = "msg-thumb";
              thumb.src = entry.dataUrl;
              thumb.alt = entry.name || "image";
              tags.appendChild(thumb);
            } else {
              const tag = document.createElement("span");
              tag.className = "doc-tag";
              tag.textContent = `📄 ${entry.name}${entry.pages ? ` (${entry.pages} p.)` : ""}`;
              tags.appendChild(tag);
            }
          }
          bubble.prepend(tags);
        }
        attachMessageActions(bubble, "user", index);
      } else if (message.role === "assistant") {
        const bubble = appendMessage("assistant", "");
        renderMarkdown(bubble, extractMessageText(message.content));
        attachMessageActions(bubble, "assistant", index);
      }
    });
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  // ---------- Edit & regenerate ----------

  function attachMessageActions(bubble, role, index) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    const action = document.createElement("button");
    action.type = "button";
    action.className = "message-action";
    if (role === "user") {
      action.textContent = "✏️ Edit";
      action.title = "Edit this message and regenerate from here";
      action.addEventListener("click", () => beginEditMessage(bubble, index));
    } else {
      action.textContent = "↻ Regenerate";
      action.title = "Generate this response again";
      action.addEventListener("click", () => regenerateFrom(index));
    }
    actions.appendChild(action);
    bubble.appendChild(actions);
  }

  /**
   * Replace the user-visible text of a message, preserving any inlined
   * document blocks (string content) and image parts (multimodal content)
   * around it.
   */
  function replaceMessageText(message, newText) {
    const oldText = message.display ?? extractMessageText(message.content);
    const swap = (text) =>
      oldText && text.endsWith(oldText)
        ? text.slice(0, text.length - oldText.length) + newText
        : newText;
    if (typeof message.content === "string") {
      message.content = swap(message.content);
    } else if (Array.isArray(message.content)) {
      const part = message.content.find((entry) => entry?.type === "text");
      if (part) part.text = swap(part.text || "");
      else message.content.push({ type: "text", text: newText });
    }
    message.display = newText;
  }

  function beginEditMessage(bubble, index) {
    if (isStreaming || bubble.classList.contains("editing") || !messages[index]) return;
    const message = messages[index];
    bubble.classList.add("editing");

    const editor = document.createElement("textarea");
    editor.className = "edit-area";
    editor.value = message.display ?? extractMessageText(message.content);

    const save = document.createElement("button");
    save.type = "button";
    save.className = "small-button primary";
    save.textContent = "Save & regenerate";
    save.addEventListener("click", () => {
      const newText = editor.value.trim();
      if (newText) submitEdit(index, newText);
    });

    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "small-button";
    cancel.textContent = "Cancel";
    // Re-rendering restores the bubble (including attachment thumbnails).
    cancel.addEventListener("click", () => renderConversationHistory());

    const controls = document.createElement("div");
    controls.className = "edit-controls";
    controls.append(save, cancel);

    bubble.textContent = "";
    bubble.append(editor, controls);
    const autosize = () => {
      editor.style.height = "auto";
      editor.style.height = `${Math.min(editor.scrollHeight, 240)}px`;
    };
    editor.addEventListener("input", autosize);
    autosize();
    editor.focus();
    editor.setSelectionRange(editor.value.length, editor.value.length);
    editor.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        save.click();
      } else if (event.key === "Escape") {
        renderConversationHistory();
      }
    });
  }

  async function submitEdit(index, newText) {
    if (isStreaming || !messages[index]) return;
    if (!modelSelect.value) {
      toast("Select a model to regenerate the response.", "error", 6000);
      return;
    }
    // Editing always drops this message's reply; warn only if more follows.
    if (messages.length - index > 2 &&
        !window.confirm("Editing this message will discard the rest of the conversation after it. Continue?")) {
      return;
    }
    replaceMessageText(messages[index], newText);
    messages = messages.slice(0, index + 1);
    renderConversationHistory();
    updateExportState();
    hideError();
    isStreaming = true;
    updateSendState();
    persistConversation();
    await streamAssistantTurn();
  }

  async function regenerateFrom(index) {
    if (isStreaming || !messages[index]) return;
    if (!modelSelect.value) {
      toast("Select a model to regenerate the response.", "error", 6000);
      return;
    }
    if (index < messages.length - 1 &&
        !window.confirm("Regenerating this response will discard the rest of the conversation after it. Continue?")) {
      return;
    }
    messages = messages.slice(0, index);
    renderConversationHistory();
    updateExportState();
    hideError();
    isStreaming = true;
    updateSendState();
    await streamAssistantTurn();
  }

  async function openConversation(id) {
    if (isStreaming || id === currentConversationId) return;
    try {
      const response = await fetch(`/api/conversations/${id}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not load the conversation.");
      messages = Array.isArray(data.messages) ? data.messages : [];
      currentConversationId = id;
      currentTitle = data.title || "";
      attachments = [];
      // Rehydrate RAG doc ids from saved metadata. After a server restart the
      // in-memory index is gone; retrieval then returns nothing and the model
      // simply answers without the document context (no error).
      docIds = new Set();
      for (const message of messages) {
        for (const meta of (message.attachments_meta || [])) {
          if (meta.doc_id) docIds.add(meta.doc_id);
        }
      }
      renderAttachments();
      renderConversationHistory();
      highlightActiveConversation();
      updateContextMeter();
      updateExportState();
      closeSidebarOnMobile();
    } catch (error) {
      toast(error.message || "Could not load the conversation.", "error", 6000);
    }
  }

  async function ensureConversation() {
    if (currentConversationId) return currentConversationId;
    const created = await fetch("/api/conversations", { method: "POST" });
    if (!created.ok) throw new Error("create failed");
    currentConversationId = (await created.json()).id;
    return currentConversationId;
  }

  async function persistConversation() {
    if (!messages.length) return;
    // Snapshot what we're saving and where, so a concurrent New-chat / open
    // can't make us write the wrong messages into the wrong conversation.
    const snapshot = JSON.stringify(messages);
    try {
      let targetId = await ensureConversation();
      let response = await fetch(`/api/conversations/${targetId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: JSON.parse(snapshot) }),
      });
      // The active conversation was deleted (e.g. mid-stream): recreate it
      // so the in-progress chat isn't silently lost.
      if (response.status === 404 && targetId === currentConversationId) {
        currentConversationId = null;
        targetId = await ensureConversation();
        response = await fetch(`/api/conversations/${targetId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: JSON.parse(snapshot) }),
        });
      }
      if (!response.ok) throw new Error(`status ${response.status}`);
      try {
        currentTitle = (await response.json()).title || currentTitle;
      } catch {
        /* Body already consumed or empty; keep the existing title. */
      }
      updateExportState();
      loadConversations();
    } catch {
      toast("This conversation could not be saved.", "error", 6000);
    }
  }

  // ---------- Export conversation ----------

  /** The export button only works on a conversation that has messages. */
  function updateExportState() {
    if (exportToggle) exportToggle.disabled = !messages.length;
    if (messages.length === 0) closeExportMenu();
  }

  function closeExportMenu() {
    exportMenu.classList.add("hidden");
    exportToggle.setAttribute("aria-expanded", "false");
  }

  function toggleExportMenu() {
    if (exportToggle.disabled) return;
    const opening = exportMenu.classList.contains("hidden");
    exportMenu.classList.toggle("hidden", !opening);
    exportToggle.setAttribute("aria-expanded", String(opening));
  }

  exportToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleExportMenu();
  });

  exportMenu.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-format]");
    if (!button) return;
    closeExportMenu();
    exportConversation(button.dataset.format);
  });

  // Dismiss the menu on an outside click or Escape.
  document.addEventListener("click", (event) => {
    if (!exportMenu.classList.contains("hidden") &&
        !exportMenu.contains(event.target) && event.target !== exportToggle) {
      closeExportMenu();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeExportMenu();
  });

  /** A filesystem-safe slug derived from the conversation title. */
  function exportFileBase() {
    const base = (currentTitle || "conversation")
      .replace(/[\\/:*?"<>|]+/g, "")  // characters illegal in filenames
      .replace(/\s+/g, "-")
      .replace(/^[-.]+|[-.]+$/g, "")
      .slice(0, 80);
    return base || "conversation";
  }

  /** Describe a message's attachments for text/markdown exports. */
  function attachmentSummary(message) {
    const meta = Array.isArray(message.attachments_meta) ? message.attachments_meta : [];
    return meta.map((entry) => entry.kind === "image"
      ? `🖼 ${entry.name || "image"}`
      : `📄 ${entry.name || "document"}${entry.pages ? ` (${entry.pages} p.)` : ""}`);
  }

  function conversationToMarkdown() {
    const lines = [`# ${currentTitle || "Conversation"}`, ""];
    lines.push(`*Exported from LocalMind on ${new Date().toLocaleString()}*`, "");
    for (const message of messages) {
      if (message.role !== "user" && message.role !== "assistant") continue;
      lines.push(message.role === "user" ? "## 🧑 You" : "## 🤖 Assistant", "");
      const attachments = attachmentSummary(message);
      if (attachments.length) {
        lines.push(attachments.map((tag) => `> ${tag}`).join("\n"), "");
      }
      const text = message.display ?? extractMessageText(message.content);
      lines.push(text || "*(no text content)*", "");
    }
    return lines.join("\n");
  }

  function conversationToJson() {
    return JSON.stringify({
      id: currentConversationId,
      title: currentTitle || "Conversation",
      exported_at: new Date().toISOString(),
      messages,
    }, null, 2);
  }

  function downloadBlob(text, mimeType, filename) {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    // Revoke after the click has been dispatched.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /**
   * "Print to PDF": render the conversation into a hidden, print-only node and
   * invoke the browser's print dialog. The user picks "Save as PDF". This reuses
   * the sanitized-Markdown renderer for fidelity and needs no extra libraries.
   */
  function exportPdf() {
    printRoot.innerHTML = "";
    const heading = document.createElement("h1");
    heading.className = "print-title";
    heading.textContent = currentTitle || "Conversation";
    printRoot.appendChild(heading);

    for (const message of messages) {
      if (message.role !== "user" && message.role !== "assistant") continue;
      const block = document.createElement("section");
      block.className = `print-message ${message.role}`;

      const label = document.createElement("div");
      label.className = "print-role";
      label.textContent = message.role === "user" ? "You" : "Assistant";
      block.appendChild(label);

      const attachments = attachmentSummary(message);
      if (attachments.length) {
        const tags = document.createElement("div");
        tags.className = "print-attachments";
        tags.textContent = attachments.join("  ·  ");
        block.appendChild(tags);
      }

      const body = document.createElement("div");
      const text = message.display ?? extractMessageText(message.content);
      // Assistant replies are Markdown; user text is shown verbatim.
      if (message.role === "assistant") {
        renderMarkdown(body, text, { withCopyButtons: false });
      } else {
        body.className = "print-user-text";
        body.textContent = text;
      }
      block.appendChild(body);
      printRoot.appendChild(block);
    }

    document.body.classList.add("printing");
    const cleanup = () => {
      document.body.classList.remove("printing");
      printRoot.innerHTML = "";
      window.removeEventListener("afterprint", cleanup);
    };
    window.addEventListener("afterprint", cleanup);
    window.print();
    // Safari/Firefox don't always fire afterprint; clear shortly after as a fallback.
    setTimeout(cleanup, 2000);
  }

  function exportConversation(format) {
    if (!messages.length) return;
    const base = exportFileBase();
    try {
      if (format === "markdown") {
        downloadBlob(conversationToMarkdown(), "text/markdown;charset=utf-8", `${base}.md`);
        toast("Conversation exported as Markdown.");
      } else if (format === "json") {
        downloadBlob(conversationToJson(), "application/json;charset=utf-8", `${base}.json`);
        toast("Conversation exported as JSON.");
      } else if (format === "pdf") {
        exportPdf();
      }
    } catch (error) {
      showError(error.message || "Export failed.");
    }
  }

  // ---------- Streaming chat ----------

  async function sendMessage() {
    const content = messageInput.value.trim();
    if (!content || isStreaming || (!imageMode && !modelSelect.value)) return;

    hideError();
    isStreaming = true;
    messageInput.value = "";
    messageInput.style.height = "auto";
    updateSendState();

    if (imageMode) {
      // Image prompts are handled separately and intentionally kept out of
      // the text-chat history sent to the LLM.
      await sendImageMessage(content);
      return;
    }

    const docs = attachments.filter((doc) => !doc.uploading && doc.kind !== "image");
    const images = attachments.filter((doc) => !doc.uploading && doc.kind === "image");
    const bubble = appendMessage("user", content);
    if (docs.length || images.length) {
      const tags = document.createElement("div");
      tags.className = "message-attachments";
      for (const image of images) {
        const thumb = document.createElement("img");
        thumb.className = "msg-thumb";
        thumb.src = image.dataUrl;
        thumb.alt = image.name;
        tags.appendChild(thumb);
      }
      for (const doc of docs) {
        const tag = document.createElement("span");
        tag.className = "doc-tag";
        tag.textContent = `📄 ${doc.name}${doc.pages ? ` (${doc.pages} p.)` : ""}`;
        tags.appendChild(tag);
      }
      bubble.prepend(tags);
    }
    // Large docs are retrieved by doc_id (RAG); small docs are inlined as text.
    // images become OpenAI image_url content parts for vision models.
    const ragDocs = docs.filter((doc) => doc.rag && doc.doc_id);
    const inlineDocs = docs.filter((doc) => !doc.rag);
    for (const doc of ragDocs) docIds.add(doc.doc_id);

    const docBlocks = inlineDocs
      .map((doc) => `[Attached document: ${doc.name}]\n${doc.text}`)
      .join("\n\n");
    const fullText = docBlocks ? `${docBlocks}\n\n${content}` : content;
    messages.push({
      role: "user",
      content: images.length
        ? [
            ...images.map((image) => ({
              type: "image_url",
              image_url: { url: image.dataUrl },
            })),
            { type: "text", text: fullText },
          ]
        : fullText,
      // Display-only fields for restoring saved conversations; the backend
      // strips them before talking to LM Studio.
      display: content,
      attachments_meta: [
        ...images.map((image) => ({ kind: "image", name: image.name, dataUrl: image.dataUrl })),
        ...docs.map((doc) => ({
          kind: "document", name: doc.name, pages: doc.pages,
          ...(doc.rag && doc.doc_id ? { doc_id: doc.doc_id } : {}),
        })),
      ],
    });
    attachments = [];
    renderAttachments();
    attachMessageActions(bubble, "user", messages.length - 1);

    // Persist the user turn before streaming so a reload/crash mid-generation
    // doesn't lose the prompt (local models can stream for minutes).
    persistConversation();

    await streamAssistantTurn();
  }

  /**
   * Stream one assistant reply for the current `messages` history into a new
   * bubble. Expects `isStreaming` to already be true; appends the reply to
   * the history and persists the conversation when done.
   */
  async function streamAssistantTurn() {
    const assistantBubble = appendMessage("assistant", "");
    assistantBubble.classList.add("pending");
    let assistantContent = "";

    chatAbortController = new AbortController();
    updateSendState();

    const payload = {
      model: modelSelect.value,
      messages: systemPrompt
        ? [{ role: "system", content: systemPrompt }, ...messages]
        : messages,
      temperature: Number(temperatureInput.value),
      max_tokens: Math.max(1, Math.floor(Number(maxTokensInput.value) || 1024)),
      ...(reasoningEffortValue() ? { reasoning_effort: reasoningEffortValue() } : {}),
      ...(docIds.size ? { doc_ids: [...docIds] } : {}),
      ...(webSearchEnabled ? { web_search: true } : {}),
      // Drive LM Studio's idle auto-unload from the load panel's TTL control.
      // It only takes effect when this message JIT-loads the model; an
      // already-loaded instance keeps the TTL it was loaded with. Unchecked
      // sends 0 to disable auto-unload.
      ttl_seconds: ttlEnabledInput.checked
        ? Math.max(1, Math.floor(Number(ttlSecondsInput.value) || 1))
        : 0,
    };

    // Lazily built structure inside the assistant bubble: an optional
    // collapsible thinking block followed by the answer text.
    let reasoningContent = "";
    let thinkingDetails = null;
    let thinkingBody = null;
    let contentSpan = null;

    function ensureContentSpan() {
      if (!contentSpan) {
        assistantBubble.textContent = "";
        contentSpan = document.createElement("span");
        assistantBubble.appendChild(contentSpan);
      }
    }

    function appendReasoning(text) {
      ensureContentSpan();
      if (!thinkingDetails) {
        thinkingDetails = document.createElement("details");
        thinkingDetails.className = "thinking";
        // Stays collapsed by default; the user can expand it to read the
        // model's reasoning rather than having it pop open on every reply.
        const summary = document.createElement("summary");
        summary.textContent = "💭 Thinking…";
        thinkingBody = document.createElement("div");
        thinkingBody.className = "thinking-body";
        thinkingDetails.append(summary, thinkingBody);
        assistantBubble.insertBefore(thinkingDetails, contentSpan);
      }
      reasoningContent += text;
      thinkingBody.textContent = reasoningContent;
    }

    function finishThinking() {
      if (thinkingDetails) {
        // Relabel once reasoning is done, but leave the open/closed state alone
        // so we don't override whatever the user toggled.
        const tokenCount = estimateTokens(reasoningContent);
        thinkingDetails.querySelector("summary").textContent =
          `💭 Thoughts (≈${formatTokens(tokenCount)} tokens)`;
      }
    }

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: chatAbortController.signal,
      });
      if (!response.ok || !response.body) {
        let detail = `Request failed with status ${response.status}.`;
        try {
          const data = await response.json();
          detail = data.error || JSON.stringify(data.detail) || detail;
        } catch { /* keep generic message */ }
        throw new Error(detail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by a blank line.
        const events = buffer.split("\n\n");
        buffer = events.pop();

        for (const event of events) {
          const line = event.trim();
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") continue;

          const parsed = JSON.parse(data);
          if (parsed.error) throw new Error(parsed.error);
          // Capture pin state before mutating, so growth doesn't unpin us.
          const pinned = isPinnedToBottom();
          if (parsed.status && !assistantContent) {
            // Transient progress note (e.g. web search); the first answer
            // tokens overwrite it via renderMarkdown below.
            ensureContentSpan();
            contentSpan.textContent = "";
            const note = document.createElement("em");
            note.className = "stream-status";
            note.textContent = parsed.status;
            contentSpan.appendChild(note);
            if (pinned) chatWindow.scrollTop = chatWindow.scrollHeight;
          }
          if (parsed.notice) {
            // Non-fatal server-side warning (e.g. web search failed).
            toast(parsed.notice, "error", 8000);
          }
          if (parsed.reasoning) {
            appendReasoning(parsed.reasoning);
            if (pinned) chatWindow.scrollTop = chatWindow.scrollHeight;
          }
          if (parsed.content) {
            if (!assistantContent) finishThinking();
            ensureContentSpan();
            assistantContent += parsed.content;
            renderMarkdown(contentSpan, assistantContent, { withCopyButtons: false });
            if (pinned) chatWindow.scrollTop = chatWindow.scrollHeight;
          }
        }
      }

      finishThinking();
      // Final render adds the code-block copy buttons.
      if (assistantContent && contentSpan) renderMarkdown(contentSpan, assistantContent);
      // Drop a leftover status note if the stream produced no answer text.
      if (!assistantContent && contentSpan) contentSpan.textContent = "";
      // Only the final answer (not the thinking) goes back into the history.
      messages.push({ role: "assistant", content: assistantContent });
      attachMessageActions(assistantBubble, "assistant", messages.length - 1);
    } catch (error) {
      if (assistantContent) {
        // Keep the partial answer in history so the conversation stays coherent.
        if (contentSpan) renderMarkdown(contentSpan, assistantContent);
        messages.push({ role: "assistant", content: assistantContent });
        attachMessageActions(assistantBubble, "assistant", messages.length - 1);
      } else {
        assistantBubble.remove();
      }
      if (error.name === "AbortError") {
        // Stopped by the user: keep whatever streamed in, no error banner.
        finishThinking();
      } else {
        showError(error.message || "The request to the server failed.");
      }
    } finally {
      chatAbortController = null;
      assistantBubble.classList.remove("pending");
      isStreaming = false;
      updateSendState();
      messageInput.focus();
      persistConversation();
    }
  }

  // ---------- Init ----------

  // loadPresets() runs after loadDefaults() so the config-default prompt is in
  // place as a fallback before presets (the source of truth) load.
  loadDefaults().then(loadPresets);
  loadModels();
  checkImageCapability();
  loadConversations();
  messageInput.focus();
})();
