"use strict";

(() => {
  const modelSelect = document.getElementById("model-select");
  const settingsToggle = document.getElementById("settings-toggle");
  const settingsPanel = document.getElementById("settings-panel");
  const clearChatButton = document.getElementById("clear-chat");
  const errorBanner = document.getElementById("error-banner");
  const errorText = document.getElementById("error-text");
  const errorRetry = document.getElementById("error-retry");
  const errorDismiss = document.getElementById("error-dismiss");
  const temperatureInput = document.getElementById("temperature");
  const temperatureValue = document.getElementById("temperature-value");
  const maxTokensInput = document.getElementById("max-tokens");
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

  /** Conversation history sent to the backend on every request. */
  let messages = [];
  let systemPrompt = "";
  let isStreaming = false;
  let statusRefreshSeconds = 10;
  let statusRefreshTimer = null;
  /** Model keys / instance ids with an in-flight load or unload request. */
  const busyModels = new Set();

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
      const { defaults, model_management: management } = await response.json();
      temperatureInput.value = defaults.temperature;
      temperatureValue.textContent = Number(defaults.temperature).toFixed(2);
      maxTokensInput.value = defaults.max_tokens;
      systemPrompt = defaults.system_prompt || "";
      if (management) {
        ttlEnabledInput.checked = Boolean(management.auto_unload_by_default);
        ttlSecondsInput.value = management.default_ttl_seconds ?? 600;
        if (management.default_context_length) {
          contextLengthInput.value = management.default_context_length;
        }
        statusRefreshSeconds = management.status_refresh_seconds || 10;
      }
    } catch {
      /* Non-fatal: the UI falls back to its hardcoded defaults. */
    }
  }

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

  function loadOptions() {
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

    for (const model of models) {
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
          label.textContent = instance.context_length
            ? `${instance.id} — ctx ${formatContext(instance.context_length)}`
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

    modelsSummary.textContent =
      `${loadedCount} of ${models.length} models loaded` +
      (loadedBytes ? ` · ~${formatBytes(loadedBytes)} in memory` : "");
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

  function updateSendState() {
    sendButton.disabled =
      isStreaming || !modelSelect.value || !messageInput.value.trim();
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

  sendButton.addEventListener("click", sendMessage);
  modelSelect.addEventListener("change", updateSendState);

  clearChatButton.addEventListener("click", () => {
    if (isStreaming) return;
    messages = [];
    chatWindow.querySelectorAll(".message").forEach((node) => node.remove());
  });

  // ---------- Streaming chat ----------

  async function sendMessage() {
    const content = messageInput.value.trim();
    if (!content || isStreaming || !modelSelect.value) return;

    hideError();
    isStreaming = true;
    messageInput.value = "";
    messageInput.style.height = "auto";
    updateSendState();

    appendMessage("user", content);
    messages.push({ role: "user", content });

    const assistantBubble = appendMessage("assistant", "");
    assistantBubble.classList.add("pending");
    let assistantContent = "";

    const payload = {
      model: modelSelect.value,
      messages: systemPrompt
        ? [{ role: "system", content: systemPrompt }, ...messages]
        : messages,
      temperature: Number(temperatureInput.value),
      max_tokens: Math.max(1, Math.floor(Number(maxTokensInput.value) || 1024)),
    };

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
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
          if (parsed.content) {
            assistantContent += parsed.content;
            assistantBubble.textContent = assistantContent;
            chatWindow.scrollTop = chatWindow.scrollHeight;
          }
        }
      }

      messages.push({ role: "assistant", content: assistantContent });
    } catch (error) {
      if (assistantContent) {
        // Keep the partial answer in history so the conversation stays coherent.
        messages.push({ role: "assistant", content: assistantContent });
      } else {
        assistantBubble.remove();
      }
      showError(error.message || "The request to the server failed.");
    } finally {
      assistantBubble.classList.remove("pending");
      isStreaming = false;
      updateSendState();
      messageInput.focus();
    }
  }

  // ---------- Init ----------

  loadDefaults();
  loadModels();
  messageInput.focus();
})();
