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

  /** Conversation history sent to the backend on every request. */
  let messages = [];
  let systemPrompt = "";
  let isStreaming = false;

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
  });

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
      const { defaults } = await response.json();
      temperatureInput.value = defaults.temperature;
      temperatureValue.textContent = Number(defaults.temperature).toFixed(2);
      maxTokensInput.value = defaults.max_tokens;
      systemPrompt = defaults.system_prompt || "";
    } catch {
      /* Non-fatal: the UI falls back to its hardcoded defaults. */
    }
  }

  // ---------- Models ----------

  async function loadModels() {
    modelSelect.disabled = true;
    modelSelect.innerHTML = "<option value=''>Loading models…</option>";
    try {
      const response = await fetch("/api/models");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to load models.");
      if (!data.models.length) {
        throw new Error("LM Studio is reachable but no models are loaded. Load a model in LM Studio first.");
      }
      modelSelect.innerHTML = "";
      for (const id of data.models) {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = id;
        modelSelect.appendChild(option);
      }
      modelSelect.disabled = false;
      updateSendState();
    } catch (error) {
      modelSelect.innerHTML = "<option value=''>No models available</option>";
      showError(error.message || "Could not reach the backend.");
    }
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
