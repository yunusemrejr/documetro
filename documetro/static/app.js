const state = {
  status: null,
  messages: [],
  busy: false,
};

const dom = {
  chatFeed: document.querySelector("#chat-feed"),
  questionForm: document.querySelector("#question-form"),
  questionInput: document.querySelector("#question-input"),
  sendButton: document.querySelector("#send-button"),
  dropzone: document.querySelector("#dropzone"),
  uploadButton: document.querySelector("#upload-button"),
  dropzoneText: document.querySelector(".dropzone__text"),
  fileInput: document.querySelector("#file-input"),
  welcomeTemplate: document.querySelector("#welcome-template"),
  infoButton: document.querySelector("#info-button"),
  configButton: document.querySelector("#config-button"),
  infoModal: document.querySelector("#info-modal"),
  configModal: document.querySelector("#config-modal"),
  infoClose: document.querySelector("#info-close"),
  configClose: document.querySelector("#config-close"),
  configForm: document.querySelector("#config-form"),
  configSave: document.querySelector("#config-save"),
  configStatus: document.querySelector("#config-status"),
  infoProvider: document.querySelector("#info-provider"),
  infoTextEmbedding: document.querySelector("#info-text-embedding"),
  infoVisualEmbedding: document.querySelector("#info-visual-embedding"),
  infoEmbeddingStrategy: document.querySelector("#info-embedding-strategy"),
  infoMultimodalModel: document.querySelector("#info-multimodal-model"),
  infoGenerationModel: document.querySelector("#info-generation-model"),
  infoReasoningProvider: document.querySelector("#info-reasoning-provider"),
  infoReasoningModel: document.querySelector("#info-reasoning-model"),
};

const CONFIG_FIELDS = [
  "openrouter_api_key",
  "openrouter_base_url",
  "openrouter_embedding_model",
  "openrouter_multimodal_embedding_model",
  "openrouter_multimodal_model",
  "openrouter_generation_model",
  "nous_api_key",
  "nous_base_url",
  "nous_reasoning_model",
];

const CONFIG_DEFAULTS = {
  openrouter_api_key: "",
  openrouter_base_url: "https://openrouter.ai/api/v1",
  openrouter_embedding_model: "nvidia/llama-3.2-nv-embedqa-1b-v2:free",
  openrouter_multimodal_embedding_model: "nvidia/llama-nemotron-embed-vl-1b-v2:free",
  openrouter_multimodal_model: "openai/gpt-4o-mini",
  openrouter_generation_model: "liquid/lfm2-8b-a1b",
  nous_api_key: "",
  nous_base_url: "https://inference-api.nousresearch.com/v1",
  nous_reasoning_model: "Hermes-4-70B",
};

let activeModal = null;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function trimLine(value, max = 220) {
  const text = String(value || "").trim();
  if (text.length <= max) {
    return text;
  }
  return `${text.slice(0, max).trimEnd()}…`;
}

function renderRichText(value) {
  return String(value || "")
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line, index, lines) => line || (index > 0 && index < lines.length - 1))
    .map((line) => `<p class="message-card__text">${escapeHtml(line)}</p>`)
    .join("");
}

function assistantMessageCard(message) {
  const lead = trimLine(message.lead || message.text || "");
  const normalizedLead = lead.toLowerCase().trim();
  const fullText = String(message.text || "").trim();
  const hasExpandedBody = fullText && fullText.toLowerCase().trim() !== normalizedLead;
  const uniqueSteps = (message.steps || [])
    .filter(Boolean)
    .filter((item) => trimLine(item, 180).toLowerCase().trim() !== normalizedLead);
  const steps = uniqueSteps
    .map((item) => `<li>${escapeHtml(trimLine(item, 180))}</li>`)
    .join("");
  const bullets = (message.bullets || [])
    .filter(Boolean)
    .filter((item) => trimLine(item, 180).toLowerCase().trim() !== normalizedLead)
    .map((item) => `<li>${escapeHtml(trimLine(item, 180))}</li>`)
    .join("");
  const trimmedQuote = trimLine(message.quote || "", 180);
  const quoteMatchesLead = trimmedQuote.toLowerCase().trim() === normalizedLead;
  const quoteMatchesStep = uniqueSteps.some((item) => trimLine(item, 180).toLowerCase().trim() === trimmedQuote.toLowerCase().trim());
  const quote = message.quote
    && !quoteMatchesLead
    && !quoteMatchesStep
    ? `
      <figure class="message-card__quote">
        <blockquote>${escapeHtml(trimmedQuote)}</blockquote>
        ${message.quote_source ? `<figcaption class="message-card__source">${escapeHtml(trimLine(message.quote_source, 120))}</figcaption>` : ""}
      </figure>
    `
    : "";
  const note = message.note ? `<p class="message-card__note">${escapeHtml(trimLine(message.note, 140))}</p>` : "";

  return `
    <div class="message-row message-row--assistant">
      <div class="assistant-badge" aria-hidden="true">
        <img src="/assets/documetro_transparent.png" alt="" />
      </div>
      <article class="message-card message-card--assistant">
        ${hasExpandedBody ? renderRichText(fullText) : (lead ? `<p class="message-card__text">${escapeHtml(lead)}</p>` : "")}
        ${steps ? `<ol class="message-card__list message-card__list--ordered">${steps}</ol>` : ""}
        ${!steps && bullets ? `<ul class="message-card__list">${bullets}</ul>` : ""}
        ${quote}
        ${note}
      </article>
    </div>
  `;
}

function userMessageCard(message) {
  return `
    <div class="message-row message-row--user">
      <article class="message-card message-card--user">
        <p class="message-card__text">${escapeHtml(message.text)}</p>
      </article>
    </div>
  `;
}

function messageCard(message) {
  if (message.role === "user") {
    return userMessageCard(message);
  }
  return assistantMessageCard(message);
}

function renderMessages() {
  if (!state.messages.length) {
    dom.chatFeed.innerHTML = "";
    dom.chatFeed.appendChild(dom.welcomeTemplate.content.cloneNode(true));
    return;
  }
  dom.chatFeed.innerHTML = state.messages.map(messageCard).join("");
  dom.chatFeed.scrollTop = dom.chatFeed.scrollHeight;
}

function setUploadSummary(status) {
  if (!status) {
    dom.dropzoneText.textContent = "Drop files or browse";
    return;
  }
  if (status.state === "processing" || status.state === "queued") {
    dom.dropzoneText.textContent = "Indexing…";
    return;
  }
  if ((status.document_count || 0) > 0) {
    dom.dropzoneText.textContent = `${status.document_count} document${status.document_count !== 1 ? 's' : ''} loaded`;
    return;
  }
  dom.dropzoneText.textContent = "Drop files or browse";
}

function renderStatus(status) {
  state.status = status;
  setUploadSummary(status);
  const canAsk = (status.document_count || 0) > 0 && status.state !== "processing" && status.state !== "queued";
  dom.questionInput.disabled = !canAsk || state.busy;
  dom.sendButton.disabled = !canAsk || state.busy;
  hydrateInfoPanel(status);
}

function hydrateInfoPanel(status) {
  if (!status) {
    return;
  }
  if (dom.infoProvider) {
    dom.infoProvider.textContent = status.provider === "openrouter" ? "OpenRouter" : "Local only";
  }
  if (dom.infoTextEmbedding && status.embedding_model) {
    dom.infoTextEmbedding.textContent = status.embedding_model;
  }
  if (dom.infoVisualEmbedding && state.config?.openrouter_multimodal_embedding_model) {
    dom.infoVisualEmbedding.textContent = state.config.openrouter_multimodal_embedding_model;
  }
  if (dom.infoEmbeddingStrategy) {
    const strategy = status.embedding_strategy || "pending";
    dom.infoEmbeddingStrategy.textContent = strategy === "disabled"
      ? "Embeddings disabled. Retrieval falls back to lexical and latent search."
      : `${strategy} corpus strategy.`;
  }
  if (dom.infoMultimodalModel) {
    dom.infoMultimodalModel.textContent = state.config?.openrouter_multimodal_model || "Disabled";
  }
  if (dom.infoGenerationModel) {
    dom.infoGenerationModel.textContent = status.generation_model || "Disabled";
  }
  if (dom.infoReasoningProvider) {
    dom.infoReasoningProvider.textContent = status.reasoning_provider === "nous" ? "Nous Hermes" : "Disabled";
  }
  if (dom.infoReasoningModel) {
    dom.infoReasoningModel.textContent = status.reasoning_model || "Disabled";
  }
}

function openModal(modal) {
  if (!modal) {
    return;
  }
  modal.hidden = false;
  activeModal = modal;
  document.body.style.overflow = "hidden";
}

function closeModal(modal = activeModal) {
  if (!modal) {
    return;
  }
  modal.hidden = true;
  if (activeModal === modal) {
    activeModal = null;
  }
  document.body.style.overflow = "";
}

function maskSecret(value) {
  const text = String(value || "").trim();
  if (!text) {
    return "Not set";
  }
  if (text.length <= 8) {
    return `${"*".repeat(Math.max(text.length - 2, 0))}${text.slice(-2)}`;
  }
  return `${text.slice(0, 4)}${"*".repeat(Math.max(text.length - 8, 4))}${text.slice(-4)}`;
}

function configMetaText(field, value) {
  const text = String(value || "").trim();
  const defaultValue = String(CONFIG_DEFAULTS[field] || "").trim();
  if (field.endsWith("_api_key")) {
    return `Current: ${maskSecret(text)}`;
  }
  if (!text && !defaultValue) {
    return "Current: Not set";
  }
  if (!text && defaultValue) {
    return `Default: ${defaultValue}`;
  }
  if (defaultValue && text === defaultValue) {
    return `Current default: ${text}`;
  }
  return `Current: ${text}`;
}

function hydrateConfigForm(config) {
  CONFIG_FIELDS.forEach((field) => {
    const input = dom.configForm?.elements.namedItem(field);
    const value = String(config?.[field] || "");
    if (input) {
      input.value = value;
      input.placeholder = CONFIG_DEFAULTS[field] || "";
    }
    const meta = document.querySelector(`[data-config-meta="${field}"]`);
    if (meta) {
      meta.textContent = configMetaText(field, value);
    }
  });
}

async function loadConfig() {
  const payload = await api("/api/config");
  state.config = payload;
  hydrateConfigForm(payload);
  hydrateInfoPanel(state.status || {});
}

async function saveConfig(formData) {
  const payload = Object.fromEntries(formData.entries());
  dom.configSave.disabled = true;
  dom.configStatus.textContent = "Saving settings…";
  try {
    const result = await api("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.config = result.config || payload;
    hydrateConfigForm(state.config);
    renderStatus(result.status || state.status || {});
    dom.configStatus.textContent = "Saved. Provider settings refreshed and workspace cleared.";
  } catch (error) {
    dom.configStatus.textContent = error.message;
  } finally {
    dom.configSave.disabled = false;
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed.");
  }
  return payload;
}

async function refreshStatus() {
  try {
    const payload = await api("/api/status");
    renderStatus(payload);
  } catch (error) {
    setUploadSummary(null);
  }
}

async function uploadFiles(files) {
  if (!files?.length) {
    return;
  }
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));
  try {
    state.busy = true;
    setUploadSummary({ state: "queued" });
    const payload = await api("/api/upload", { method: "POST", body: formData });
    if ((payload.accepted || []).length) {
      state.messages.push({
        role: "assistant",
        text: "Files added.",
        lead: "Files added.",
        bullets: [],
        steps: [],
        note: "",
        quote: "",
        quote_source: "",
      });
      renderMessages();
    }
  } catch (error) {
    state.messages.push({
      role: "assistant",
      text: error.message,
      lead: error.message,
      bullets: [],
      steps: [],
      note: "",
      quote: "",
      quote_source: "",
    });
    renderMessages();
  } finally {
    state.busy = false;
    dom.fileInput.value = "";
    await refreshStatus();
  }
}

async function askQuestion(question) {
  state.messages.push({ role: "user", text: question });
  renderMessages();
  try {
    state.busy = true;
    renderStatus(state.status || { document_count: 0, state: "ready" });
    dom.questionInput.value = "";
    const payload = await api("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    state.messages.push({
      role: "assistant",
      text: payload.answer,
      lead: payload.lead || payload.answer,
      bullets: payload.bullets || [],
      steps: payload.steps || [],
      note: payload.note || "",
      quote: payload.quote || "",
      quote_source: payload.quote_source || "",
      template: payload.template || "",
    });
    renderMessages();
  } catch (error) {
    state.messages.push({
      role: "assistant",
      text: error.message,
      lead: error.message,
      bullets: [],
      steps: [],
      note: "",
      quote: "",
      quote_source: "",
    });
    renderMessages();
  } finally {
    state.busy = false;
    await refreshStatus();
  }
}

dom.uploadButton.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  dom.fileInput.click();
});

dom.fileInput.addEventListener("change", () => uploadFiles(dom.fileInput.files));

dom.dropzone.addEventListener("click", (event) => {
  if (event.target.closest("#upload-button")) {
    return;
  }
  dom.fileInput.click();
});

dom.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    dom.fileInput.click();
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dom.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dom.dropzone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dom.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dom.dropzone.classList.remove("is-dragover");
  });
});

dom.dropzone.addEventListener("drop", (event) => {
  uploadFiles(event.dataTransfer?.files);
});

dom.questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = dom.questionInput.value.trim();
  if (!question || state.busy) {
    return;
  }
  await askQuestion(question);
});

dom.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    dom.questionForm.dispatchEvent(new Event("submit"));
  }
});

dom.infoButton?.addEventListener("click", () => openModal(dom.infoModal));
dom.configButton?.addEventListener("click", async () => {
  dom.configStatus.textContent = "";
  await loadConfig().catch((error) => {
    dom.configStatus.textContent = error.message;
  });
  openModal(dom.configModal);
});
dom.infoClose?.addEventListener("click", () => closeModal(dom.infoModal));
dom.configClose?.addEventListener("click", () => closeModal(dom.configModal));
document.querySelectorAll("[data-close-modal]").forEach((node) => {
  node.addEventListener("click", () => closeModal(node.parentElement));
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModal();
  }
});
dom.configForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  await saveConfig(new FormData(dom.configForm));
});

renderMessages();
refreshStatus();
loadConfig().catch(() => {});
setInterval(refreshStatus, 2000);
