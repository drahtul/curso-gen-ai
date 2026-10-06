const LOCAL_HOSTS = ["localhost", "127.0.0.1", "[::1]", ""];

function resolveApiBase() {
  const configured = (window.AGENT_CONFIG?.apiBaseUrl || "").trim();
  const value =
    configured ||
    (LOCAL_HOSTS.includes(location.hostname)
      ? "http://127.0.0.1:8000"
      : location.origin);
  const trimmed = value.replace(/\/+$/, "");
  return trimmed.startsWith("/") ? `${location.origin}${trimmed}` : trimmed;
}

const API_BASE = resolveApiBase();

const GREETING =
  "Hola. Puedo ayudarte con productos, stock, politicas de la tienda y cotizaciones.";
const MEMORY_LOST_NOTICE =
  "El backend reinicio su memoria para este chat; se muestra la ultima copia " +
  "guardada en este navegador, pero el agente ya no recuerda estos mensajes.";

const THREADS_KEY = "agent-front:threads";
const MAX_THREADS = 50;

function loadThreads() {
  try {
    const raw = localStorage.getItem(THREADS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveThreads(threads) {
  try {
    localStorage.setItem(THREADS_KEY, JSON.stringify(threads));
  } catch {
    /* localStorage lleno o modo privado: se pierde el historial, no rompe la app */
  }
}

function findThread(threads, id) {
  return threads.find((t) => t.id === id);
}

function titleFrom(text) {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length > 42
    ? `${clean.slice(0, 42)}…`
    : clean || "Nueva conversacion";
}

const el = {
  sidebar: document.getElementById("sidebar"),
  sidebarToggle: document.getElementById("sidebar-toggle"),
  threadList: document.getElementById("thread-list"),
  newChat: document.getElementById("new-chat"),
  messages: document.getElementById("messages"),
  form: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  threadId: document.getElementById("thread-id"),
  dot: document.getElementById("status-dot"),
  suggestions: document.getElementById("suggestions"),
};

let threads = loadThreads();
let threadId = null;
let busy = false;

function persistFromDom() {
  if (!threadId) return;
  const thread = findThread(threads, threadId);
  if (!thread) return;
  thread.messages = Array.from(el.messages.querySelectorAll(".msg")).map(
    (wrap) => ({
      role: [...wrap.classList].find((c) => c !== "msg") || "agent",
      text: wrap.querySelector(".bubble")?.textContent ?? "",
    }),
  );
  thread.updatedAt = Date.now();
  saveThreads(threads);
  renderThreadList();
}

function applyHistory(id, historyTurns, fallbackTitle) {
  const messages = historyTurns.map((t) => ({ role: t.role, text: t.content }));
  let thread = findThread(threads, id);
  if (!thread) {
    const firstUserText = historyTurns.find((t) => t.role === "user")?.content;
    thread = {
      id,
      title: firstUserText ? titleFrom(firstUserText) : fallbackTitle,
      updatedAt: Date.now(),
      messages: [],
    };
    threads.unshift(thread);
    threads = threads.slice(0, MAX_THREADS);
  }
  thread.messages = messages;
  thread.updatedAt = Date.now();
  saveThreads(threads);
  if (id === threadId) renderMessages(messages);
  renderThreadList();
}

function renderThreadList() {
  const sorted = [...threads].sort((a, b) => b.updatedAt - a.updatedAt);
  el.threadList.replaceChildren();

  if (sorted.length === 0) {
    const empty = document.createElement("li");
    empty.className = "thread-empty";
    empty.textContent = "Todavia no hay chats.";
    el.threadList.appendChild(empty);
    return;
  }

  for (const thread of sorted) {
    const item = document.createElement("li");
    item.className = `thread-item${thread.id === threadId ? " active" : ""}`;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "thread-select";
    button.textContent = thread.title;
    button.title = thread.title;
    button.addEventListener("click", () => selectThread(thread.id));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "thread-delete";
    remove.textContent = "×";
    remove.title = "Eliminar chat";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteThread(thread.id);
    });

    item.appendChild(button);
    item.appendChild(remove);
    el.threadList.appendChild(item);
  }
}

function renderMessages(messages) {
  el.messages.replaceChildren();
  for (const msg of messages) {
    addMessage(msg.text, msg.role);
  }
}

async function fetchHistory(id) {
  try {
    const res = await fetch(
      `${API_BASE}/chat/${encodeURIComponent(id)}/history`,
    );
    if (res.status === 404) return { ok: false, lost: true };
    if (!res.ok) return { ok: false, lost: false };
    const data = await res.json();
    return {
      ok: true,
      history: Array.isArray(data.history) ? data.history : [],
    };
  } catch {
    return { ok: false, lost: false };
  }
}

async function selectThread(id) {
  if (id === threadId) return;
  const thread = findThread(threads, id);
  if (!thread) return;

  threadId = id;
  el.threadId.textContent = id;
  renderMessages(thread.messages);
  renderThreadList();
  closeSidebarOnMobile();
  el.input.focus();

  const result = await fetchHistory(id);
  if (threadId !== id) return;

  if (result.ok) {
    applyHistory(id, result.history, thread.title);
  } else if (result.lost) {
    addMessage(MEMORY_LOST_NOTICE, "info");
  }
}

function deleteThread(id) {
  if (!confirm("Eliminar este chat?")) return;
  threads = threads.filter((t) => t.id !== id);
  saveThreads(threads);
  if (id === threadId) {
    startNewChat();
  } else {
    renderThreadList();
  }
}

function startNewChat() {
  threadId = null;
  el.threadId.textContent = "—";
  el.messages.replaceChildren();
  addMessage(GREETING, "agent");
  renderThreadList();
  closeSidebarOnMobile();
  el.input.focus();
}

function closeSidebarOnMobile() {
  if (window.matchMedia("(max-width: 780px)").matches) {
    el.sidebar.classList.remove("open");
  }
}

function addMessage(text, role) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  el.messages.appendChild(wrap);
  el.messages.scrollTop = el.messages.scrollHeight;
  return wrap;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg agent";
  wrap.innerHTML =
    '<div class="bubble typing"><span></span><span></span><span></span></div>';
  el.messages.appendChild(wrap);
  el.messages.scrollTop = el.messages.scrollHeight;
  return wrap;
}

function setBusy(value) {
  busy = value;
  el.send.disabled = value;
  el.input.disabled = value;
}

async function checkHealth() {
  el.dot.className = "dot";
  try {
    const res = await fetch(`${API_BASE}/health`);
    el.dot.className = res.ok ? "dot ok" : "dot bad";
    el.dot.title = res.ok
      ? "backend conectado"
      : `backend respondio ${res.status}`;
  } catch {
    el.dot.className = "dot bad";
    el.dot.title = "backend no alcanzable";
  }
}

async function sendMessage(text) {
  const isFirstMessage = !threadId;
  addMessage(text, "user");
  const typing = addTyping();
  setBusy(true);

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, thread_id: threadId }),
    });

    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`HTTP ${res.status}: ${detail.slice(0, 300)}`);
    }

    const data = await res.json();
    typing.remove();
    threadId = data.thread_id;
    el.threadId.textContent = threadId;

    if (Array.isArray(data.history) && data.history.length > 0) {
      applyHistory(threadId, data.history, titleFrom(text));
    } else {
      if (isFirstMessage) {
        threads.unshift({
          id: threadId,
          title: titleFrom(text),
          updatedAt: Date.now(),
          messages: [],
        });
        threads = threads.slice(0, MAX_THREADS);
      }
      addMessage(data.reply || "(respuesta vacia)", "agent");
      persistFromDom();
    }
    el.dot.className = "dot ok";
  } catch (err) {
    typing.remove();
    addMessage(`Error: ${err.message}`, "error");
    el.dot.className = "dot bad";
    if (threadId) persistFromDom();
  } finally {
    setBusy(false);
    el.input.focus();
  }
}

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = el.input.value.trim();
  if (!text || busy) return;
  el.input.value = "";
  el.input.style.height = "auto";
  sendMessage(text);
});

el.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.form.requestSubmit();
  }
});

el.input.addEventListener("input", () => {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 160)}px`;
});

el.suggestions.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button || busy) return;
  sendMessage(button.textContent.trim());
});

el.newChat.addEventListener("click", startNewChat);

el.sidebarToggle.addEventListener("click", () => {
  el.sidebar.classList.toggle("open");
});

renderThreadList();
checkHealth();
el.input.focus();
