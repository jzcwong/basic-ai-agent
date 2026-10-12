"""
Web UI for the Falcon agent: chat with the falcon-mcp backed agent in a browser.

Reuses MCPBridge and Agent from falcon-agent.py and adds a small HTTP server
(standard library only) that streams the agent's text and tool calls to the page.

Usage:
  python falcon-agent-web.py                       # UI at http://127.0.0.1:8080, falcon-mcp at http://127.0.0.1:8000/mcp
  python falcon-agent-web.py --port 9000
  python falcon-agent-web.py --transport stdio     # agent launches falcon-mcp via uvx

The UI has no login and acts with your Falcon API credentials, so it only
listens on 127.0.0.1.
"""

import argparse
import importlib.util
import json
import os
import secrets
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

# falcon-agent.py has a hyphen in its name, so load it by path instead of `import`
_spec = importlib.util.spec_from_file_location(
    "falcon_agent", Path(__file__).with_name("falcon-agent.py")
)
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)

TOOL_PREVIEW_CHARS = 5_000  # how much of each tool result is shown in the browser
APPROVAL_TIMEOUT = 300      # seconds to wait for Approve/Deny before treating it as denied


# ---------------------------------------------------------------------------
# Agent that reports progress as events instead of printing
# ---------------------------------------------------------------------------

class WebAgent(fa.Agent):
    def __init__(self, mcp: fa.MCPBridge):
        super().__init__(mcp)
        self.lock = threading.Lock()  # one response at a time per conversation
        self._emit = None
        self._active_tool_id = None
        self._pending: dict[str, dict] = {}  # approval id -> {"event", "approved"}

    def _close_dangling_tool_use(self):
        # A response cut short (Stop button, closed tab) can leave tool_use blocks
        # without results, which the API rejects; answer them so history stays valid
        if not self.messages or self.messages[-1]["role"] != "assistant":
            return
        ids = [b.id for b in self.messages[-1]["content"] if b.type == "tool_use"]
        if ids:
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": i, "is_error": True,
                 "content": "Cancelled: the user stopped this response."}
                for i in ids
            ]})

    def run_web(self, user_input: str, emit):
        self._emit = emit
        self._close_dangling_tool_use()
        self._append_user_text(user_input)
        self._trim_history()

        for _ in range(self.max_iterations):
            try:
                with fa.client.messages.stream(
                    model=fa.MODEL,
                    max_tokens=fa.MAX_TOKENS,
                    system=fa.build_system_prompt(),
                    tools=self.tools,
                    messages=self.messages,
                ) as stream:
                    for text_delta in stream.text_stream:
                        emit({"type": "text", "text": text_delta})
                    final_message = stream.get_final_message()
            except anthropic.APIError as e:
                emit({"type": "error", "message": f"API request failed: {e}"})
                return

            self.messages.append({"role": "assistant", "content": final_message.content})

            if final_message.stop_reason == "max_tokens":
                emit({"type": "warning", "message": "Response hit max_tokens and may be incomplete."})

            # Execute every tool_use block present so each one gets a matching tool_result
            tool_uses = [b for b in final_message.content if b.type == "tool_use"]
            if not tool_uses:
                return

            tool_results = []
            for block in tool_uses:
                emit({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
                self._active_tool_id = block.id
                content, is_error = self._execute(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                })
                preview = content[:TOOL_PREVIEW_CHARS]
                if len(content) > TOOL_PREVIEW_CHARS:
                    preview += f"\n...[{len(content) - TOOL_PREVIEW_CHARS} more characters not shown]"
                emit({"type": "tool_result", "id": block.id, "content": preview, "is_error": is_error})
            self.messages.append({"role": "user", "content": tool_results})

        emit({"type": "error", "message": "Stopped after max iterations without a final answer."})

    # ---- approval in the browser -------------------------------------------

    def _approve(self, name: str, args: dict) -> bool:
        approval_id = uuid.uuid4().hex
        decision = {"event": threading.Event(), "approved": False}
        self._pending[approval_id] = decision
        try:
            self._emit({"type": "approval", "id": approval_id, "tool_id": self._active_tool_id})
            decision["event"].wait(APPROVAL_TIMEOUT)
            return decision["approved"]
        finally:
            self._pending.pop(approval_id, None)

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        decision = self._pending.get(approval_id)
        if decision is None:
            return False
        decision["approved"] = approved
        decision["event"].set()
        return True

    def deny_pending(self):
        for decision in list(self._pending.values()):
            decision["event"].set()


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class AgentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, mcp: fa.MCPBridge, info: dict):
        super().__init__(("127.0.0.1", port), Handler)
        self.mcp = mcp
        self.info = info
        self.agent = WebAgent(mcp)
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.allowed_origins = {f"http://{h}" for h in self.allowed_hosts}

    def new_conversation(self):
        # Swap in a fresh agent; a response still finishing on the old one can't touch it
        old, self.agent = self.agent, WebAgent(self.mcp)
        old.deny_pending()


class Handler(BaseHTTPRequestHandler):
    server: AgentServer

    def log_message(self, format, *args):
        pass  # keep the terminal quiet

    def _request_allowed(self) -> bool:
        # Checking Host and Origin stops other websites (including via DNS rebinding)
        # from driving the agent through your browser
        if self.headers.get("Host") not in self.server.allowed_hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in self.server.allowed_origins

    def _send(self, status: int, body: bytes, content_type: str, extra_headers=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in extra_headers:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, obj):
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _read_json(self) -> dict | None:
        # Requiring application/json also forces a CORS preflight for cross-site requests
        if self.headers.get_content_type() != "application/json":
            return None
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def do_GET(self):
        if not self._request_allowed():
            return self._send_json(403, {"error": "Forbidden"})
        if self.path == "/":
            nonce = secrets.token_urlsafe(16)
            csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
                   "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            body = PAGE.replace("__NONCE__", nonce).encode()
            return self._send(200, body, "text/html; charset=utf-8",
                              [("Content-Security-Policy", csp), ("Cache-Control", "no-store")])
        if self.path == "/api/info":
            return self._send_json(200, self.server.info)
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self._request_allowed():
            return self._send_json(403, {"error": "Forbidden"})
        data = self._read_json()
        if data is None:
            return self._send_json(400, {"error": "Expected a JSON object body"})

        if self.path == "/api/chat":
            return self._chat(data)
        if self.path == "/api/approve":
            ok = self.server.agent.resolve_approval(str(data.get("id")), data.get("approved") is True)
            return self._send_json(200 if ok else 404, {"ok": ok})
        if self.path == "/api/stop":
            self.server.agent.deny_pending()
            return self._send_json(200, {"ok": True})
        if self.path == "/api/reset":
            self.server.new_conversation()
            return self._send_json(200, {"ok": True})
        self._send_json(404, {"error": "Not found"})

    def _chat(self, data: dict):
        message = data.get("message")
        if not isinstance(message, str) or not message.strip():
            return self._send_json(400, {"error": "Message is empty"})

        agent = self.server.agent
        if not agent.lock.acquire(blocking=False):
            return self._send_json(409, {"error": "The agent is still finishing the previous response. Try again shortly."})
        try:
            # Stream newline-delimited JSON events; the body ends when the connection closes
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

            def emit(event: dict):
                self.wfile.write((json.dumps(event, default=str) + "\n").encode())
                self.wfile.flush()

            try:
                agent.run_web(message.strip(), emit)
                emit({"type": "done"})
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser stopped or closed the tab; history is repaired on the next message
            except Exception as e:
                try:
                    emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
                except OSError:
                    pass
        finally:
            agent.lock.release()


# ---------------------------------------------------------------------------
# Page (single file: HTML, CSS and JS)
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Falcon Agent</title>
<style>
:root { --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --border:#2a2f3a; --text:#e6e8eb;
        --muted:#9aa3af; --accent:#e5323b; --ok:#3fb950; --err:#f85149; --warn:#d29922; }
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { background: var(--bg); color: var(--text); display: flex; flex-direction: column;
       font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
header { display: flex; align-items: center; gap: 14px; padding: 12px 20px;
         border-bottom: 1px solid var(--border); background: var(--panel); }
header h1 { font-size: 16px; margin: 0; font-weight: 600; }
header h1 span { color: var(--accent); }
#status { font-size: 12px; color: var(--muted); }
#status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
               background: var(--ok); margin-right: 6px; }
.spacer { flex: 1; }
button { font: inherit; font-size: 14px; cursor: pointer; border: 1px solid var(--border);
         background: var(--panel2); color: var(--text); border-radius: 8px; padding: 6px 12px; }
button:hover { border-color: #3d4554; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button:disabled { opacity: .5; cursor: default; }
main { flex: 1; overflow-y: auto; }
#chat { max-width: 900px; margin: 0 auto; padding: 24px 20px 16px; display: flex;
        flex-direction: column; gap: 14px; }
.msg.user { align-self: flex-end; background: #2b3445; padding: 10px 14px; max-width: 80%;
            border-radius: 14px 14px 4px 14px; white-space: pre-wrap; word-break: break-word; }
.md { overflow-wrap: anywhere; }
.md p { margin: 6px 0; }
.md ul, .md ol { margin: 6px 0; padding-left: 22px; }
.md h1, .md h2, .md h3, .md h4 { margin: 14px 0 6px; line-height: 1.3; }
.md h1 { font-size: 20px; } .md h2 { font-size: 18px; } .md h3 { font-size: 16px; } .md h4 { font-size: 15px; }
.md table { border-collapse: collapse; margin: 8px 0; font-size: 13px; display: block; overflow-x: auto; }
.md th, .md td { border: 1px solid var(--border); padding: 5px 9px; text-align: left; vertical-align: top; }
.md th { background: var(--panel2); }
.md code { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--panel2);
           padding: 1px 5px; border-radius: 4px; }
.md pre { background: #0a0c10; border: 1px solid var(--border); padding: 10px 12px;
          border-radius: 8px; overflow-x: auto; }
.md pre code { background: none; padding: 0; }
.md a { color: #58a6ff; }
.md hr { border: 0; border-top: 1px solid var(--border); }
.tool { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; font-size: 13px; }
.tool summary { padding: 8px 12px; cursor: pointer; display: flex; align-items: center; gap: 8px; list-style: none; }
.tool summary::-webkit-details-marker { display: none; }
.tool .name { font-family: ui-monospace, Menlo, monospace; }
.tool .state { margin-left: auto; color: var(--muted); }
.tool.ok .state, .tool.ok .icon { color: var(--ok); }
.tool.error .state, .tool.error .icon { color: var(--err); }
.tool.waiting .state { color: var(--warn); }
.tool .body { padding: 0 12px 10px; }
.tool .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; margin-top: 8px; }
.tool pre { margin: 4px 0 0; background: #0a0c10; border: 1px solid var(--border); border-radius: 6px;
            padding: 8px 10px; max-height: 320px; overflow: auto; white-space: pre-wrap;
            word-break: break-word; font: 12px ui-monospace, Menlo, monospace; }
.approval { display: flex; gap: 8px; align-items: center; padding: 0 12px 10px; color: var(--warn); }
.notice { font-size: 13px; padding: 8px 12px; border-radius: 8px; border: 1px solid; }
.notice.error { color: var(--err); border-color: #5c2b2b; background: #2a1515; }
.notice.warning { color: var(--warn); border-color: #5a4513; background: #2a2212; }
.thinking { color: var(--muted); font-size: 13px; display: flex; align-items: center; gap: 8px; }
.spinner { width: 12px; height: 12px; border: 2px solid var(--border); border-top-color: var(--accent);
           border-radius: 50%; animation: spin .8s linear infinite; display: inline-block; flex: none; }
@keyframes spin { to { transform: rotate(360deg); } }
#empty { text-align: center; color: var(--muted); margin-top: 8vh; }
#empty h2 { color: var(--text); font-weight: 600; margin-bottom: 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 20px; }
.chips button { border-radius: 999px; font-size: 13px; }
footer { border-top: 1px solid var(--border); background: var(--panel); padding: 12px 20px 8px; }
#form { max-width: 900px; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end; }
#input { flex: 1; resize: none; min-height: 44px; max-height: 200px; padding: 11px 14px; border-radius: 10px;
         border: 1px solid var(--border); background: var(--bg); color: var(--text); font: inherit; }
#input:focus { outline: none; border-color: #4a5263; }
#send { min-width: 76px; height: 44px; }
.hint { text-align: center; color: var(--muted); font-size: 11px; margin-top: 6px; }
#tools-panel { position: fixed; top: 0; right: 0; bottom: 0; width: 400px; max-width: 92vw; z-index: 10;
               background: var(--panel); border-left: 1px solid var(--border); padding: 16px;
               display: flex; flex-direction: column; gap: 10px; transform: translateX(100%); transition: transform .2s; }
#tools-panel.open { transform: none; }
#tools-panel .top { display: flex; align-items: center; justify-content: space-between; }
#tools-filter { padding: 7px 10px; border-radius: 8px; border: 1px solid var(--border);
                background: var(--bg); color: var(--text); font: inherit; font-size: 13px; }
#tools-list { overflow-y: auto; }
.tool-item { padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.tool-item .name { font-family: ui-monospace, Menlo, monospace; }
.tool-item .desc { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1><span>Falcon</span> Agent</h1>
  <div id="status">Connecting…</div>
  <div class="spacer"></div>
  <button id="tools-btn" type="button">Tools</button>
  <button id="reset-btn" type="button">New chat</button>
</header>

<main id="scroller">
  <div id="chat">
    <div id="empty">
      <h2>Ask about your Falcon environment</h2>
      <div>The agent calls Falcon tools for live data. Each tool call is shown as it runs; click one to see its input and result.</div>
      <div class="chips">
        <button type="button">Show high and critical detections from the last 24 hours</button>
        <button type="button">Summarize open incidents</button>
        <button type="button">List Windows hosts not seen in the last 7 days</button>
        <button type="button">What do we know about the threat actor FANCY BEAR?</button>
      </div>
    </div>
  </div>
</main>

<footer>
  <form id="form">
    <textarea id="input" rows="1" placeholder="Ask a Falcon question…" autofocus></textarea>
    <button id="send" class="primary" type="submit">Send</button>
  </form>
  <div class="hint">Enter to send · Shift+Enter for a new line</div>
</footer>

<aside id="tools-panel">
  <div class="top"><strong>Available tools</strong><button id="tools-close" type="button">Close</button></div>
  <input id="tools-filter" placeholder="Filter tools…">
  <div id="tools-list"></div>
</aside>

<script nonce="__NONCE__">
const $ = (id) => document.getElementById(id);
const chat = $("chat"), scroller = $("scroller"), input = $("input"), sendBtn = $("send");
const emptyState = $("empty");
const thinking = make("div", "thinking");
thinking.append(make("span", "spinner"), make("span", null, "Working…"));

let controller = null;  // AbortController for the response being streamed
let running = null;     // promise for the response being streamed
let toolCards = {};     // tool_use id -> card element
let tools = [];

function make(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function post(path, body, signal) {
  return fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(body), signal });
}

// ---- scrolling: follow the stream unless the user scrolled up to read ----
const nearBottom = () => scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
function follow(fn) {
  const stick = nearBottom();
  const result = fn();
  if (stick) scroller.scrollTop = scroller.scrollHeight;
  return result;
}
function add(e) {
  emptyState.remove();
  return follow(() => chat.insertBefore(e, thinking.isConnected ? thinking : null));
}

// ---- markdown: HTML is escaped first, so model or tool text can't inject markup ----
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const inline = (s) => esc(s)
  .replace(/`([^`]+)`/g, "<code>$1</code>")
  .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
  .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
const BULLET = /^\s*[-*+]\s+/, NUMBER = /^\s*\d+[.)]\s+/, HR = /^\s*(-{3,}|\*{3,})\s*$/;
const isRow = (l) => /^\s*\|.*\|\s*$/.test(l);
const isSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => inline(c.trim()));
const isTable = (lines, i) => isRow(lines[i]) && i + 1 < lines.length && isSep(lines[i + 1]);
const startsBlock = (lines, i) => {
  const l = lines[i];
  return /^```/.test(l) || /^#{1,6}\s/.test(l) || HR.test(l) || BULLET.test(l) || NUMBER.test(l) || isTable(lines, i);
};

function md(src) {
  const lines = src.split("\n");
  let html = "", i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^```/.test(l)) {
      const code = [];
      for (i++; i < lines.length && !/^```/.test(lines[i]); i++) code.push(lines[i]);
      i++;
      html += "<pre><code>" + esc(code.join("\n")) + "</code></pre>";
    } else if (/^#{1,6}\s/.test(l)) {
      const m = l.match(/^(#{1,6})\s+(.*)$/), n = Math.min(m[1].length, 4);
      html += `<h${n}>${inline(m[2])}</h${n}>`;
      i++;
    } else if (HR.test(l)) {
      html += "<hr>";
      i++;
    } else if (isTable(lines, i)) {
      html += "<table><thead><tr>" + cells(l).map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
      for (i += 2; i < lines.length && isRow(lines[i]); i++)
        html += "<tr>" + cells(lines[i]).map((c) => `<td>${c}</td>`).join("") + "</tr>";
      html += "</tbody></table>";
    } else if (BULLET.test(l) || NUMBER.test(l)) {
      const re = BULLET.test(l) ? BULLET : NUMBER;
      html += re === BULLET ? "<ul>" : `<ol start="${parseInt(l.trim(), 10)}">`;
      for (; i < lines.length && re.test(lines[i]); i++) html += `<li>${inline(lines[i].replace(re, ""))}</li>`;
      html += re === BULLET ? "</ul>" : "</ol>";
    } else if (!l.trim()) {
      i++;
    } else {
      const para = [];
      for (; i < lines.length && lines[i].trim() && !startsBlock(lines, i); i++) para.push(inline(lines[i]));
      html += `<p>${para.join("<br>")}</p>`;
    }
  }
  return html;
}

// ---- tool call cards ----
function toolCard(ev) {
  const card = make("details", "tool running");
  const summary = make("summary");
  summary.append(make("span", "spinner"), make("span", "name", ev.name), make("span", "state", "running"));
  const body = make("div", "body");
  body.append(make("div", "label", "Input"), make("pre", null, JSON.stringify(ev.input, null, 2)));
  card.append(summary, body);
  toolCards[ev.id] = card;
  add(card);
}

function setToolState(card, cls, state, icon) {
  card.className = "tool " + cls;
  card.querySelector(".state").textContent = state;
  if (icon) card.querySelector(".spinner")?.replaceWith(make("span", "icon", icon));
}

function finishTool(ev) {
  const card = toolCards[ev.id];
  if (!card) return;
  setToolState(card, ev.is_error ? "error" : "ok", ev.is_error ? "error" : "done", ev.is_error ? "✕" : "✓");
  card.querySelector(".approval")?.remove();
  follow(() => card.querySelector(".body").append(make("div", "label", "Result"), make("pre", null, ev.content)));
}

function askApproval(ev) {
  const card = toolCards[ev.tool_id];
  if (!card) return;
  setToolState(card, "waiting", "needs approval");
  card.open = true;
  const row = make("div", "approval");
  const yes = make("button", "primary", "Approve"), no = make("button", null, "Deny");
  const decide = (approved) => {
    row.replaceChildren(make("span", null, approved ? "Approved" : "Denied"));
    setToolState(card, "running", "running");
    post("/api/approve", { id: ev.id, approved });
  };
  yes.onclick = () => decide(true);
  no.onclick = () => decide(false);
  row.append(make("span", null, "Run this tool?"), yes, no);
  follow(() => card.append(row));
}

// ---- sending a message and reading the event stream ----
function setBusy(busy) {
  sendBtn.textContent = busy ? "Stop" : "Send";
  sendBtn.classList.toggle("primary", !busy);
}

async function send(text) {
  text = text.trim();
  if (controller || !text) return;
  emptyState.remove();
  chat.append(make("div", "msg user", text), thinking);
  scroller.scrollTop = scroller.scrollHeight;
  input.value = "";
  autosize();
  controller = new AbortController();
  setBusy(true);

  let current = null;  // agent text block currently receiving tokens
  const handle = (ev) => {
    if (ev.type === "text") {
      if (!current) { current = add(make("div", "msg agent md")); current.src = ""; }
      current.src += ev.text;
      follow(() => { current.innerHTML = md(current.src); });
    } else if (ev.type === "tool_use") {
      current = null;
      toolCard(ev);
    } else if (ev.type === "tool_result") {
      finishTool(ev);
    } else if (ev.type === "approval") {
      askApproval(ev);
    } else if (ev.type === "warning" || ev.type === "error") {
      current = null;
      add(make("div", "notice " + ev.type, ev.message));
    }
  };

  running = (async () => {
    try {
      const res = await post("/api/chat", { message: text }, controller.signal);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Request failed (HTTP ${res.status})`);
      }
      const reader = res.body.getReader(), decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          if (line.trim()) handle(JSON.parse(line));
        }
      }
    } catch (e) {
      add(make("div", e.name === "AbortError" ? "notice warning" : "notice error",
               e.name === "AbortError" ? "Stopped." : e.message));
    } finally {
      for (const card of Object.values(toolCards))
        if (/running|waiting/.test(card.className)) {
          setToolState(card, "error", "cancelled", "✕");
          card.querySelector(".approval")?.remove();
        }
      toolCards = {};
      thinking.remove();
      controller = null;
      setBusy(false);
      input.focus();
    }
  })();
  await running;
}

function stop() {
  controller?.abort();
  post("/api/stop", {});  // releases any approval the server is waiting on
}

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

$("form").onsubmit = (e) => { e.preventDefault(); controller ? stop() : send(input.value); };
input.oninput = autosize;
input.onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!controller) send(input.value);
  }
};
for (const chip of document.querySelectorAll(".chips button")) chip.onclick = () => send(chip.textContent);

$("reset-btn").onclick = async () => {
  if (controller) { stop(); await running; }
  await post("/api/reset", {});
  chat.replaceChildren(emptyState);
  input.focus();
};

// ---- header status and tools panel ----
function renderTools() {
  const q = $("tools-filter").value.toLowerCase();
  $("tools-list").replaceChildren(...tools
    .filter((t) => (t.name + " " + t.description).toLowerCase().includes(q))
    .map((t) => {
      const item = make("div", "tool-item");
      item.append(make("div", "name", t.name), make("div", "desc", t.description));
      return item;
    }));
}
$("tools-filter").oninput = renderTools;
$("tools-btn").onclick = () => $("tools-panel").classList.toggle("open");
$("tools-close").onclick = () => $("tools-panel").classList.remove("open");

fetch("/api/info").then((r) => r.json()).then((info) => {
  tools = info.tools;
  $("status").replaceChildren(make("span", "dot"),
    document.createTextNode(`falcon-mcp (${info.transport}) · ${tools.length} tools · ${info.model}`));
  renderTools();
}).catch(() => { $("status").textContent = "Could not reach the agent server"; });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Falcon agent web UI")
    p.add_argument("--port", type=int, default=int(os.environ.get("FALCON_AGENT_WEB_PORT", 8080)),
                   help="port for the web UI on 127.0.0.1 (default: 8080)")
    p.add_argument("--transport",
                   choices=["streamable-http", "sse", "stdio"],
                   default=os.environ.get("FALCON_MCP_TRANSPORT", "streamable-http"))
    p.add_argument("--url", default=os.environ.get("FALCON_MCP_URL"),
                   help="MCP endpoint (default: http://127.0.0.1:8000/mcp, or /sse for sse)")
    p.add_argument("--modules", default="detections,incidents,hosts,intel",
                   help="falcon-mcp modules (stdio transport only)")
    args = p.parse_args()
    if args.url is None:
        path = "/sse" if args.transport == "sse" else "/mcp"
        args.url = f"http://127.0.0.1:8000{path}"
    return args


if __name__ == "__main__":
    args = parse_args()

    target = "uvx falcon-mcp" if args.transport == "stdio" else args.url
    print(f"Connecting to falcon-mcp ({args.transport}) at {target} ...")
    try:
        mcp = fa.connect_falcon_mcp(args.transport, args.url, args.modules)
    except Exception as e:
        sys.exit(f"[error] Could not connect to falcon-mcp ({type(e).__name__}: {e})")

    if not mcp.tools:
        mcp.close()
        sys.exit("[error] falcon-mcp returned no tools; check --modules and API scopes.")

    info = {
        "model": fa.MODEL,
        "transport": args.transport,
        "tools": [
            {"name": t.name, "description": (t.description or "").strip().split("\n")[0][:300]}
            for t in mcp.tools
        ],
    }
    try:
        server = AgentServer(args.port, mcp, info)
    except OSError as e:
        mcp.close()
        sys.exit(f"[error] Could not start the web server on port {args.port} ({e})")

    print(f"Connected: {len(mcp.tools)} Falcon tools loaded.")
    print(f"Open http://127.0.0.1:{args.port} in your browser (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        server.server_close()
        mcp.close()
