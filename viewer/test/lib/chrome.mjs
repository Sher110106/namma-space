// Minimal headless-Chrome driver over the DevTools protocol (Node's built-in WebSocket, no packages).
// Chrome: Playwright's cached chrome-headless-shell with Metal ANGLE, so WebGL reports a real GPU.
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export function chromePath() {
  if (process.env.NS_CHROME) return process.env.NS_CHROME;
  const base = path.join(os.homedir(), "Library/Caches/ms-playwright");
  const dirs = fs.readdirSync(base).filter((d) => d.startsWith("chromium_headless_shell-")).sort().reverse();
  for (const d of dirs) {
    const p = path.join(base, d, "chrome-headless-shell-mac-arm64", "chrome-headless-shell");
    if (fs.existsSync(p)) return p;
  }
  throw new Error("chrome-headless-shell not found; set NS_CHROME");
}

export async function launchChrome({ width = 1280, height = 720, extraArgs = [] } = {}) {
  const userDir = fs.mkdtempSync(path.join(os.tmpdir(), "ns-chrome-"));
  const args = ["--remote-debugging-port=0", `--user-data-dir=${userDir}`, "--use-gl=angle", "--use-angle=metal",
    "--ignore-gpu-blocklist", "--enable-gpu", "--no-sandbox", "--hide-scrollbars", "--mute-audio",
    "--no-first-run", "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
    `--window-size=${width},${height}`, ...extraArgs, "about:blank"];
  const proc = spawn(chromePath(), args, { stdio: ["ignore", "ignore", "pipe"] });
  const port = await new Promise((resolve, reject) => {
    let buf = "";
    const to = setTimeout(() => reject(new Error("chrome did not start: " + buf)), 15000);
    proc.stderr.on("data", (d) => {
      buf += d;
      const m = /DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)\//.exec(buf);
      if (m) { clearTimeout(to); resolve(+m[1]); }
    });
    proc.on("exit", (c) => reject(new Error("chrome exited " + c + " " + buf)));
  });
  const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  return {
    port, version: version.Browser, proc,
    async newPage(opts = {}) { return Page.open(port, { width, height, ...opts }); },
    async close() { try { proc.kill("SIGKILL"); } catch {} await new Promise((r) => setTimeout(r, 100)); fs.rmSync(userDir, { recursive: true, force: true }); },
  };
}

export class Page {
  static async open(port, { width = 1280, height = 720, dpr = 1, mobile = false, touch = false, userAgent = null } = {}) {
    const t = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
    const p = new Page(t.webSocketDebuggerUrl, t.id, port);
    await p.ready;
    await p.send("Page.enable"); await p.send("Runtime.enable"); await p.send("Log.enable"); await p.send("Network.enable");
    await p.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: dpr, mobile });
    if (touch) await p.send("Emulation.setTouchEmulationEnabled", { enabled: true, maxTouchPoints: 5 });
    if (userAgent) await p.send("Network.setUserAgentOverride", { userAgent });
    return p;
  }
  constructor(wsUrl, id, port) {
    this.id = id; this.port = port; this.ws = new WebSocket(wsUrl); this.n = 0; this.pending = new Map();
    this.console = []; this.exceptions = []; this.requests = []; this.failures = []; this.handlers = new Map();
    this.ready = new Promise((r) => (this.ws.onopen = r));
    this.ws.onmessage = (e) => this.onMessage(JSON.parse(e.data));
  }
  onMessage(m) {
    if (m.id && this.pending.has(m.id)) { const { res, rej } = this.pending.get(m.id); this.pending.delete(m.id); return m.error ? rej(new Error(m.error.message)) : res(m.result); }
    switch (m.method) {
      case "Runtime.consoleAPICalled":
        this.console.push({ type: m.params.type, text: m.params.args.map((a) => a.value ?? a.description ?? "").join(" ") }); break;
      case "Runtime.exceptionThrown":
        this.exceptions.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text); break;
      case "Log.entryAdded":
        if (m.params.entry.level === "error") this.console.push({ type: "log-error", text: `${m.params.entry.text} ${m.params.entry.url || ""}` }); break;
      case "Network.requestWillBeSent": this.requests.push({ url: m.params.request.url, id: m.params.requestId, t: m.params.timestamp }); break;
      case "Network.loadingFailed": this.failures.push({ id: m.params.requestId, error: m.params.errorText }); break;
    }
    const h = this.handlers.get(m.method); if (h) h(m.params);
  }
  on(method, fn) { this.handlers.set(method, fn); }
  send(method, params = {}) { return new Promise((res, rej) => { const id = ++this.n; this.pending.set(id, { res, rej }); this.ws.send(JSON.stringify({ id, method, params })); }); }
  async goto(url) { this.requests.length = 0; this.failures.length = 0; this.console.length = 0; this.exceptions.length = 0; await this.send("Page.navigate", { url }); }
  async eval(expr, { await: aw = true } = {}) {
    const r = await this.send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: aw });
    if (r.exceptionDetails) throw new Error("eval failed: " + (r.exceptionDetails.exception?.description || r.exceptionDetails.text) + " in " + expr.slice(0, 200));
    return r.result.value;
  }
  async waitFor(expr, timeoutMs = 30000, every = 100) {
    const t0 = Date.now();
    for (;;) {
      let v = null; try { v = await this.eval(expr); } catch { v = null; }
      if (v) return v;
      if (Date.now() - t0 > timeoutMs) throw new Error("timeout waiting for: " + expr);
      await new Promise((r) => setTimeout(r, every));
    }
  }
  async screenshot(file, clip) {
    const r = await this.send("Page.captureScreenshot", { format: "png", ...(clip ? { clip: { ...clip, scale: 1 } } : {}) });
    fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, Buffer.from(r.data, "base64")); return file;
  }
  async key(code, down = true) {
    const map = { KeyW: ["w", 87], KeyA: ["a", 65], KeyS: ["s", 83], KeyD: ["d", 68], ArrowUp: ["ArrowUp", 38], ArrowDown: ["ArrowDown", 40],
      ArrowLeft: ["ArrowLeft", 37], ArrowRight: ["ArrowRight", 39], ShiftLeft: ["Shift", 16], Escape: ["Escape", 27], KeyR: ["r", 82],
      KeyH: ["h", 72], KeyM: ["m", 77], KeyT: ["t", 84], KeyC: ["c", 67], Digit1: ["1", 49], Digit2: ["2", 50], Slash: ["?", 191], KeyF: ["f", 70] };
    const [key, vk] = map[code] || [code, 0];
    await this.send("Input.dispatchKeyEvent", { type: down ? "rawKeyDown" : "keyUp", code, key, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk });
  }
  async tap(code, ms = 60) { await this.key(code, true); await new Promise((r) => setTimeout(r, ms)); await this.key(code, false); }
  async mouse(type, x, y, extra = {}) { await this.send("Input.dispatchMouseEvent", { type, x, y, button: "left", buttons: type === "mouseReleased" ? 0 : 1, clickCount: 1, ...extra }); }
  async click(x, y) { await this.mouse("mouseMoved", x, y, { buttons: 0 }); await this.mouse("mousePressed", x, y); await this.mouse("mouseReleased", x, y); }
  async touch(type, points) { await this.send("Input.dispatchTouchEvent", { type, touchPoints: points.map((p, i) => ({ x: p[0], y: p[1], id: i })) }); }
  async throttle(bytesPerSec) { await this.send("Network.emulateNetworkConditions", { offline: false, latency: 20, downloadThroughput: bytesPerSec, uploadThroughput: bytesPerSec }); }
  async close() { try { await fetch(`http://127.0.0.1:${this.port}/json/close/${this.id}`); } catch {} try { this.ws.close(); } catch {} }
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
