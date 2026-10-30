// Static file server for tests: Range support, optional CORS, several mount points, request log.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".wasm": "application/wasm", ".txt": "text/plain",
};

/**
 * mounts: [{prefix: "/vendor/", dir: "/abs/dir"}, ...] checked in order; the last entry should be {prefix: "/", dir}.
 * opts.cors: send access-control-allow-origin: * (and expose range headers).
 */
export function startServer(mounts, opts = {}) {
  const log = [];
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, "http://x");
    let p = decodeURIComponent(url.pathname);
    log.push({ path: p, range: req.headers.range || null, t: Date.now() });
    let file = null;
    for (const m of mounts) {
      if (p.startsWith(m.prefix)) {
        const rel = p.slice(m.prefix.length);
        const cand = path.resolve(m.dir, rel);
        if (!cand.startsWith(path.resolve(m.dir))) continue;
        if (fs.existsSync(cand) && fs.statSync(cand).isFile()) { file = cand; break; }
        if (fs.existsSync(cand) && fs.statSync(cand).isDirectory() && fs.existsSync(path.join(cand, "index.html"))) { file = path.join(cand, "index.html"); break; }
      }
    }
    const headers = { "cache-control": "no-store", "accept-ranges": "bytes" };
    if (opts.cors) { headers["access-control-allow-origin"] = "*"; headers["access-control-expose-headers"] = "content-length, content-range, accept-ranges"; }
    if (req.method === "OPTIONS") { res.writeHead(204, { ...headers, "access-control-allow-headers": "range" }); return res.end(); }
    if (!file) { res.writeHead(404, headers); return res.end("not found"); }
    const st = fs.statSync(file);
    headers["content-type"] = MIME[path.extname(file).toLowerCase()] || "application/octet-stream";
    const rng = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || "");
    let start = 0, end = st.size - 1, code = 200;
    if (rng) {
      if (rng[1] !== "") start = +rng[1];
      if (rng[2] !== "") end = Math.min(end, +rng[2]);
      if (rng[1] === "" && rng[2] !== "") { start = Math.max(0, st.size - +rng[2]); end = st.size - 1; }
      code = 206; headers["content-range"] = `bytes ${start}-${end}/${st.size}`;
    }
    headers["content-length"] = end - start + 1;
    res.writeHead(code, headers);
    if (req.method === "HEAD") return res.end();
    const stream = fs.createReadStream(file, { start, end, highWaterMark: 64 * 1024 });
    if (!opts.bytesPerSec || st.size < 1_000_000) return stream.pipe(res);
    // throttle: send in 64 KB chunks paced to bytesPerSec (small files pass at full speed)
    const t0 = Date.now(); let sent = 0;
    stream.on("data", (chunk) => {
      sent += chunk.length; stream.pause();
      const due = t0 + (sent / opts.bytesPerSec) * 1000;
      res.write(chunk, () => setTimeout(() => stream.resume(), Math.max(0, due - Date.now())));
    });
    stream.on("end", () => res.end());
  });
  return new Promise((resolve) => {
    server.listen(opts.port || 0, "127.0.0.1", () => {
      const port = server.address().port;
      resolve({ port, origin: `http://127.0.0.1:${port}`, log, close: () => new Promise((r) => server.close(() => r())) });
    });
  });
}
