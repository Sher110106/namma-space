// PLY -> SPZ with Spark's own writer, run in headless Chrome (Spark needs a Worker).
// usage: node spz_transcode.mjs <in.ply> <out.spz> [maxSh=1] [version=3] [fractionalBits=12]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "../../test/lib/server.mjs";
import { launchChrome, sleep } from "../../test/lib/chrome.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const [inPly, outSpz, maxSh = "1", version = "3", fracBits = "12"] = process.argv.slice(2);
const tmp = fs.mkdtempSync("/tmp/ns-spz-");
fs.writeFileSync(path.join(tmp, "index.html"), `<!doctype html><meta charset=utf-8>
<script type="importmap">{"imports":{"three":"/vendor/three.module.js","three/addons/":"/vendor/addons/","@sparkjsdev/spark":"/vendor/spark.module.js"}}</script>
<script type="module">
import { transcodeSpz } from "@sparkjsdev/spark";
try {
  const bytes = new Uint8Array(await (await fetch("/in.ply")).arrayBuffer());
  const r = await transcodeSpz({ inputs: [{ fileBytes: bytes, fileType: "ply" }], maxSh: ${+maxSh}, version: ${+version}, fractionalBits: ${+fracBits} });
  window.__out = new Uint8Array(r.fileBytes); window.__done = true;
} catch (e) { window.__err = String(e && e.stack || e); }
</script>`);
fs.copyFileSync(inPly, path.join(tmp, "in.ply"));
const srv = await startServer([{ prefix: "/vendor/", dir: path.join(here, "../../vendor") }, { prefix: "/", dir: tmp }]);
const chrome = await launchChrome();
const page = await chrome.newPage();
await page.goto(srv.origin + "/index.html");
try {
  await page.waitFor("window.__done || window.__err", 300000, 500);
  const err = await page.eval("window.__err || null");
  if (err) throw new Error(err);
  const n = await page.eval("window.__out.length");
  const chunks = []; const step = 3 * 1024 * 1024;
  for (let o = 0; o < n; o += step) {
    const b64 = await page.eval(`(()=>{const a=window.__out.subarray(${o},${Math.min(n, o + step)});let s="";for(let i=0;i<a.length;i+=32768)s+=String.fromCharCode.apply(null,a.subarray(i,i+32768));return btoa(s)})()`);
    chunks.push(Buffer.from(b64, "base64"));
  }
  fs.writeFileSync(outSpz, Buffer.concat(chunks));
  console.log("wrote", outSpz, n, "bytes");
} finally { await page.close(); await chrome.close(); await srv.close(); fs.rmSync(tmp, { recursive: true, force: true }); }
process.exit(0);
