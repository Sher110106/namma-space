// Test environment: serves a web folder, launches a fresh Chrome per test, and exposes helpers.
import fs from "node:fs";
import path from "node:path";
import { startServer } from "./server.mjs";
import { launchChrome, sleep } from "./chrome.mjs";

export { sleep };

export async function withBrowser(web, fn, { server = {}, chrome = {}, page = {}, pageUrl = "/index.html" } = {}) {
  const { extraMounts = [], ...serverOpts } = server;
  const srv = await startServer([...extraMounts, { prefix: "/", dir: web }], serverOpts);
  const br = await launchChrome(chrome);
  const pg = await br.newPage(page);
  try {
    return await fn({ srv, br, pg, url: (q = "") => `${srv.origin}${pageUrl}${q}` });
  } finally {
    await pg.close(); await br.close(); await srv.close();
  }
}

/** Open the viewer and wait for the first splat frame (or an error). */
export async function openViewer(ctx, query = "", { timeout = 60000, start = false } = {}) {
  await ctx.pg.goto(ctx.url(query));
  await ctx.pg.waitFor("window.__ns && (window.__ns.ready || window.__ns.error)", timeout, 100);
  if (start) { await ctx.pg.eval("document.getElementById('s-go').click()"); await sleep(300); }
  return ctx.pg.eval("({ff: __ns.firstFrameMs, err: __ns.error})");
}

export async function waitFullLoad(pg, timeout = 60000) { await pg.waitFor("__ns.S.fullLoaded === true", timeout, 100); }

export function sceneJson(web, name) {
  const idx = JSON.parse(fs.readFileSync(path.join(web, "scenes/index.json"), "utf8"));
  const n = name || idx.default;
  return { name: n, dir: path.join(web, "scenes", n), scene: JSON.parse(fs.readFileSync(path.join(web, "scenes", n, "scene.json"), "utf8")) };
}

export function consoleErrors(pg) {
  return [...pg.console.filter((c) => c.type === "error" || c.type === "log-error").map((c) => `${c.type}: ${c.text}`), ...pg.exceptions.map((e) => `exception: ${e}`)];
}
