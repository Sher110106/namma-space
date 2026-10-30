// Recording helpers: serve a bundle, drive headless Chrome over CDP, pipe screencast frames straight into ffmpeg.
// No frame files are written. Reuses the viewer's own CDP driver (viewer/test/lib), unmodified.
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "../../viewer/test/lib/server.mjs";
import { launchChrome, sleep } from "../../viewer/test/lib/chrome.mjs";

export { sleep };
export const HERE = path.dirname(fileURLToPath(import.meta.url));

/** One screencast recording: frames (JPEG) arrive with arrival times and go to ffmpeg, which makes 30 fps video. */
export class Recorder {
  constructor(pg, outFile, { w, h, outW = 1920, outH = 1080, fps = 30, quality = 88, crf = 17 } = {}) {
    Object.assign(this, { pg, outFile, w, h, outW, outH, fps, quality, crf, frames: 0, last: null, t0: null, tLast: null, gaps: [] });
  }
  async start() {
    const vf = `fps=${this.fps},scale=${this.outW}:${this.outH}:flags=bicubic,format=yuv420p`;
    this.ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-use_wallclock_as_timestamps", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-i", "-",
      "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-crf", String(this.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", this.outFile],
      { stdio: ["pipe", "inherit", "inherit"] });
    this.ffDone = new Promise((res) => this.ff.on("close", res));
    this.ff.stdin.on("error", () => {});
    this.pg.on("Page.screencastFrame", (p) => {
      const now = Date.now();
      const buf = Buffer.from(p.data, "base64");
      if (this.t0 === null) this.t0 = now;
      if (this.tLast !== null && now - this.tLast > 100) this.gaps.push(now - this.tLast);
      this.tLast = now; this.last = buf; this.frames++;
      if (!this.ff.stdin.destroyed) this.ff.stdin.write(buf);
      this.pg.send("Page.screencastFrameAck", { sessionId: p.sessionId }).catch(() => {});
    });
    await this.pg.send("Page.startScreencast", { format: "jpeg", quality: this.quality, maxWidth: this.w, maxHeight: this.h, everyNthFrame: 1 });
    this.tStart = Date.now();
    return this;
  }
  /** Seconds since start() */
  get t() { return (Date.now() - this.tStart) / 1000; }
  async stop() {
    await this.pg.send("Page.stopScreencast").catch(() => {});
    if (this.last && !this.ff.stdin.destroyed) this.ff.stdin.write(this.last);   // repeat the last frame so the clip runs to the stop time
    this.ff.stdin.end();
    await this.ffDone;
    return { frames: this.frames, seconds: (Date.now() - this.tStart) / 1000, gaps_over_100ms: this.gaps.length, worst_gap_ms: Math.max(0, ...this.gaps) };
  }
}

export async function openBundle(web, { w = 1280, h = 720, dpr = 1, extraArgs = [] } = {}) {
  const srv = await startServer([{ prefix: "/", dir: web }]);
  const br = await launchChrome({ width: w, height: h, extraArgs });
  const pg = await br.newPage({ width: w, height: h, dpr });
  return { srv, br, pg, url: (q = "") => `${srv.origin}/index.html${q}`, async close() { await pg.close(); await br.close(); await srv.close(); } };
}
