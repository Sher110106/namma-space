// Minimal PNG reader for 8-bit grey (or RGB/RGBA, first channel) images, no packages. Used to load grid.png in Node tests.
import fs from "node:fs";
import zlib from "node:zlib";

export function readPng(file) {
  const buf = fs.readFileSync(file);
  let off = 8, w = 0, h = 0, depth = 0, ctype = 0; const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off), type = buf.toString("latin1", off + 4, off + 8), body = buf.subarray(off + 8, off + 8 + len);
    if (type === "IHDR") { w = body.readUInt32BE(0); h = body.readUInt32BE(4); depth = body[8]; ctype = body[9]; if (body[12]) throw new Error("interlaced PNG"); }
    else if (type === "IDAT") idat.push(body);
    else if (type === "IEND") break;
    off += 12 + len;
  }
  if (depth !== 8 || ![0, 2, 6].includes(ctype)) throw new Error(`unsupported PNG depth ${depth} type ${ctype}`);
  const bpp = ctype === 0 ? 1 : ctype === 2 ? 3 : 4, stride = w * bpp;
  const raw = zlib.inflateSync(Buffer.concat(idat)), out = new Uint8Array(w * h * bpp);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)], src = y * (stride + 1) + 1, dst = y * stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? out[dst + x - bpp] : 0, b = y ? out[dst - stride + x] : 0, c = x >= bpp && y ? out[dst - stride + x - bpp] : 0;
      let v = raw[src + x];
      if (f === 1) v += a; else if (f === 2) v += b; else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      out[dst + x] = v & 255;
    }
  }
  const grey = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) grey[i] = out[i * bpp];
  return { width: w, height: h, data: grey };
}
