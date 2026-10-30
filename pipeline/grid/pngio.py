"""Minimal 8-bit PNG writer and reader (grey or RGB), standard library plus numpy.

The writer uses filter type 0 and zlib level 9. The reader handles all five filter
types, so it can also read PNGs made by other tools (tests only).
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_png(path: str | Path, arr: np.ndarray) -> None:
    """Write a uint8 array of shape (H, W) as grey or (H, W, 3) as RGB."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim == 2:
        ctype, h, w = 0, arr.shape[0], arr.shape[1]
        rows = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        ctype, h, w = 2, arr.shape[0], arr.shape[1]
        rows = arr.reshape(h, w * 3)
    else:
        raise ValueError(f"unsupported shape {arr.shape}")
    raw = np.zeros((h, rows.shape[1] + 1), dtype=np.uint8)
    raw[:, 1:] = rows  # filter type 0 in column 0
    ihdr = struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0)
    out = _SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 9)) + _chunk(b"IEND", b"")
    Path(path).write_bytes(out)


def read_png(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    if data[:8] != _SIG:
        raise ValueError("not a PNG")
    pos, idat, ihdr = 8, [], None
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    if ihdr is None:
        raise ValueError("no IHDR")
    w, h, depth, ctype, _, _, interlace = ihdr
    if depth != 8 or ctype not in (0, 2) or interlace != 0:
        raise ValueError("only 8-bit non-interlaced grey/RGB PNG supported")
    bpp = 1 if ctype == 0 else 3
    stride = w * bpp
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8).reshape(h, stride + 1)
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    for i in range(h):
        ft = raw[i, 0]
        line = raw[i, 1:].astype(np.int32)
        if ft == 0:
            cur = line
        elif ft == 1:
            cur = line.copy()
            for k in range(bpp, stride):
                cur[k] = (cur[k] + cur[k - bpp]) & 255
        elif ft == 2:
            cur = (line + prev) & 255
        elif ft == 3:
            cur = line.copy()
            for k in range(stride):
                left = cur[k - bpp] if k >= bpp else 0
                cur[k] = (cur[k] + ((left + prev[k]) >> 1)) & 255
        elif ft == 4:
            cur = line.copy()
            for k in range(stride):
                a = cur[k - bpp] if k >= bpp else 0
                b = prev[k]
                c = prev[k - bpp] if k >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[k] = (cur[k] + pred) & 255
        else:
            raise ValueError(f"bad filter {ft}")
        out[i] = cur
        prev = cur
    return out.reshape(h, w) if ctype == 0 else out.reshape(h, w, 3)
