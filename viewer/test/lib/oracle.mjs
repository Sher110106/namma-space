// Independent shortest path on a walkable grid: Dijkstra, 8-connected, no corner cutting, no wall penalty. Used to judge route length.
/** Shortest path length in metres between two cell indices (row * width + col); Infinity when none. */
export function oracle(g, s, tIdx) {
  const { w, h } = g, N = w * h, dist = new Float64Array(N).fill(Infinity), done = new Uint8Array(N);
  const walk = (i, j) => i >= 0 && j >= 0 && i < h && j < w && g.data[i * w + j] === 255;
  dist[s] = 0;
  const heap = [[0, s]];
  const push = (e) => { heap.push(e); let i = heap.length - 1; while (i) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { let c = 2 * i + 1; if (c >= heap.length) break; if (c + 1 < heap.length && heap[c + 1][0] < heap[c][0]) c++; if (heap[c][0] >= heap[i][0]) break; [heap[c], heap[i]] = [heap[i], heap[c]]; i = c; } } return top; };
  while (heap.length) {
    const [d, u] = pop(); if (done[u]) continue; done[u] = 1; if (u === tIdx) return d * g.cell;
    const ui = (u / w) | 0, uj = u % w;
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
      if (!di && !dj) continue; const ni = ui + di, nj = uj + dj; if (!walk(ni, nj)) continue;
      if (di && dj && !(walk(ui + di, uj) && walk(ui, uj + dj))) continue;
      const nd = d + (di && dj ? Math.SQRT2 : 1), k = ni * w + nj; if (nd < dist[k]) { dist[k] = nd; push([nd, k]); }
    }
  }
  return Infinity;
}

