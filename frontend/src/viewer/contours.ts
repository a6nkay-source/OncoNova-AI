/**
 * Outlines for label overlays. Display only: the mask itself is never changed.
 *
 * Marching squares at the 0.5 level, run on the grid of voxel centres. Every vertex is
 * the midpoint of an edge shared by one inside and one outside voxel, so the outline
 * runs exactly along the mask's own voxel boundary and only cuts each staircase corner
 * on the diagonal. Each convex corner loses 1/8 voxel and each concave corner gains
 * 1/8, so a simple outline encloses exactly 0.5 voxel less than its voxel count (and a
 * hole's outline gives that half voxel back). Diagonal-only neighbours are kept apart.
 *
 * Coordinates are plane pixels: voxel (col, row) covers [col, col + 1) x [row, row + 1).
 */

/** Closed polyline as x0, y0, x1, y1, ...; the last point joins the first. */
export type Loop = Float32Array;

export function traceContours(labels: ArrayLike<number>, width: number, height: number, value: number): Loop[] {
  // Padded grid with an empty border so every outline closes.
  const W = width + 2;
  const H = height + 2;
  const grid = new Uint8Array(W * H);
  let minC = width;
  let maxC = -1;
  let minR = height;
  let maxR = -1;
  for (let r = 0; r < height; r++)
    for (let c = 0; c < width; c++) {
      if (labels[r * width + c] !== value) continue;
      grid[(r + 1) * W + c + 1] = 1;
      if (c < minC) minC = c;
      if (c > maxC) maxC = c;
      if (r < minR) minR = r;
      if (r > maxR) maxR = r;
    }
  if (maxC < 0) return [];

  // Edge ids: the edge from padded centre (i, j) to (i + 1, j) is 2 * (j * W + i); the
  // edge from (i, j) to (i, j + 1) is that plus one. Each crossed edge joins two segments.
  const first = new Map<number, number>();
  const second = new Map<number, number>();
  const add = (e: number, other: number) => (first.has(e) ? second.set(e, other) : first.set(e, other));
  const join = (a: number, b: number) => {
    add(a, b);
    add(b, a);
  };
  for (let j = minR; j <= maxR + 1; j++)
    for (let i = minC; i <= maxC + 1; i++) {
      const tl = grid[j * W + i];
      const tr = grid[j * W + i + 1];
      const br = grid[(j + 1) * W + i + 1];
      const bl = grid[(j + 1) * W + i];
      const code = (tl << 3) | (tr << 2) | (br << 1) | bl;
      if (code === 0 || code === 15) continue;
      const top = 2 * (j * W + i);
      const bottom = 2 * ((j + 1) * W + i);
      const left = top + 1;
      const right = 2 * (j * W + i + 1) + 1;
      switch (code) {
        case 1:
        case 14:
          join(left, bottom);
          break;
        case 2:
        case 13:
          join(bottom, right);
          break;
        case 3:
        case 12:
          join(left, right);
          break;
        case 4:
        case 11:
          join(top, right);
          break;
        case 6:
        case 9:
          join(top, bottom);
          break;
        case 7:
        case 8:
          join(top, left);
          break;
        case 5: // tr and bl inside, kept apart
          join(top, right);
          join(left, bottom);
          break;
        case 10: // tl and br inside, kept apart
          join(top, left);
          join(bottom, right);
          break;
      }
    }

  const point = (e: number): [number, number] => {
    const k = e >> 1;
    const i = k % W;
    const j = (k - i) / W;
    // Padded centre (i, j) sits at plane (i - 0.5, j - 0.5).
    return e & 1 ? [i - 0.5, j] : [i, j - 0.5];
  };
  const loops: Loop[] = [];
  const seen = new Set<number>();
  for (const start of first.keys()) {
    if (seen.has(start)) continue;
    const pts: number[] = [];
    let prev = -1;
    let cur = start;
    do {
      seen.add(cur);
      const [x, y] = point(cur);
      pts.push(x, y);
      const a = first.get(cur)!;
      const next = a !== prev ? a : second.get(cur)!;
      prev = cur;
      cur = next;
    } while (cur !== start);
    loops.push(Float32Array.from(pts));
  }
  return loops;
}

/** Signed area of a closed loop (shoelace formula), in plane pixels squared. */
export function loopArea(loop: Loop): number {
  let s = 0;
  const n = loop.length / 2;
  for (let p = 0; p < n; p++) {
    const q = (p + 1) % n;
    s += loop[2 * p] * loop[2 * q + 1] - loop[2 * q] * loop[2 * p + 1];
  }
  return s / 2;
}
