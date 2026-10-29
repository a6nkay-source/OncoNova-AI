import type { VolumeData } from "../api/client";
import { isAxisAligned } from "./geometry";

export type Plane = "axial" | "coronal" | "sagittal";

export interface PlaneInfo {
  width: number; // voxels
  height: number;
  widthMm: number;
  heightMm: number;
  count: number; // number of slices along the plane normal
}

/** Display conventions: radiological (patient right on screen left), anterior/superior at top. */
export function planeInfo(shape: [number, number, number], spacing: [number, number, number], plane: Plane): PlaneInfo {
  const [nz, ny, nx] = shape;
  const [sx, sy, sz] = spacing;
  if (plane === "axial") return { width: nx, height: ny, widthMm: nx * sx, heightMm: ny * sy, count: nz };
  if (plane === "coronal") return { width: nx, height: nz, widthMm: nx * sx, heightMm: nz * sz, count: ny };
  return { width: ny, height: nz, widthMm: ny * sy, heightMm: nz * sz, count: nx };
}

/** Flat voxel index for display pixel (col, row) on slice ``k`` of ``plane``. */
export function voxelIndex(shape: [number, number, number], plane: Plane, k: number, col: number, row: number): number {
  const [nz, ny, nx] = shape;
  if (plane === "axial") return k * ny * nx + row * nx + col;
  if (plane === "coronal") return (nz - 1 - row) * ny * nx + k * nx + col;
  return (nz - 1 - row) * ny * nx + col * nx + k;
}

/** Voxel (z, y, x) under display pixel (col, row). */
export function pixelToVoxel(shape: [number, number, number], plane: Plane, k: number, col: number, row: number): [number, number, number] {
  const nz = shape[0];
  if (plane === "axial") return [k, row, col];
  if (plane === "coronal") return [nz - 1 - row, k, col];
  return [nz - 1 - row, col, k];
}

export function sliceIndexFor(plane: Plane, zyx: [number, number, number]): number {
  return plane === "axial" ? zyx[0] : plane === "coronal" ? zyx[1] : zyx[2];
}

/** Crosshair position (col, row) on ``plane`` for voxel zyx. */
export function crosshair(shape: [number, number, number], plane: Plane, zyx: [number, number, number]): [number, number] {
  const nz = shape[0];
  if (plane === "axial") return [zyx[2], zyx[1]];
  if (plane === "coronal") return [zyx[2], nz - 1 - zyx[0]];
  return [zyx[1], nz - 1 - zyx[0]];
}

export function voxelValue(v: VolumeData, i: number): number {
  return v.data[i] * v.scale + v.offset;
}

/** Raw label values of one slice, row-major in display order (width x height). */
export function sliceLabels(v: VolumeData, plane: Plane, k: number): Uint8Array {
  const info = planeInfo(v.shape, v.spacing, plane);
  const out = new Uint8Array(info.width * info.height);
  for (let row = 0; row < info.height; row++)
    for (let col = 0; col < info.width; col++) out[row * info.width + col] = v.data[voxelIndex(v.shape, plane, k, col, row)];
  return out;
}

/** LPS axis along the normal of ``plane``: x for sagittal, y for coronal, z for axial. */
export const normalAxis = (plane: Plane): 0 | 1 | 2 => (plane === "axial" ? 2 : plane === "coronal" ? 1 : 0);

/**
 * Position of slice ``k`` along the plane normal (LPS mm). Null without an origin, or for an
 * oblique scan, whose slices are not at one patient coordinate.
 */
export function sliceOffsetMm(v: VolumeData, plane: Plane, k: number): number | null {
  if (!v.origin || !isAxisAligned(v)) return null;
  const axis = normalAxis(plane);
  return v.origin[axis] + k * v.spacing[axis];
}

// ----- colour maps -------------------------------------------------------------

export type RGB = [number, number, number];

export const LESION_COLORS: Record<number, RGB> = {
  1: [47, 208, 184], // primary - teal
  2: [244, 183, 64], // node-region - amber
  3: [229, 140, 255], // other - violet
};

export const STRUCTURE_COLORS: Record<number, RGB> = {
  1: [98, 168, 255], // lung L
  2: [124, 140, 255], // lung R
  3: [255, 110, 130], // heart
  4: [255, 160, 90], // oesophagus
  5: [255, 226, 90], // spinal cord
  6: [0, 0, 0], // body: not drawn
};

function lerpStops(stops: [number, RGB][], t: number): RGB {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    if (x <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const f = (x - t0) / (t1 - t0 || 1);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return stops[stops.length - 1][1];
}

function buildLut(stops: [number, RGB][]): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const c = lerpStops(stops, i / 255);
    lut[i * 3] = c[0];
    lut[i * 3 + 1] = c[1];
    lut[i * 3 + 2] = c[2];
  }
  return lut;
}

export const LUTS = {
  // perceptual dose wash: blue -> cyan -> green -> yellow -> red
  dose: buildLut([
    [0, [30, 60, 200]],
    [0.25, [20, 170, 230]],
    [0.5, [40, 210, 120]],
    [0.72, [240, 220, 40]],
    [0.86, [250, 130, 30]],
    [1, [230, 30, 50]],
  ]),
  uncertainty: buildLut([
    [0, [40, 10, 80]],
    [0.4, [160, 40, 140]],
    [0.75, [250, 120, 70]],
    [1, [255, 240, 160]],
  ]),
  saliency: buildLut([
    [0, [60, 0, 0]],
    [0.5, [230, 60, 20]],
    [1, [255, 245, 150]],
  ]),
};

export type LutName = keyof typeof LUTS;
