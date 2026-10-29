import { crosshair, pixelToVoxel, type Plane, type PlaneInfo } from "./volume";

/**
 * Pan and zoom for one slice view. Everything that maps between voxels, plane pixels and
 * the screen (image, outlines, crosshair, markers, clicks, brush) goes through these
 * functions, so they cannot drift apart.
 *
 * Plane pixels: voxel (col, row) covers [col, col + 1) x [row, row + 1). Screen: CSS
 * pixels from the canvas's top-left corner. Zoom 1 fits the whole slice at its
 * physical aspect ratio.
 */
export interface ViewState {
  zoom: number;
  /** Plane point shown at the middle of the canvas. */
  center: [number, number];
}

export interface Viewport {
  width: number;
  height: number;
}

export const MIN_ZOOM = 0.5;
export const MAX_ZOOM = 16;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export const fitView = (info: PlaneInfo): ViewState => ({ zoom: 1, center: [info.width / 2, info.height / 2] });

/** CSS pixels per millimetre. */
export const pxPerMm = (info: PlaneInfo, vp: Viewport, zoom: number) => Math.min(vp.width / info.widthMm, vp.height / info.heightMm) * zoom;

/** CSS pixels covered by one plane pixel along screen x and y (they differ for anisotropic voxels). */
export function pixelSize(info: PlaneInfo, vp: Viewport, zoom: number): [number, number] {
  const k = pxPerMm(info, vp, zoom);
  return [(info.widthMm / info.width) * k, (info.heightMm / info.height) * k];
}

export function planeToScreen(info: PlaneInfo, vp: Viewport, view: ViewState, u: number, v: number): [number, number] {
  const [kx, ky] = pixelSize(info, vp, view.zoom);
  return [vp.width / 2 + (u - view.center[0]) * kx, vp.height / 2 + (v - view.center[1]) * ky];
}

export function screenToPlane(info: PlaneInfo, vp: Viewport, view: ViewState, x: number, y: number): [number, number] {
  const [kx, ky] = pixelSize(info, vp, view.zoom);
  return [view.center[0] + (x - vp.width / 2) / kx, view.center[1] + (y - vp.height / 2) / ky];
}

/** Keep the centre on the slice so the image can never be dragged out of sight. */
const clampCenter = (info: PlaneInfo, c: [number, number]): [number, number] => [clamp(c[0], 0, info.width), clamp(c[1], 0, info.height)];

/** Zoom by ``factor`` keeping the plane point under screen (x, y) fixed. */
export function zoomAt(info: PlaneInfo, vp: Viewport, view: ViewState, factor: number, x: number, y: number): ViewState {
  const zoom = clamp(view.zoom * factor, MIN_ZOOM, MAX_ZOOM);
  const [u, v] = screenToPlane(info, vp, view, x, y);
  const [kx, ky] = pixelSize(info, vp, zoom);
  return { zoom, center: clampCenter(info, [u - (x - vp.width / 2) / kx, v - (y - vp.height / 2) / ky]) };
}

/** Move the image by (dx, dy) screen pixels. */
export function panBy(info: PlaneInfo, vp: Viewport, view: ViewState, dx: number, dy: number): ViewState {
  const [kx, ky] = pixelSize(info, vp, view.zoom);
  return { zoom: view.zoom, center: clampCenter(info, [view.center[0] - dx / kx, view.center[1] - dy / ky]) };
}

export const centerOn = (info: PlaneInfo, view: ViewState, u: number, v: number): ViewState => ({ zoom: view.zoom, center: clampCenter(info, [u, v]) });

// ----- voxels <-> screen (the only path markers, the crosshair and clicks use) -----

type Shape = [number, number, number];

/** Plane point at the centre of voxel zyx, i.e. the middle of the pixel it is drawn as. */
export function voxelToPlane(shape: Shape, plane: Plane, zyx: [number, number, number]): [number, number] {
  const [col, row] = crosshair(shape, plane, zyx);
  return [col + 0.5, row + 0.5];
}

export function voxelToScreen(info: PlaneInfo, vp: Viewport, view: ViewState, shape: Shape, plane: Plane, zyx: [number, number, number]): [number, number] {
  const [u, v] = voxelToPlane(shape, plane, zyx);
  return planeToScreen(info, vp, view, u, v);
}

/** Plane pixel (col, row) under a screen point, or null outside the slice. */
export function screenToPixel(info: PlaneInfo, vp: Viewport, view: ViewState, x: number, y: number): [number, number] | null {
  const [u, v] = screenToPlane(info, vp, view, x, y);
  const col = Math.floor(u);
  const row = Math.floor(v);
  return col >= 0 && row >= 0 && col < info.width && row < info.height ? [col, row] : null;
}

/** Voxel (z, y, x) under a screen point on slice ``k``, or null outside the slice. */
export function screenToVoxel(
  info: PlaneInfo,
  vp: Viewport,
  view: ViewState,
  shape: Shape,
  plane: Plane,
  k: number,
  x: number,
  y: number,
): [number, number, number] | null {
  const p = screenToPixel(info, vp, view, x, y);
  return p ? pixelToVoxel(shape, plane, k, p[0], p[1]) : null;
}
