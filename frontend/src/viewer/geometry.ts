import type { VolumeData } from "../api/client";

/**
 * Patient geometry on the client. The server reorients every scan to LPS (+x left,
 * +y posterior, +z superior) without resampling, so axis-aligned scans arrive with an
 * identity direction; only oblique scans keep a rotation.
 */
const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
// Same tolerances as the server's Volume.same_geometry.
const TOL = { spacing: 1e-3, origin: 1e-2, direction: 1e-3 };

const close = (a: readonly number[], b: readonly number[], tol: number) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= tol);

/** True when the array axes run exactly along L, P and S, so plane labels and slice positions are exact. */
export const isAxisAligned = (v: VolumeData) => !v.direction || close(v.direction, IDENTITY, TOL.direction);

/** LPS millimetres of the centre of voxel (z, y, x), as ``Volume.index_to_physical`` on the server. */
export function voxelToPatient(v: VolumeData, zyx: [number, number, number]): [number, number, number] | null {
  if (!v.origin) return null;
  const d = v.direction ?? IDENTITY;
  const ijk = [zyx[2] * v.spacing[0], zyx[1] * v.spacing[1], zyx[0] * v.spacing[2]];
  const p = (r: number) => v.origin![r] + d[r * 3] * ijk[0] + d[r * 3 + 1] * ijk[1] + d[r * 3 + 2] * ijk[2];
  return [p(0), p(1), p(2)];
}

/** One LPS coordinate as an anatomical direction, e.g. -12.5 on x is "R 12.5". */
export function patientAxis(axis: 0 | 1 | 2, mm: number): string {
  const [neg, pos] = (["RL", "AP", "IS"] as const)[axis];
  const v = Math.abs(mm) < 0.05 ? 0 : mm;
  return `${v < 0 ? neg : pos} ${Math.abs(v).toFixed(1)}`;
}

const dims = (v: VolumeData) => [...v.shape].reverse().join("×");
const mm3 = (s: readonly number[]) => s.map((x) => x.toFixed(2)).join("×");

/**
 * Why ``overlay`` must not be drawn on ``image``, or null when both share one voxel grid.
 * Matching voxel counts are not enough: a mask from another scan, or on a shifted grid,
 * would be drawn in the wrong place without any visible sign.
 */
export function geometryMismatch(image: VolumeData, overlay: VolumeData): string | null {
  if (image.frame && overlay.frame && image.frame !== overlay.frame) return `it belongs to scan ${overlay.frame}, not ${image.frame}`;
  if (overlay.shape.some((n, i) => n !== image.shape[i])) return `its matrix ${dims(overlay)} differs from the scan's ${dims(image)}`;
  if (!close(overlay.spacing, image.spacing, TOL.spacing)) return `its voxel size ${mm3(overlay.spacing)} mm differs from the scan's ${mm3(image.spacing)} mm`;
  if (!image.origin !== !overlay.origin) return "its position in the patient is unknown";
  if (image.origin && overlay.origin && !close(overlay.origin, image.origin, TOL.origin)) {
    const off = Math.hypot(...overlay.origin.map((o, i) => o - image.origin![i]));
    return `its grid is shifted ${off.toFixed(1)} mm from the scan's`;
  }
  if (!close(overlay.direction ?? IDENTITY, image.direction ?? IDENTITY, TOL.direction)) return "its axes are rotated relative to the scan's";
  return null;
}
