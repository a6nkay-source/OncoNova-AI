"""Physical measurements of segmented structures."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from skimage.measure import marching_cubes, mesh_surface_area

from onconova.imaging.volume import Volume


def volume_cc(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> float:
    return float(np.count_nonzero(mask)) * float(np.prod(spacing_xyz)) / 1000.0


def surface_area_mm2(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> float | None:
    if np.count_nonzero(mask) < 8:
        return None
    padded = np.pad(mask.astype(np.float32), 1)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing_zyx)
    return float(mesh_surface_area(verts, faces))


def sphericity(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> float | None:
    """1.0 for a sphere, lower for irregular shapes. Voxelisation biases it slightly low."""
    area = surface_area_mm2(mask, spacing_zyx)
    if not area:
        return None
    vol = float(np.count_nonzero(mask)) * float(np.prod(spacing_zyx))
    return float(min(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area, 1.0))


def max_axial_diameter_mm(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> float:
    """Longest in-plane extent over all axial slices (a measurement, not a RECIST read).

    The longest distance between voxel centres, plus half a voxel's width along that
    direction (in mm, so anisotropic pixels are handled). Centres alone underestimate;
    adding a whole voxel overestimates round shapes. On simulated ellipses 10-40 mm across
    this leaves a shape-dependent error of about -0.7 to +0.6 voxel (see tests); a binary
    mask cannot resolve the diameter more finely than that.
    """
    best = 0.0
    sx, sy = spacing_xyz[0], spacing_xyz[1]
    for z in np.unique(np.nonzero(mask)[0]):
        pts = np.argwhere(mask[z]).astype(float) * np.array([sy, sx])
        if len(pts) == 1:
            best = max(best, max(sx, sy))  # below resolution: report one voxel
            continue
        try:
            hull = pts[ConvexHull(pts).vertices] if len(pts) >= 3 else pts
        except QhullError:  # collinear voxels
            hull = pts
        d2 = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(d2.argmax()), d2.shape)
        dy, dx = (hull[j] - hull[i]) / np.sqrt(d2[i, j])
        best = max(best, float(np.sqrt(d2[i, j])) + 0.5 * (sy * abs(dy) + sx * abs(dx)))
    return best


def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 2))
    return labels, int(n)


def describe_component(volume: Volume, mask: np.ndarray, class_map: np.ndarray | None = None) -> dict[str, Any]:
    idx = np.argwhere(mask)
    centroid_idx = idx.mean(0)
    out: dict[str, Any] = {
        "volume_cc": round(volume_cc(mask, volume.spacing), 4),
        "centroid_mm": [round(float(v), 2) for v in volume.index_to_physical(centroid_idx)],
        "centroid_index_zyx": [round(float(v), 1) for v in centroid_idx],
        "bbox_index_zyx": [idx.min(0).tolist(), idx.max(0).tolist()],
        "max_axial_diameter_mm": round(max_axial_diameter_mm(mask, volume.spacing), 2),
        "sphericity": None,
    }
    sph = sphericity(mask, volume.spacing_zyx)
    out["sphericity"] = None if sph is None else round(sph, 4)
    if class_map is not None:
        values, counts = np.unique(class_map[mask], return_counts=True)
        out["class_id"] = int(values[np.argmax(counts)])
    return out


def mesh(mask: np.ndarray, volume: Volume, step: int = 1, max_faces: int = 60000) -> dict[str, Any] | None:
    """Surface mesh in LPS millimetres for 3-D display."""
    if np.count_nonzero(mask) < 8:
        return None
    padded = np.pad(mask.astype(np.float32), 1)
    padded = ndimage.gaussian_filter(padded, 0.6)
    verts, faces, normals, _ = marching_cubes(padded, level=0.5, step_size=step)
    while len(faces) > max_faces and step < 4:
        step += 1
        verts, faces, normals, _ = marching_cubes(padded, level=0.5, step_size=step)
    verts_phys = volume.index_to_physical(verts - 1)
    return {
        "vertices": np.round(verts_phys, 2).ravel().tolist(),
        "faces": faces.astype(int).ravel().tolist(),
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
        "coordinate_system": "LPS mm",
    }
