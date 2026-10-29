"""Move images onto a model's working grid and bring its outputs back.

A model trained at one voxel spacing sees a scan at another spacing as magnified or
shrunk, so inputs are resampled to the training spacing first. The new grid covers
the same patient region: its first and last voxel edges coincide with the scan's
(to within one model voxel), and its direction is the scan's. Outputs are resampled
back onto the scan's own grid, so masks, measurements and the viewer stay in the
scan's geometry.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import SimpleITK as sitk

from onconova.imaging.volume import Volume


def needs_resampling(volume: Volume, spacing_xyz: tuple[float, float, float] | None, rtol: float = 0.02) -> bool:
    return spacing_xyz is not None and not np.allclose(volume.spacing, spacing_xyz, rtol=rtol)


def model_grid(volume: Volume, spacing_xyz: tuple[float, float, float]) -> Volume:
    """An empty volume at ``spacing_xyz`` covering the same patient region as ``volume``."""
    s = np.asarray(volume.spacing, dtype=float)
    t = np.asarray(spacing_xyz, dtype=float)
    size_xyz = np.maximum(1, np.round(np.asarray(volume.shape[::-1]) * s / t)).astype(int)
    # Align the first voxel edges: old edge = origin - D s/2, new centre = edge + D t/2.
    origin = np.asarray(volume.origin) + volume.direction_matrix @ ((t - s) / 2)
    return Volume(np.zeros(tuple(size_xyz[::-1]), np.float32), spacing=tuple(float(v) for v in t),
                  origin=tuple(float(v) for v in origin), direction=volume.direction, modality=volume.modality)


def _resample(array: np.ndarray, source: Volume, target: Volume, *, linear: bool) -> np.ndarray:
    img = source.with_array(np.ascontiguousarray(array, dtype=np.float32) if linear else array).to_sitk()
    out = sitk.Resample(img, target.to_sitk(), sitk.Transform(3, sitk.sitkIdentity),
                        sitk.sitkLinear if linear else sitk.sitkNearestNeighbor, 0.0, img.GetPixelID(),
                        True)  # nearest-neighbour extrapolation at the edges, never a constant fill
    return sitk.GetArrayFromImage(out)


def to_grid(array: np.ndarray, source: Volume, target: Volume, *, linear: bool = True) -> np.ndarray:
    """Resample an array laid out on ``source``'s grid onto ``target``'s grid."""
    return _resample(array, source, target, linear=linear)


def channels_to_grid(probs: np.ndarray, source: Volume, target: Volume) -> Iterator[np.ndarray]:
    """Resample class probabilities (C, z, y, x) channel by channel, to keep memory to one channel.

    Linear interpolation is a convex combination, so the channels still sum to one.
    """
    for c in range(probs.shape[0]):
        yield _resample(probs[c], source, target, linear=True)
