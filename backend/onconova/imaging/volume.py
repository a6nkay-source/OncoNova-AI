"""A 3-D image that keeps voxel data and physical geometry together."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import SimpleITK as sitk

IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass
class Volume:
    """Voxel array plus geometry in SimpleITK conventions.

    ``array`` is indexed ``[z, y, x]``. ``spacing`` and ``origin`` are in millimetres
    and ordered ``(x, y, z)``; ``direction`` is the row-major 3x3 cosine matrix. All
    physical coordinates are DICOM/ITK patient coordinates (LPS).
    """

    array: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, ...] = IDENTITY_DIRECTION
    modality: str = "UNKNOWN"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.array.shape)  # type: ignore[return-value]

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        return (self.spacing[2], self.spacing[1], self.spacing[0])

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def direction_matrix(self) -> np.ndarray:
        return np.asarray(self.direction, dtype=float).reshape(3, 3)

    @property
    def is_oblique(self) -> bool:
        d = np.abs(self.direction_matrix)
        return bool(np.any((d > 1e-3) & (d < 1 - 1e-3)))

    def index_to_physical(self, index_zyx: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        """Map voxel index/indices ``[..., (z, y, x)]`` to LPS millimetres ``[..., (x, y, z)]``."""
        idx = np.asarray(index_zyx, dtype=float)
        ijk = idx[..., ::-1] * np.asarray(self.spacing)
        return ijk @ self.direction_matrix.T + np.asarray(self.origin)

    def with_array(self, array: np.ndarray, **changes: Any) -> Volume:
        if array.shape != self.array.shape:
            raise ValueError(f"shape {array.shape} does not match volume shape {self.array.shape}")
        return replace(self, array=array, **changes)

    def same_geometry(self, other: Volume, tol: float = 1e-3) -> bool:
        return (
            self.shape == other.shape
            and np.allclose(self.spacing, other.spacing, atol=tol)
            and np.allclose(self.origin, other.origin, atol=tol * 10)
            and np.allclose(self.direction, other.direction, atol=tol)
        )

    def geometry(self) -> dict[str, Any]:
        return {
            "shape_zyx": list(self.shape),
            "spacing_xyz_mm": [round(float(s), 6) for s in self.spacing],
            "origin_xyz_mm": [round(float(o), 4) for o in self.origin],
            "direction": [round(float(d), 6) for d in self.direction],
            "coordinate_system": "LPS",
            "oblique": self.is_oblique,
        }

    def to_sitk(self) -> sitk.Image:
        image = sitk.GetImageFromArray(self.array)
        image.SetSpacing(tuple(float(s) for s in self.spacing))
        image.SetOrigin(tuple(float(o) for o in self.origin))
        image.SetDirection(tuple(float(d) for d in self.direction))
        return image

    @classmethod
    def from_sitk(cls, image: sitk.Image, *, modality: str = "UNKNOWN",
                  metadata: dict[str, Any] | None = None, dtype: Any = None) -> Volume:
        array = sitk.GetArrayFromImage(image)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return cls(
            array=array,
            spacing=tuple(float(s) for s in image.GetSpacing()),  # type: ignore[arg-type]
            origin=tuple(float(o) for o in image.GetOrigin()),  # type: ignore[arg-type]
            direction=tuple(float(d) for d in image.GetDirection()),
            modality=modality,
            metadata=dict(metadata or {}),
        )
