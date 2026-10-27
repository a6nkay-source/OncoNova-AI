"""Rigid registration between scans of the same patient (SimpleITK)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from onconova.imaging.volume import Volume


@dataclass
class RegistrationResult:
    transform: sitk.Transform
    final_metric: float
    iterations: int
    translation_mm: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]
    stop_condition: str

    @property
    def qc(self) -> str:
        """Coarse quality flag. Large motion between serial scans needs a human look."""
        moved = float(np.linalg.norm(self.translation_mm))
        rotated = float(np.max(np.abs(self.rotation_deg)))
        return "review" if moved > 30 or rotated > 10 else "ok"

    def summary(self) -> dict:
        return {
            "method": "rigid (Euler 3-D), Mattes mutual information, 3-level pyramid",
            "final_metric": round(self.final_metric, 5),
            "iterations": self.iterations,
            "translation_mm": [round(t, 2) for t in self.translation_mm],
            "rotation_deg": [round(r, 2) for r in self.rotation_deg],
            "stop_condition": self.stop_condition,
            "qc": self.qc,
        }


def register_rigid(fixed: Volume, moving: Volume, *, seed: int = 7) -> RegistrationResult:
    """Estimate the transform that maps fixed-space points into the moving image."""
    fixed_img = sitk.Cast(fixed.to_sitk(), sitk.sitkFloat32)
    moving_img = sitk.Cast(moving.to_sitk(), sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-3, numberOfIterations=200, gradientMagnitudeTolerance=1e-6
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    reg.SetInitialTransform(initial, inPlace=False)
    transform = reg.Execute(fixed_img, moving_img)

    euler = sitk.Euler3DTransform(sitk.CompositeTransform(transform).GetNthTransform(0)) \
        if transform.GetName() == "CompositeTransform" else sitk.Euler3DTransform(transform)
    angles = np.degrees([euler.GetAngleX(), euler.GetAngleY(), euler.GetAngleZ()])
    return RegistrationResult(
        transform=transform,
        final_metric=float(reg.GetMetricValue()),
        iterations=int(reg.GetOptimizerIteration()),
        translation_mm=tuple(float(t) for t in euler.GetTranslation()),  # type: ignore[arg-type]
        rotation_deg=tuple(float(a) for a in angles),  # type: ignore[arg-type]
        stop_condition=reg.GetOptimizerStopConditionDescription(),
    )


def resample_to(reference: Volume, moving: Volume, transform: sitk.Transform | None = None,
                *, is_label: bool = False, default_value: float | None = None) -> np.ndarray:
    """Resample ``moving`` onto the grid of ``reference`` (nearest-neighbour for labels)."""
    if default_value is None:
        default_value = 0.0 if is_label else float(moving.array.min())
    moving_img = moving.to_sitk()
    out = sitk.Resample(
        moving_img,
        reference.to_sitk(),
        transform or sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        default_value,
        moving_img.GetPixelID(),
    )
    return sitk.GetArrayFromImage(out)
