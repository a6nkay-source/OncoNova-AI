"""Geometric beam illustration. SIMULATED - NON-CLINICAL. Not a dose calculation.

For each beam angle in the axial plane, a binary aperture is cut to the
projection of the illustrative target, blurred to mimic a penumbra and
attenuated with a single exponential along the beam path through the body.
Beams are summed, smoothed and scaled so the median inside the target is 100%.

This ignores tissue heterogeneity, scatter physics, beam energy, field shaping,
couch and gantry geometry, and optimisation. Its only purpose is to show
qualitatively why more beam directions conform the high-dose region to a
target while spreading low dose through more tissue, and why larger margins
irradiate more normal tissue. Values are relative and have no units of Gy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from onconova.imaging.volume import Volume
from onconova.safety import SIMULATION_LABEL

ATTENUATION_PER_MM = 0.0045  # illustrative single coefficient, not a measured value
PENUMBRA_SIGMA_MM = 3.0


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    n_beams: int
    margin_mm: float
    description: str

    def to_dict(self) -> dict:
        return {**asdict(self), "label": SIMULATION_LABEL, "hypothetical": True}


SCENARIOS = {
    s.id: s for s in [
        Scenario("S1", "Hypothetical A: 3 coplanar beams, 5 mm margin", 3, 5.0,
                 "Few beam directions: entry paths carry high dose through normal tissue."),
        Scenario("S2", "Hypothetical B: 7 coplanar beams, 5 mm margin", 7, 5.0,
                 "More directions spread entry dose and conform the high-dose region."),
        Scenario("S3", "Hypothetical C: 36-angle arc-like, 5 mm margin", 36, 5.0,
                 "Many directions: tight high-dose region, wider low-dose spread."),
        Scenario("S4", "Hypothetical D: 7 coplanar beams, 10 mm margin", 7, 10.0,
                 "Same beams as B with a larger margin: more normal tissue in the high-dose region."),
    ]
}


def expand(mask: np.ndarray, margin_mm: float, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    if margin_mm <= 0:
        return mask.copy()
    dist = ndimage.distance_transform_edt(~mask, sampling=spacing_zyx)
    return dist <= margin_mm


def illustrate(volume: Volume, target: np.ndarray, body: np.ndarray, scenario: Scenario) -> dict:
    """Return the relative dose grid (1.0 = target median) and the expanded target."""
    if not target.any():
        raise ValueError("target mask is empty")
    sx, sy = volume.spacing[0], volume.spacing[1]
    warnings = [SIMULATION_LABEL]
    if abs(sx - sy) > 1e-3:
        warnings.append("In-plane spacing is anisotropic; beam rotation distorts the illustration.")
    ptv = expand(target, scenario.margin_mm, volume.spacing_zyx)
    dose = np.zeros(volume.shape, dtype=np.float32)
    penumbra_px = PENUMBRA_SIGMA_MM / sx
    for angle in np.linspace(0, 360, scenario.n_beams, endpoint=False):
        # Rotate so the beam travels along +y (rows), cut the aperture, rotate back.
        rot_ptv = ndimage.rotate(ptv.astype(np.float32), -angle, axes=(1, 2), reshape=False, order=1) > 0.5
        rot_body = ndimage.rotate(body.astype(np.float32), -angle, axes=(1, 2), reshape=False, order=1) > 0.5
        aperture = rot_ptv.any(axis=1)  # (z, x) beam's-eye view per slice
        # Open the aperture by about one penumbra width so the target edge is not underdosed.
        aperture = ndimage.binary_dilation(aperture, iterations=max(1, int(round(penumbra_px)))).astype(np.float32)
        aperture = ndimage.gaussian_filter(aperture, sigma=(PENUMBRA_SIGMA_MM / volume.spacing[2], penumbra_px))
        depth = np.cumsum(rot_body, axis=1) * sy
        beam = aperture[:, None, :] * np.exp(-ATTENUATION_PER_MM * depth) * rot_body
        # Equal-weight beams at the target: each beam delivers the same mean to the target.
        beam /= max(float(beam[rot_ptv].mean()), 1e-6)
        dose += ndimage.rotate(beam.astype(np.float32), angle, axes=(1, 2), reshape=False, order=1)
    dose = ndimage.gaussian_filter(dose, sigma=0.8) * body
    median = float(np.median(dose[ptv])) or 1.0
    dose = (dose / median).astype(np.float32)
    return {"dose": dose, "ptv": ptv, "warnings": warnings, "units": "relative (1.0 = median in illustrative target)"}
