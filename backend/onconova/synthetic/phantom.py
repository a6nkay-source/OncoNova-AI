"""Synthetic thoracic CT phantoms with known ground truth. Every output is DEMO DATA.

Anatomy is built from analytic shapes in LPS patient coordinates (mm, origin at the
volume centre): body wall, lungs with vessel-like tubes, heart, mediastinal fat,
oesophagus, vertebra and spinal cord, plus a lobulated primary lesion, an optional
lymph-node-region lesion and optional new nodules at follow-up. The phantom exists
to exercise the pipeline. It is not anatomically accurate and resembles no person.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from onconova.imaging.volume import Volume
from onconova.safety import DEMO_LABEL

GENERATOR_VERSION = "thorax-phantom-0.1"
SHAPE_ZYX = (72, 128, 128)
SPACING_XYZ = (2.2, 2.2, 2.5)

LESION_CLASSES = {1: "primary_tumor", 2: "lymph_node_region_lesion", 3: "other_lesion"}
STRUCTURE_CLASSES = {1: "lung_left", 2: "lung_right", 3: "heart", 4: "esophagus", 5: "spinal_cord", 6: "body_other"}

BODY_SEMI_XY = (128.0, 100.0)
LUNG_SEMI = np.array([50.0, 68.0, 85.0])
HEART_CENTER = np.array([15.0, -28.0, -30.0])
HEART_SEMI = np.array([46.0, 38.0, 42.0])
MED_FAT_CENTER = np.array([0.0, -14.0, 34.0])
MED_FAT_SEMI = np.array([24.0, 28.0, 38.0])
MAX_PRIMARY_RADIUS_MM = 24.0


def lung_center(side: int) -> np.ndarray:
    """side=+1 is the patient's left lung (+x in LPS), side=-1 the right."""
    return np.array([side * 64.0, -5.0, 5.0])


@dataclass
class Sphere:
    center: tuple[float, float, float]
    radius: float


@dataclass
class TimepointSpec:
    tp_id: str
    days: int
    primary_scale: float
    node_radius_mm: float
    other_lesions: list[Sphere]
    shift_mm: tuple[float, float, float]
    rotation_deg: float


@dataclass
class PatientSpec:
    patient_id: str
    seed: int
    trajectory: str
    primary_side: int
    primary_center: tuple[float, float, float]
    primary_radius_mm: float
    lobules: list[tuple[tuple[float, float, float], float]]
    node_center: tuple[float, float, float]
    vessels: list[tuple[tuple[float, float, float], tuple[float, float, float], float]]
    timepoints: list[TimepointSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _inside_ellipsoid(point: np.ndarray, center: np.ndarray, semi: np.ndarray) -> bool:
    return float(np.sum(((point - center) / semi) ** 2)) <= 1.0


def _random_point_in_lung(rng: np.random.Generator, side: int, core: float) -> np.ndarray:
    for _ in range(200):
        u = rng.normal(size=3)
        u = u / np.linalg.norm(u) * rng.uniform(0, 1) ** (1 / 3)
        point = lung_center(side) + u * LUNG_SEMI * core
        if not _inside_ellipsoid(point, HEART_CENTER, HEART_SEMI * 1.3):
            return point
    return lung_center(side) + np.array([side * 10.0, 0.0, 30.0])


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    t = np.clip(np.dot(p - a, d) / np.dot(d, d), 0, 1)
    return float(np.linalg.norm(p - (a + t * d)))


def random_patient_spec(patient_id: str, seed: int, *, n_timepoints: int = 3, trajectory: str | None = None,
                        node: bool | None = None, new_lesion: bool | None = None) -> PatientSpec:
    rng = np.random.default_rng(seed)
    trajectory = trajectory or str(rng.choice(["growing", "stable", "responding"], p=[0.4, 0.3, 0.3]))
    side = int(rng.choice([-1, 1]))
    center = _random_point_in_lung(rng, side, core=0.4)
    r0 = float(rng.uniform(9.0, 19.0))
    lobules = [(tuple(float(v) for v in rng.uniform(-0.35, 0.35, 3)), float(rng.uniform(0.55, 0.8)))
               for _ in range(int(rng.integers(2, 5)))]
    k = {"growing": rng.uniform(0.25, 0.6), "stable": rng.uniform(-0.08, 0.08),
         "responding": rng.uniform(-0.6, -0.25)}[trajectory]  # log volume change per 84 days
    has_node = bool(rng.random() < 0.35) if node is None else node
    node_center = (float(rng.choice([-1, 1]) * rng.uniform(10, 15)), float(-14 + rng.uniform(-5, 5)),
                   float(34 + rng.uniform(-10, 10)))
    node_r0 = float(rng.uniform(5.0, 8.5)) if has_node else 0.0
    if new_lesion is None:
        new_lesion = trajectory == "growing" and bool(rng.random() < 0.5)

    reach = MAX_PRIMARY_RADIUS_MM * 1.2 + 8
    vessels = []
    for i in range(12):
        if len(vessels) == 5:
            break
        s = 1 if i % 2 == 0 else -1
        start = np.array([s * rng.uniform(22, 30), rng.uniform(-15, 10), rng.uniform(-20, 40)])
        direction = np.array([s * rng.uniform(0.6, 1.0), rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)])
        end = start + direction / np.linalg.norm(direction) * rng.uniform(35, 60)
        if _segment_distance(center, start, end) > reach:
            vessels.append((tuple(map(float, start)), tuple(map(float, end)), float(rng.uniform(1.6, 2.4))))

    days = [0, 84 + int(rng.integers(-14, 15)), 168 + int(rng.integers(-14, 15))]
    new_at = n_timepoints - 1 if new_lesion and n_timepoints > 1 else None
    other_center = _random_point_in_lung(rng, -side, core=0.45)
    other_radius = float(rng.uniform(4.0, 7.0))

    timepoints = []
    for i in range(n_timepoints):
        d = days[i] if i < len(days) else days[-1] + 84 * (i - 2)
        scale = min(float(np.exp(k * d / 84 / 3)), MAX_PRIMARY_RADIUS_MM / r0)
        node_r = node_r0 * float(np.exp(0.8 * k * d / 84 / 3)) if has_node else 0.0
        others = [Sphere(tuple(map(float, other_center)), other_radius)] if new_at is not None and i >= new_at else []
        shift = (0.0, 0.0, 0.0) if i == 0 else tuple(float(v) for v in rng.uniform(-6, 6, 3))
        rotation = 0.0 if i == 0 else float(rng.uniform(-3, 3))
        timepoints.append(TimepointSpec(f"T{i}", d, scale, node_r, others, shift, rotation))  # type: ignore[arg-type]

    return PatientSpec(patient_id, seed, trajectory, side, tuple(map(float, center)), r0, lobules,  # type: ignore[arg-type]
                       node_center, vessels, timepoints)


def _grid(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> tuple[np.ndarray, ...]:
    nz, ny, nx = shape
    sx, sy, sz = spacing
    z = ((np.arange(nz) - (nz - 1) / 2) * sz).astype(np.float32)
    y = ((np.arange(ny) - (ny - 1) / 2) * sy).astype(np.float32)
    x = ((np.arange(nx) - (nx - 1) / 2) * sx).astype(np.float32)
    return np.meshgrid(z, y, x, indexing="ij")


def render_timepoint(spec: PatientSpec, tp: TimepointSpec, *, shape: tuple[int, int, int] = SHAPE_ZYX,
                     spacing: tuple[float, float, float] = SPACING_XYZ) -> tuple[Volume, Volume, Volume]:
    """Return (CT image in HU, lesion label map, structure label map) for one timepoint."""
    rng = np.random.default_rng(spec.seed * 1000 + int(tp.tp_id[1:]))
    pz, py, px = _grid(shape, spacing)
    theta = np.radians(tp.rotation_deg)
    tx, ty, tz = tp.shift_mm
    ax = np.cos(theta) * (px - tx) + np.sin(theta) * (py - ty)
    ay = -np.sin(theta) * (px - tx) + np.cos(theta) * (py - ty)
    az = pz - tz

    def ellipsoid(center: np.ndarray, semi: np.ndarray) -> np.ndarray:
        return ((ax - center[0]) / semi[0]) ** 2 + ((ay - center[1]) / semi[1]) ** 2 + ((az - center[2]) / semi[2]) ** 2 <= 1

    def sphere(center, radius: float) -> np.ndarray:
        c = np.asarray(center, dtype=np.float32)
        return (ax - c[0]) ** 2 + (ay - c[1]) ** 2 + (az - c[2]) ** 2 <= radius ** 2

    def tube(a, b, radius: float) -> np.ndarray:
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        d = b - a
        t = np.clip(((ax - a[0]) * d[0] + (ay - a[1]) * d[1] + (az - a[2]) * d[2]) / float(d @ d), 0, 1)
        return (ax - a[0] - t * d[0]) ** 2 + (ay - a[1] - t * d[1]) ** 2 + (az - a[2] - t * d[2]) ** 2 <= radius ** 2

    def smooth_noise(sigma: float) -> np.ndarray:
        field_ = gaussian_filter(rng.normal(size=shape).astype(np.float32), sigma)
        return field_ / (field_.std() + 1e-6)

    bx, by = BODY_SEMI_XY
    body = (ax / bx) ** 2 + (ay / by) ** 2 <= 1
    inner = (ax / (bx - 9)) ** 2 + (ay / (by - 9)) ** 2 <= 1
    heart = ellipsoid(HEART_CENTER, HEART_SEMI)
    heart_margin = ellipsoid(HEART_CENTER, HEART_SEMI + 4)
    vertebra = (ax ** 2 + (ay - 62) ** 2 <= 16 ** 2) & body
    canal_ring = (ax ** 2 + (ay - 86) ** 2 <= 11 ** 2) & body
    canal = ax ** 2 + (ay - 86) ** 2 <= 8 ** 2
    cord = ax ** 2 + (ay - 86) ** 2 <= 4.5 ** 2
    bone = vertebra | canal_ring
    lung_l = ellipsoid(lung_center(1), LUNG_SEMI) & ~heart_margin & ~bone & body
    lung_r = ellipsoid(lung_center(-1), LUNG_SEMI) & ~heart_margin & ~bone & body
    lungs = lung_l | lung_r
    med_fat = ellipsoid(MED_FAT_CENTER, MED_FAT_SEMI) & ~lungs & ~heart
    esophagus = ((ax + 6) ** 2 + (ay - 30) ** 2 <= 7 ** 2) & body & ~heart & ~lungs

    img = np.full(shape, -1000.0, dtype=np.float32)
    img[body] = 45.0
    img[body & ~inner] = -95.0
    img[med_fat] = -85.0
    img[lungs] = -860.0 + 18.0 * smooth_noise(3.0)[lungs]
    img[heart] = 40.0
    img[esophagus] = 25.0
    img[bone] = 650.0
    img[canal] = 8.0
    img[cord] = 30.0
    for a, b, r in spec.vessels:
        vessel = tube(a, b, r) & lungs
        img[vessel] = 40.0

    radius = spec.primary_radius_mm * tp.primary_scale
    center = np.asarray(spec.primary_center)
    tumor = sphere(center, 0.8 * radius)
    for offset, frac in spec.lobules:
        tumor |= sphere(center + np.asarray(offset) * radius, frac * radius)
    tumor &= body
    img[tumor] = 35.0 + 12.0 * smooth_noise(1.5)[tumor]
    if radius > 18:
        img[sphere(center, 0.35 * radius) & tumor] = 10.0

    node = sphere(spec.node_center, tp.node_radius_mm) & body if tp.node_radius_mm > 0 else np.zeros(shape, bool)
    img[node] = 42.0
    others = np.zeros(shape, dtype=bool)
    for lesion in tp.other_lesions:
        others |= sphere(lesion.center, lesion.radius) & body
    img[others] = 38.0

    img = gaussian_filter(img, 0.7) + rng.normal(0.0, 12.0, size=shape).astype(np.float32)
    img = np.clip(img, -1024, 3071).astype(np.int16)

    lesions = np.zeros(shape, dtype=np.uint8)
    lesions[tumor] = 1
    lesions[node] = 2
    lesions[others] = 3
    structures = np.zeros(shape, dtype=np.uint8)
    structures[body] = 6
    structures[lung_l] = 1
    structures[lung_r] = 2
    structures[heart] = 3
    structures[esophagus] = 4
    structures[cord] = 5
    structures[lesions > 0] = 6

    nz, ny, nx = shape
    origin = (-(nx - 1) / 2 * spacing[0], -(ny - 1) / 2 * spacing[1], -(nz - 1) / 2 * spacing[2])
    meta = {
        "demo": True,
        "label": DEMO_LABEL,
        "generator": GENERATOR_VERSION,
        "patient_id": spec.patient_id,
        "timepoint": tp.tp_id,
        "days_from_baseline": tp.days,
        "ground_truth": {
            "primary_radius_mm": round(radius, 2),
            "node_radius_mm": round(tp.node_radius_mm, 2),
            "other_lesions": len(tp.other_lesions),
            "pose_shift_mm": list(tp.shift_mm),
            "pose_rotation_deg": tp.rotation_deg,
        },
    }
    image = Volume(img.astype(np.float32), spacing, origin, modality="CT", metadata=meta)
    return (image, Volume(lesions, spacing, origin, metadata={"classes": LESION_CLASSES}),
            Volume(structures, spacing, origin, metadata={"classes": STRUCTURE_CLASSES}))
