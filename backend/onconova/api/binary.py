"""Binary volume responses: raw little-endian arrays plus geometry headers.

Clients reconstruct values as ``raw * X-Scale + X-Offset`` with ``raw`` read in
``X-Dtype``, C order, shape ``X-Shape`` (z, y, x). ``X-Frame`` names the image grid the
voxels belong to (``case/timepoint``); a client must draw an overlay only on an image with
the same frame and geometry.
"""

from __future__ import annotations

import numpy as np
from fastapi import Response

from onconova.imaging.volume import Volume

EXPOSED_HEADERS = ["X-Shape", "X-Spacing", "X-Origin", "X-Direction", "X-Dtype", "X-Scale", "X-Offset", "X-Data-Label",
                   "X-Frame", "X-Beam-Angles", "X-Slice-Index"]


def volume_response(volume: Volume, array: np.ndarray, *, kind: str, frame: str, label: str | None = None,
                    vmax: float | None = None) -> Response:
    if kind == "image":
        if volume.modality == "CT":
            raw, scale, offset = np.clip(array, -32768, 32767).astype("<i2"), 1.0, 0.0
        else:
            lo, hi = float(array.min()), float(array.max())
            scale = (hi - lo) / 65535.0 or 1.0
            offset = lo + 32768 * scale
            raw = np.round((array - offset) / scale).clip(-32768, 32767).astype("<i2")
        dtype = "int16"
    elif kind == "label":
        raw, scale, offset, dtype = array.astype(np.uint8), 1.0, 0.0, "uint8"
    elif kind == "map":  # continuous map in [0, vmax]
        top = vmax or 1.0
        raw = np.round(np.clip(array / top, 0, 1) * 255).astype(np.uint8)
        scale, offset, dtype = top / 255.0, 0.0, "uint8"
    else:
        raise ValueError(kind)
    headers = {
        "X-Shape": ",".join(str(s) for s in raw.shape),
        "X-Spacing": ",".join(f"{s:.6g}" for s in volume.spacing),
        "X-Origin": ",".join(f"{o:.6g}" for o in volume.origin),
        "X-Direction": ",".join(f"{d:.6g}" for d in volume.direction),
        "X-Dtype": dtype,
        "X-Scale": f"{scale:.9g}",
        "X-Offset": f"{offset:.9g}",
        "X-Frame": frame,
        "Cache-Control": "no-store",
    }
    if label:
        headers["X-Data-Label"] = label
    return Response(content=np.ascontiguousarray(raw).tobytes(), media_type="application/octet-stream", headers=headers)
